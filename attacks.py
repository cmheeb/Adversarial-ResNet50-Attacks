"""
Adversarial attacks on ResNet50 brain tumor classifier using Foolbox.

Attacks implemented:
  - FGSM   (Fast Gradient Sign Method)      — Linf, single step
  - PGD    (Projected Gradient Descent)      — Linf, iterative
  - DeepFool                                 — L2,  minimal perturbation

Modes:
  (default)  Fixed epsilon grid: [0.01, 0.03, 0.05] for Linf, [0.5, 1.0, 2.0] for L2.
  --bisect   Binary-search down from eps-max to find minimum fooling epsilon.
  --sweep    Step up from eps-start in eps-step increments; stop the moment
             an image is fooled and save the result.
"""

import argparse
import shutil
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from tqdm import tqdm
from foolbox import PyTorchModel
from foolbox.attacks import (
    LinfFastGradientAttack,
    LinfProjectedGradientDescentAttack,
    L2DeepFoolAttack,
)
from PIL import Image
from pathlib import Path

# Config
# obtain path to work from any directory
BASE       = Path(__file__).parent
MODEL_PATH = BASE / "model" / "resnet50_brain_tumor.torchscript.pt"
DATA_DIR   = BASE / "data"
RESULTS    = BASE / "results"
FOOLED_DIR = RESULTS / "fooled"   # attacks that misclassify the image stored here

# Classifications
CLASSES  = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 224   # ResNet50 expects 224×224 input

# ImageNet normalisation constants
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Prefer Apple Silicon GPU (mps) → CUDA → CPU, in that order.
DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)

# Only L infinite attacks (FGSM, PGD) are compatible with bisect/sweep modes since
# their epsilon directly controls the maximum per-pixel change.  DeepFool
# minimises the L2 norm and already finds the smallest perturbation by design
LINF_ATTACKS = {
    "FGSM": LinfFastGradientAttack,
    "PGD":  LinfProjectedGradientDescentAttack,
}

# Default epsilon grids used in fixed-epsilon mode
ATTACKS_CFG = {
    "FGSM":     (LinfFastGradientAttack,             [0.01, 0.03, 0.05]),
    "PGD":      (LinfProjectedGradientDescentAttack, [0.01, 0.03, 0.05]),
    "DeepFool": (L2DeepFoolAttack,                   [0.5,  1.0,  2.0 ]),
}

# Bisect mode: stop refining when the interval is smaller than this tolerance
# 20 iterations gives precision of eps_max / 2^20
BISECT_TOL      = 1e-4
BISECT_MAX_ITER = 20

# Sweep mode default range/step (--eps-start, --eps-step, --eps-max)
SWEEP_EPS_START_DEFAULT = 0.00001
SWEEP_EPS_STEP_DEFAULT  = 0.0001
SWEEP_EPS_MAX_DEFAULT   = 0.005


# Load model
def load_model() -> PyTorchModel:
    """
    Load the TorchScript brain-tumor classifier and wrap it in a Foolbox
    PyTorchModel

    The Foolbox wrapper handles ImageNet normalisation internally so that all
    attack algorithms operate in the raw [0, 1] pixel space — no manual
    pre/post-processing is needed inside the attack loop
    """
    model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
    model.eval()
    return PyTorchModel(
        model,
        bounds=(0, 1),
        preprocessing=dict(mean=MEAN, std=STD, axis=-3),
        device=DEVICE,
    )


# Image utilities
def load_image(path: Path) -> torch.Tensor:
    """
    Open an image file, resize to the model's expected input size, and convert
    it to a (1, 3, H, W) float tensor in the [0, 1] range ready for Foolbox
    """
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return T.ToTensor()(img).unsqueeze(0).to(DEVICE)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """
    Convert a (1, 3, H, W) float tensor back to a PIL Image for display.
    Values are clipped to [0, 1] before scaling to uint8 to handle any minor
    floating-point overshoot introduced by the attack
    """
    arr = tensor.squeeze(0).cpu().numpy()
    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)
    return Image.fromarray(arr)


def predict(fmodel: PyTorchModel, image: torch.Tensor) -> tuple[str, float]:
    """
    Run a single forward pass and return the top predicted class name and its
    softmax confidence score.  torch.no_grad() is used to skip gradient
    computation
    """
    with torch.no_grad():
        probs = torch.softmax(fmodel(image), dim=1)[0]
    idx = probs.argmax().item()
    return CLASSES[idx], probs[idx].item()


# Binary search for minimum fooling epsilon
def bisect_epsilon(fmodel: PyTorchModel, attack_cls,
                   image: torch.Tensor, true_label: int, orig_label: str,
                   eps_max: float) -> tuple[float | None, torch.Tensor | None]:
    """
    Find the smallest Linf epsilon within (0, eps_max] that causes the model to
    misclassify the image, using binary search.

    The search starts by confirming that eps_max itself fools the model.  If it
    does not, (None, None) is returned immediately without further iterations.
    Otherwise the interval [lo, hi] is halved up to BISECT_MAX_ITER times,
    always keeping 'hi' as the last known fooling epsilon.  Convergence is
    declared when hi - lo < BISECT_TOL (~0.0001).

    Returns (min_eps, adversarial_tensor) on success, (None, None) otherwise.
    """
    labels = torch.tensor([true_label], device=DEVICE)

    # Verify the upper bound can actually fool the model before committing to
    # the full search — saves up to BISECT_MAX_ITER calls if it cannot
    _, advs, _ = attack_cls()(fmodel, image, labels, epsilons=[eps_max])
    if advs[0] is None or predict(fmodel, advs[0])[0] == orig_label:
        return None, None

    lo, hi = 0.0, eps_max
    best_adv, best_eps = advs[0], eps_max

    for _ in range(BISECT_MAX_ITER):
        if hi - lo < BISECT_TOL:
            break
        mid = (lo + hi) / 2.0
        _, advs_mid, _ = attack_cls()(fmodel, image, labels, epsilons=[mid])
        adv_mid = advs_mid[0]
        if adv_mid is not None and predict(fmodel, adv_mid)[0] != orig_label:
            # Midpoint fools the model — tighten the upper bound
            hi, best_adv, best_eps = mid, adv_mid, mid
        else:
            # Midpoint did not fool — raise the lower bound
            lo = mid

    return best_eps, best_adv


# Linear sweep for minimum fooling epsilon
def sweep_epsilon(fmodel: PyTorchModel, attack_cls,
                  image: torch.Tensor, true_label: int, orig_label: str,
                  eps_start: float, eps_step: float,
                  eps_max: float) -> tuple[float | None, torch.Tensor | None]:
    """
    Find the minimum fooling epsilon by sweeping upward from eps_start in
    eps_step increments and stopping at the first value that causes
    misclassification.

    All epsilons are passed to the attack in a single Foolbox call.
    For FGSM - the gradient is computed once and each epsilon is just a different
    scalar multiple of the sign gradient.  For PGD, Foolbox runs one optimisation 
    per epsilon, but batching avoids repeated model-loading overhead.

    Images that are not fooled at any epsilon up to eps_max return (None, None)
    and are not saved, keeping the results clean.
    """
    # Build the epsilon list once, np.round avoids floating-point drift that
    # could silently skip or repeat values near machine epsilon boundaries
    epsilons = list(np.round(
        np.arange(eps_start, eps_max + eps_step * 0.01, eps_step), 8
    ))
    if not epsilons:
        return None, None

    labels = torch.tensor([true_label], device=DEVICE)
    _, advs, _ = attack_cls()(fmodel, image, labels, epsilons=epsilons)

    # Iterate from smallest to largest, return the first adversarial that
    # changed the predicted class
    for eps, adv in zip(epsilons, advs):
        if adv is not None and predict(fmodel, adv)[0] != orig_label:
            return float(eps), adv

    return None, None


# Visualisation
def save_comparison(original: torch.Tensor, adversarial: torch.Tensor,
                    orig_pred: tuple, adv_pred: tuple,
                    attack_name: str, epsilon: float,
                    img_name: str, l2_norm: float, fooled: bool) -> Path:
    """
    Save a three-panel comparison image for a single adversarial example:
      Panel 1 — original image with its predicted class and confidence.
      Panel 2 — adversarial image with the new (potentially wrong) prediction.
      Panel 3 — per-pixel perturbation magnitude visualised as a heat map.
                Brighter areas indicate where the attack concentrated its changes.

    If the image was fooled (prediction changed), the file is moved from
    results/ into results/fooled/
    """
    RESULTS.mkdir(exist_ok=True)
    FOOLED_DIR.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: original scan
    axes[0].imshow(tensor_to_pil(original))
    axes[0].set_title(f"Original\n{orig_pred[0]}  ({orig_pred[1]*100:.1f}%)", fontsize=11)
    axes[0].axis("off")

    # Panel 2: adversarial scan — visually identical but potentially misclassified
    axes[1].imshow(tensor_to_pil(adversarial))
    axes[1].set_title(
        f"Adversarial [{attack_name}]\n{adv_pred[0]}  ({adv_pred[1]*100:.1f}%)\nepsilon={epsilon:.5f}",
        fontsize=11,
    )
    axes[1].axis("off")

    # Panel 3: absolute pixel-level difference averaged across colour channels,
    # normalised to [0, 1] and displayed with a 'hot' colour map so that
    # brighter regions show where the perturbation is strongest
    diff = (adversarial - original).squeeze(0).cpu().numpy()
    diff_vis = np.abs(diff).mean(axis=0)
    diff_vis = (diff_vis - diff_vis.min()) / (diff_vis.max() - diff_vis.min() + 1e-8)
    im = axes[2].imshow(diff_vis, cmap="hot")
    axes[2].set_title(f"Perturbation\nL2 norm: {l2_norm:.4f}", fontsize=11)
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046)

    plt.suptitle(f"Adversarial Attack: {attack_name} | Image: {img_name}", fontsize=13)
    plt.tight_layout()

    fname = RESULTS / f"{img_name}_{attack_name}_eps{epsilon:.5f}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()

    # move to the dedicated fooled subfolder if the prediction changed
    if fooled:
        dest = FOOLED_DIR / fname.name
        shutil.move(str(fname), dest)
        return dest
    return fname


# Collect images
def collect_images(data_dir: Path) -> list[tuple[Path, str]]:
    """
    Walk each class subdirectory under data_dir and return a list of
    (image_path, class_name) tuples.  The class name is taken from the
    directory name, which must match one of the entries in CLASSES.
    Both .jpg and .png files are included.
    """
    images = []
    for cls in CLASSES:
        class_dir = data_dir / cls
        if class_dir.is_dir():
            for p in sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png")):
                images.append((p, cls))
    return images


# Attack one image (standard fixed-epsilon mode)
def attack_image(fmodel: PyTorchModel, img_path: Path, true_label_name: str,
                 attacks_cfg: dict) -> list[dict]:
    """
    Run every configured attack at every epsilon in its fixed grid and record
    the result for each (attack, epsilon) combination.

    Both fooled and non-fooled results are saved as comparison images so that
    the effect of each epsilon level can be inspected visually.  Fooled images
    are moved to results/fooled/.

    Returns a list of result dicts, one per (attack, epsilon) pair.
    """
    image      = load_image(img_path)
    img_name   = img_path.stem
    true_label = CLASSES.index(true_label_name)
    orig_pred  = predict(fmodel, image)
    results    = []

    for name, (attack_cls, epsilons) in attacks_cfg.items():
        attack = attack_cls()
        labels = torch.tensor([true_label], device=DEVICE)
        # pass the full epsilon list
        _, advs, success = attack(fmodel, image, labels, epsilons=epsilons)

        for eps, adv, suc in zip(epsilons, advs, success):
            if adv is None:
                continue
            adv_pred = predict(fmodel, adv)
            l2_norm  = (adv - image).norm(p=2).item()
            fooled   = adv_pred[0] != orig_pred[0]
            save_comparison(image, adv, orig_pred, adv_pred,
                            name, eps, img_name, l2_norm, fooled)
            results.append({
                "image": img_name, "true": true_label_name,
                "attack": name, "epsilon": eps,
                "original": orig_pred[0], "adversarial": adv_pred[0],
                "fooled": fooled, "l2_norm": l2_norm,
            })
    return results


# Attack one image (bisect mode)
def attack_image_bisect(fmodel: PyTorchModel, img_path: Path,
                        true_label_name: str, attacks_cfg: dict,
                        eps_max: float) -> list[dict]:
    """
    Run each Linf attack using binary search to find the minimum fooling epsilon
    for this specific image (see bisect_epsilon).  DeepFool is run at its
    standard epsilon grid since it already computes the true minimum L2
    perturbation internally.

    Only the single adversarial at the minimum epsilon is saved per attack,
    reducing disk usage compared to the fixed-epsilon mode.  Images that cannot be
    fooled within eps_max are recorded with fooled=False and nothing is saved.
    """
    image      = load_image(img_path)
    img_name   = img_path.stem
    true_label = CLASSES.index(true_label_name)
    orig_pred  = predict(fmodel, image)
    results    = []

    for name, (attack_cls, epsilons) in attacks_cfg.items():
        if name in LINF_ATTACKS:
            min_eps, adv = bisect_epsilon(
                fmodel, attack_cls, image, true_label, orig_pred[0], eps_max
            )
            if adv is None:
                # record the failure so the summary fool-rate is accurate
                results.append({
                    "image": img_name, "true": true_label_name,
                    "attack": name, "epsilon": None,
                    "original": orig_pred[0], "adversarial": None,
                    "fooled": False, "l2_norm": None,
                })
                continue
            adv_pred = predict(fmodel, adv)
            l2_norm  = (adv - image).norm(p=2).item()
            save_comparison(image, adv, orig_pred, adv_pred,
                            name, min_eps, img_name, l2_norm, fooled=True)
            results.append({
                "image": img_name, "true": true_label_name,
                "attack": name, "epsilon": min_eps,
                "original": orig_pred[0], "adversarial": adv_pred[0],
                "fooled": True, "l2_norm": l2_norm,
            })
        else:
            # DeepFool finds the minimal L2 perturbation on its own — run it
            # at the smallest listed epsilon and store the actual L2 norm.
            attack = attack_cls()
            labels = torch.tensor([true_label], device=DEVICE)
            _, advs, _ = attack(fmodel, image, labels, epsilons=epsilons)
            adv = advs[0]
            if adv is None:
                continue
            adv_pred = predict(fmodel, adv)
            l2_norm  = (adv - image).norm(p=2).item()
            fooled   = adv_pred[0] != orig_pred[0]
            save_comparison(image, adv, orig_pred, adv_pred,
                            name, epsilons[0], img_name, l2_norm, fooled)
            results.append({
                "image": img_name, "true": true_label_name,
                "attack": name, "epsilon": l2_norm,
                "original": orig_pred[0], "adversarial": adv_pred[0],
                "fooled": fooled, "l2_norm": l2_norm,
            })
    return results


# Attack one image (sweep mode)
def attack_image_sweep(fmodel: PyTorchModel, img_path: Path,
                       true_label_name: str, attacks_cfg: dict,
                       eps_start: float, eps_step: float,
                       eps_max: float) -> list[dict]:
    """
    Run each Linf attack using a fine-grained upward sweep to find the minimum
    fooling epsilon (see sweep_epsilon).  Unlike bisect mode, the sweep starts
    from near-zero and walks upward

    DeepFool is run normally and its result is only saved if it fools the model,
    consistent with the sweep philosophy of saving only successful attacks.
    """
    image      = load_image(img_path)
    img_name   = img_path.stem
    true_label = CLASSES.index(true_label_name)
    orig_pred  = predict(fmodel, image)
    results    = []

    for name, (attack_cls, epsilons) in attacks_cfg.items():
        if name in LINF_ATTACKS:
            min_eps, adv = sweep_epsilon(
                fmodel, attack_cls, image, true_label, orig_pred[0],
                eps_start, eps_step, eps_max,
            )
            # image is robust within the ceiling — record the miss for stats
            if adv is None:
                results.append({
                    "image": img_name, "true": true_label_name,
                    "attack": name, "epsilon": None,
                    "original": orig_pred[0], "adversarial": None,
                    "fooled": False, "l2_norm": None,
                })
                continue
            adv_pred = predict(fmodel, adv)
            l2_norm  = (adv - image).norm(p=2).item()
            save_comparison(image, adv, orig_pred, adv_pred,
                            name, min_eps, img_name, l2_norm, fooled=True)
            results.append({
                "image": img_name, "true": true_label_name,
                "attack": name, "epsilon": min_eps,
                "original": orig_pred[0], "adversarial": adv_pred[0],
                "fooled": True, "l2_norm": l2_norm,
            })
        else:
            # DeepFool: already finds the minimal L2 perturbation.  Only save
            # if it successfully changes the prediction.
            attack = attack_cls()
            labels = torch.tensor([true_label], device=DEVICE)
            _, advs, _ = attack(fmodel, image, labels, epsilons=epsilons)
            adv = advs[0]
            if adv is None:
                continue
            adv_pred = predict(fmodel, adv)
            l2_norm  = (adv - image).norm(p=2).item()
            fooled   = adv_pred[0] != orig_pred[0]
            if fooled:
                save_comparison(image, adv, orig_pred, adv_pred,
                                name, l2_norm, img_name, l2_norm, fooled=True)
            results.append({
                "image": img_name, "true": true_label_name,
                "attack": name, "epsilon": l2_norm,
                "original": orig_pred[0], "adversarial": adv_pred[0],
                "fooled": fooled, "l2_norm": l2_norm,
            })
    return results


# Summary printers
def print_summary_fixed(all_results: list[dict]) -> None:
    """
    Print a per-(attack, epsilon) summary table for fixed-epsilon mode, showing the
    number and percentage of images fooled and the average L2 perturbation norm
    at each epsilon level.
    """
    print(f"\n{'='*70}")
    print(f"{'Attack':<12} {'epsilon':>6}  {'Fooled':>8}  {'Total':>7}  {'Rate':>7}  {'Avg L2':>8}")
    print(f"{'-'*70}")
    groups = defaultdict(list)
    for r in all_results:
        groups[(r["attack"], r["epsilon"])].append(r)
    for (atk, eps), rows in sorted(groups.items()):
        n_fooled = sum(1 for r in rows if r["fooled"])
        avg_l2   = sum(r["l2_norm"] for r in rows) / len(rows)
        rate     = n_fooled / len(rows) * 100
        print(f"{atk:<12} {eps:>6}  {n_fooled:>8}  {len(rows):>7}  {rate:>6.1f}%  {avg_l2:>8.4f}")
    print(f"{'='*70}")


def print_summary_bisect(all_results: list[dict]) -> None:
    """
    Print a per-attack summary table for bisect and sweep modes.  Because
    epsilon varies per image in these modes, the table shows the distribution
    of minimum fooling epsilons (min, median, max) and the average L2 norm
    across all fooled images, giving a sense of how close the dataset sits to
    the model's decision boundaries.
    """
    print(f"\n{'='*80}")
    print(f"{'Attack':<12}  {'Fooled':>8}  {'Total':>7}  {'Rate':>7}  "
          f"{'Min epsilon':>8}  {'Med epsilon':>8}  {'Max epsilon':>8}  {'Avg L2':>8}")
    print(f"{'-'*80}")
    groups = defaultdict(list)
    for r in all_results:
        groups[r["attack"]].append(r)
    for atk, rows in sorted(groups.items()):
        fooled_rows = [r for r in rows if r["fooled"]]
        n_fooled    = len(fooled_rows)
        rate        = n_fooled / len(rows) * 100
        if fooled_rows:
            epsilons = [r["epsilon"] for r in fooled_rows if r["epsilon"] is not None]
            l2s      = [r["l2_norm"] for r in fooled_rows if r["l2_norm"] is not None]
            min_e, med_e, max_e = min(epsilons), float(np.median(epsilons)), max(epsilons)
            avg_l2 = sum(l2s) / len(l2s)
            print(f"{atk:<12}  {n_fooled:>8}  {len(rows):>7}  {rate:>6.1f}%  "
                  f"{min_e:>8.5f}  {med_e:>8.5f}  {max_e:>8.5f}  {avg_l2:>8.4f}")
        else:
            print(f"{atk:<12}  {n_fooled:>8}  {len(rows):>7}  {rate:>6.1f}%  "
                  f"{'—':>8}  {'—':>8}  {'—':>8}  {'—':>8}")
    print(f"{'='*80}")


# Main
def main(args) -> None:
    """
    Loads the model, builds the image list (single image or full dataset), 
    then iterates with a tqdm progress bar calling the appropriate
    per-image attack function depending on the selected mode.  
    Prints a summary table when complete.
    """
    RESULTS.mkdir(exist_ok=True)
    FOOLED_DIR.mkdir(exist_ok=True)

    print(f"Device : {DEVICE}")
    print(f"Model  : {MODEL_PATH.name}")
    if args.sweep:
        n_steps = int((args.eps_max - args.eps_start) / args.eps_step) + 1
        print(f"Mode   : sweep  (start={args.eps_start}, step={args.eps_step}, "
              f"max={args.eps_max}, ~{n_steps} steps/image)\n")
    elif args.bisect:
        print(f"Mode   : bisect  (eps_max={args.eps_max}, tol={BISECT_TOL}, "
              f"max_iter={BISECT_MAX_ITER})\n")
    else:
        print(f"Mode   : fixed-epsilon\n")

    fmodel = load_model()

    # filter the attack config to only the attack(s) requested on the CLI
    attacks_cfg = {k: v for k, v in ATTACKS_CFG.items()
                   if args.attack == "all" or k == args.attack}

    # Build the list of (image_path, class_label) pairs to process
    # A single --image argument overrides the full dataset scan
    if args.image:
        img_path = Path(args.image)
        # Infer the label from the filename stem or parent directory name,
        # falling back to --label if neither matches a known class
        label = img_path.stem if img_path.stem in CLASSES else img_path.parent.name
        if label not in CLASSES:
            label = args.label
        image_list = [(img_path, label)]
    else:
        image_list = collect_images(DATA_DIR)

    print(f"Images : {len(image_list)}")
    print(f"Attacks: {', '.join(attacks_cfg)}")
    if args.sweep or args.bisect:
        linf  = [k for k in attacks_cfg if k in LINF_ATTACKS]
        other = [k for k in attacks_cfg if k not in LINF_ATTACKS]
        mode_label = "sweep" if args.sweep else "bisect"
        print(f"         L∞ {mode_label}: {linf}  |  minimal-norm as-is: {other}")
    print()

    all_results  = []
    fooled_count = 0   # running total shown in the progress bar postfix

    with tqdm(total=len(image_list), unit="img", dynamic_ncols=True) as pbar:
        for img_path, true_label_name in image_list:
            if args.sweep:
                results = attack_image_sweep(
                    fmodel, img_path, true_label_name, attacks_cfg,
                    args.eps_start, args.eps_step, args.eps_max,
                )
            elif args.bisect:
                results = attack_image_bisect(
                    fmodel, img_path, true_label_name, attacks_cfg, args.eps_max
                )
            else:
                results = attack_image(fmodel, img_path, true_label_name, attacks_cfg)

            for r in results:
                if r["fooled"]:
                    fooled_count += 1
            all_results.extend(results)
            pbar.set_postfix(fooled=fooled_count)
            pbar.update(1)

    # print the appropriate summary table for the chosen mode
    if args.sweep or args.bisect:
        print_summary_bisect(all_results)
    else:
        print_summary_fixed(all_results)

    n_total  = len(all_results)
    n_fooled = sum(1 for r in all_results if r["fooled"])
    print(f"\nOverall: {n_fooled}/{n_total} fooled ({n_fooled/n_total*100:.1f}%)")
    print(f"Results  → {RESULTS}/")
    print(f"Fooled   → {FOOLED_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial attacks on brain tumor classifier")
    parser.add_argument("--image",     default=None,
                        help="Single image path (omit to run on full data/ dataset)")
    parser.add_argument("--label",     default="glioma", choices=CLASSES,
                        help="True class label (fallback if not inferrable from path)")
    parser.add_argument("--attack",    default="all",
                        choices=["all", "FGSM", "PGD", "DeepFool"],
                        help="Which attack to run (default: all)")
    parser.add_argument("--bisect",    action="store_true",
                        help="Binary-search minimum fooling epsilon per image for L∞ attacks")
    parser.add_argument("--sweep",     action="store_true",
                        help="Step up from eps-start in eps-step increments; only save "
                             "an image if it is fooled within eps-max")
    parser.add_argument("--eps-start", type=float, default=SWEEP_EPS_START_DEFAULT,
                        help=f"Sweep start epsilon (default: {SWEEP_EPS_START_DEFAULT})")
    parser.add_argument("--eps-step",  type=float, default=SWEEP_EPS_STEP_DEFAULT,
                        help=f"Sweep increment (default: {SWEEP_EPS_STEP_DEFAULT})")
    parser.add_argument("--eps-max",   type=float, default=SWEEP_EPS_MAX_DEFAULT,
                        help=f"Ceiling epsilon for sweep/bisect (default: {SWEEP_EPS_MAX_DEFAULT})")
    args = parser.parse_args()
    if args.sweep and args.bisect:
        parser.error("--sweep and --bisect are mutually exclusive")
    main(args)
