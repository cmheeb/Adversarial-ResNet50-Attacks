"""
gallery.py — Visual gallery of the most extreme adversarial examples.

Reads results/fooled/, picks the N entries with the smallest fooling epsilon
(from FGSM / PGD bisect runs), re-runs each attack at the stored epsilon to
get the adversarial prediction, and renders a side-by-side grid:
  left  : original MRI scan  (true class label)
  right : adversarial        (what the model predicted instead)

Usage:
  python gallery.py                  # top 16, all Linf attacks
  python gallery.py --top 32
  python gallery.py --attack FGSM
  python gallery.py --cols 3
"""

import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torchvision.transforms as T
from tqdm import tqdm
from foolbox import PyTorchModel
from foolbox.attacks import LinfFastGradientAttack, LinfProjectedGradientDescentAttack
from pathlib import Path
from PIL import Image

BASE       = Path(__file__).parent
MODEL_PATH = BASE / "model" / "resnet50_brain_tumor.torchscript.pt"
DATA_DIR   = BASE / "data"
RESULTS    = BASE / "results"
FOOLED_DIR = RESULTS / "fooled"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 224
MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]

DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)

CLASS_COLORS = {
    "glioma":      "#f85149",
    "meningioma":  "#f0883e",
    "notumor":     "#3fb950",
    "pituitary":   "#58a6ff",
}
ATTACK_COLORS = {
    "FGSM": "#d2a8ff",
    "PGD":  "#79c0ff",
}
ATTACK_CLS = {
    "FGSM": LinfFastGradientAttack,
    "PGD":  LinfProjectedGradientDescentAttack,
}

FILENAME_RE = re.compile(r'^(.+)_(FGSM|PGD|DeepFool)_eps([0-9.]+)\.png$')

# Model
def load_model() -> PyTorchModel:
    model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
    model.eval()
    return PyTorchModel(model, bounds=(0, 1),
                        preprocessing=dict(mean=MEAN, std=STD, axis=-3),
                        device=DEVICE)

def load_tensor(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return T.ToTensor()(img).unsqueeze(0).to(DEVICE)

def predict(fmodel, image: torch.Tensor) -> tuple[str, float]:
    with torch.no_grad():
        probs = torch.softmax(fmodel(image), dim=1)[0]
    idx = probs.argmax().item()
    return CLASSES[idx], probs[idx].item()

def adversarial_prediction(fmodel, orig_path: Path, true_class: str,
                            attack_name: str, epsilon: float) -> tuple[str, float]:
    """Re-run the attack at the stored epsilon to get the adversarial prediction."""
    image      = load_tensor(orig_path)
    true_label = CLASSES.index(true_class)
    labels     = torch.tensor([true_label], device=DEVICE)
    _, advs, _ = ATTACK_CLS[attack_name]()(fmodel, image, labels, epsilons=[epsilon])
    if advs[0] is None:
        return predict(fmodel, image)
    return predict(fmodel, advs[0])

# File helpers 
def parse_fooled_dir(attacks: tuple[str, ...]) -> list[dict]:
    """
    Parse results/fooled/, keep only FGSM/PGD, then deduplicate by image stem —
    retaining whichever attack achieved the smallest epsilon for that image.
    Sorted ascending by epsilon so callers get the most fragile images first.
    """
    best: dict[str, dict] = {}   # stem → best entry so far
    for p in sorted(FOOLED_DIR.glob("*.png")):
        m = FILENAME_RE.match(p.name)
        if not m:
            continue
        stem, attack, eps_str = m.group(1), m.group(2), m.group(3)
        if attack not in attacks:
            continue
        eps = float(eps_str)
        if stem not in best or eps < best[stem]["epsilon"]:
            best[stem] = {"path": p, "stem": stem, "attack": attack, "epsilon": eps}
    return sorted(best.values(), key=lambda x: x["epsilon"])

def find_original(stem: str) -> tuple[Path | None, str | None]:
    for cls in CLASSES:
        for ext in ("jpg", "png", "jpeg"):
            p = DATA_DIR / cls / f"{stem}.{ext}"
            if p.exists():
                return p, cls
    return None, None

def load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))

def crop_adversarial(comp_path: Path) -> np.ndarray:
    """Crop the adversarial (middle) panel from the 3-panel comparison PNG."""
    img = np.array(Image.open(comp_path))
    h, w = img.shape[:2]
    x0 = max(0, w // 3 - 10)
    x1 = max(0, 2 * w // 3 - 10)
    panel = img[:, x0:x1]
    title_h = int(h * 0.24)
    return panel[title_h:, :]

# Gallery renderer
def build_gallery(entries: list[dict], fmodel: PyTorchModel,
                  cols: int = 4, out_path: Path | None = None) -> Path:
    n    = len(entries)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 4.8, rows * 3.6 + 1.4))
    fig.patch.set_facecolor("#0d1117")

    fig.text(0.5, 0.997,
             "Most Vulnerable Brain MRI Scans — Fooled at Minimum Perturbation",
             ha="center", va="top", fontsize=13, color="white", fontweight="bold")
    fig.text(0.5, 0.974,
             "Left: original  ·  Right: adversarial  (visually identical — model prediction changed)",
             ha="center", va="top", fontsize=8.5, color="#8b949e")

    gs = gridspec.GridSpec(
        rows, cols * 2,
        figure=fig,
        top=0.93, bottom=0.03,
        left=0.01, right=0.99,
        hspace=0.7, wspace=0.04,
    )

    print("Querying model for adversarial predictions ...")
    for i, entry in enumerate(tqdm(entries, unit="img", dynamic_ncols=True)):
        row, col = divmod(i, cols)
        ax_orig = fig.add_subplot(gs[row, col * 2])
        ax_adv  = fig.add_subplot(gs[row, col * 2 + 1])

        orig_path, cls = find_original(entry["stem"])
        cls_color = CLASS_COLORS.get(cls or "", "#8b949e")
        atk_color = ATTACK_COLORS.get(entry["attack"], "#e6edf3")

        # original image
        if orig_path:
            ax_orig.imshow(load_image(orig_path))
            orig_pred, orig_conf = predict(fmodel, load_tensor(orig_path))
        else:
            ax_orig.set_facecolor("#161b22")
            orig_pred, orig_conf = cls or "?", 0.0
        ax_orig.axis("off")
        ax_orig.set_title(f"{orig_pred}  {orig_conf*100:.0f}%",
                          fontsize=7.5, color=cls_color, pad=3, fontweight="bold")

        # adversarial image + model prediction
        try:
            ax_adv.imshow(crop_adversarial(entry["path"]))
        except Exception:
            ax_adv.set_facecolor("#161b22")
        ax_adv.axis("off")

        if orig_path and cls:
            adv_pred, adv_conf = adversarial_prediction(
                fmodel, orig_path, cls, entry["attack"], entry["epsilon"]
            )
        else:
            adv_pred, adv_conf = "?", 0.0

        adv_color = CLASS_COLORS.get(adv_pred, "#e6edf3")
        ax_adv.set_title(
            f"{adv_pred}  {adv_conf*100:.0f}%\n"
            f"{entry['attack']}  epsilon={entry['epsilon']:.5f}",
            fontsize=7.5, color=adv_color, pad=3,
        )

        # subtle border on each cell
        for ax in (ax_orig, ax_adv):
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("#30363d")
                spine.set_linewidth(0.6)

    # hide unused slots
    for i in range(n, rows * cols):
        row, col = divmod(i, cols)
        for sub_col in (col * 2, col * 2 + 1):
            ax = fig.add_subplot(gs[row, sub_col])
            ax.axis("off")
            ax.set_facecolor("#0d1117")

    # legend
    legend_x = 0.01
    for cls, color in CLASS_COLORS.items():
        fig.text(legend_x, 0.008, f"● {cls}", fontsize=7, color=color, va="bottom")
        legend_x += 0.1
    legend_x += 0.02
    for attack, color in ATTACK_COLORS.items():
        if any(e["attack"] == attack for e in entries):
            fig.text(legend_x, 0.008, f"■ {attack}", fontsize=7, color=color, va="bottom")
            legend_x += 0.07

    out = out_path or RESULTS / "gallery.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return out

# Main
def main(args):
    if not FOOLED_DIR.exists() or not any(FOOLED_DIR.glob("*.png")):
        print(f"No images found in {FOOLED_DIR}")
        print("Run attacks.py --bisect first to generate fooled results.")
        return

    attacks = ("FGSM", "PGD") if args.attack == "all" else (args.attack,)
    entries = parse_fooled_dir(attacks)

    if not entries:
        print(f"No fooled images found for attacks: {attacks}")
        return

    top_entries = entries[: args.top]
    print(f"Found {len(entries)} fooled images.  Rendering top {len(top_entries)} by smallest epsilon ...")

    print("Loading model ...")
    fmodel = load_model()

    out = build_gallery(top_entries, fmodel, cols=args.cols)
    print(f"Gallery saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gallery of adversarial examples from results/fooled/")
    parser.add_argument("--top",    type=int, default=16,
                        help="Number of entries to show (default: 16)")
    parser.add_argument("--cols",   type=int, default=4,
                        help="Columns in the grid (default: 4)")
    parser.add_argument("--attack", default="all",
                        choices=["all", "FGSM", "PGD"],
                        help="Which attack results to include (default: all L∞)")
    args = parser.parse_args()
    main(args)
