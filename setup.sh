#!/usr/bin/env bash
# Setup script — creates venv and installs all dependencies.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Creating virtual environment..."
python3 -m venv venv

echo "Installing dependencies..."
venv/bin/pip install --upgrade pip --quiet
venv/bin/pip install -r requirements.txt --quiet

echo ""
echo "Setup complete. To run attacks:"
echo "  source venv/bin/activate"
echo "  python attacks.py --image data/samples/glioma.jpg --attack all"
echo ""
echo "Options:"
echo "  --image   path to any .jpg/.png brain MRI image"
echo "  --label   true class: glioma | meningioma | notumor | pituitary"
echo "  --attack  FGSM | PGD | DeepFool | CW_L2 | all"
