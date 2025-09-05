# scripts/setup/install_dependencies.sh

#!/usr/bin/env bash
# Install Virtue dependencies in activated venv

echo "Installing core dependencies..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

echo "Dependencies installed."
