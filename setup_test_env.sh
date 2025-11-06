#!/bin/bash

# Setup script for testing docstrange changes in isolated environment

echo "Setting up test environment for docstrange..."
echo "=============================================="

# Activate test virtual environment
echo "Activating test virtual environment..."
source .venv_test/bin/activate.fish 2>/dev/null || source .venv_test/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install docstrange in editable mode
echo "Installing docstrange in editable mode..."
pip install -e .

# Install test dependencies if needed
echo "Installing test dependencies..."
pip install pytest pillow

echo ""
echo "✓ Test environment setup complete!"
echo ""
echo "To use this environment:"
echo "  Fish shell: source .venv_test/bin/activate.fish"
echo "  Bash/Zsh:   source .venv_test/bin/activate"
echo ""
echo "To run tests:"
echo "  python test_json_schema.py"
echo ""
