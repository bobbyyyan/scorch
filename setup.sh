#!/bin/bash
set -e

echo "=== Scorch Development Setup ==="
echo

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
PYTHON_VERSION_OK=$(python3 -c 'import sys; print(sys.version_info[:2] >= (3, 9))')

if [ "$PYTHON_VERSION_OK" == "False" ]; then
    echo "Error: Python >= 3.9 required, but found Python $PYTHON_VERSION"
    exit 1
fi
echo "Python version: $PYTHON_VERSION"

# Determine script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo
echo "Installing dependencies..."
pip install -r requirements.txt

# Install scorch in editable/develop mode
# Use setup.py develop since setup.py imports torch at the top level
echo
echo "Installing scorch in develop mode..."

# On macOS, set C++ include path to find standard library headers
if [[ "$(uname)" == "Darwin" ]]; then
    export CPLUS_INCLUDE_PATH="/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1"
fi

python setup.py develop

# Verify installation
echo
echo "Verifying installation..."
python3 -c "import scorch; print(f'scorch {scorch.__version__ if hasattr(scorch, \"__version__\") else \"(installed)\"}')"

echo
echo "=== Setup complete! ==="
echo
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo
echo "To run tests:"
echo "  pytest"
echo
echo "To run linting/formatting checks:"
echo "  bash pre-commit.sh"
