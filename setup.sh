#!/bin/bash
set -e

echo "=== Scorch Development Setup ==="
echo

# Determine script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for macOS and required dependencies
if [[ "$(uname)" == "Darwin" ]]; then
    echo "Detected macOS"

    # Check for libomp (required for OpenMP support)
    if [ ! -d "/opt/homebrew/opt/libomp" ] && [ ! -d "/usr/local/opt/libomp" ]; then
        echo
        echo "Warning: libomp not found. OpenMP support requires libomp."
        echo "Install it with: brew install libomp"
        echo
    fi

    # On macOS, we MUST use system clang (not Homebrew's LLVM) for torch compatibility
    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++
    echo "Using system clang for C++ compilation"
fi

# Check if conda is available
if command -v conda &> /dev/null; then
    echo
    echo "Conda detected. Using conda environment for better PyTorch compatibility..."

    # Initialize conda for this shell
    eval "$(conda shell.bash hook)"

    # Create conda environment if it doesn't exist
    if ! conda env list | grep -q "^scorch "; then
        echo "Creating conda environment 'scorch'..."
        # Use Python 3.11 which has best PyTorch compatibility
        conda create -y -n scorch python=3.11
    fi

    # Activate conda environment
    echo "Activating conda environment..."
    conda activate scorch

    # Install PyTorch via conda for better macOS compatibility
    echo
    echo "Installing PyTorch via conda..."
    conda install -y pytorch -c pytorch

    # Install other dependencies via pip
    # Note: numpy<2 required for compatibility with conda pytorch
    echo
    echo "Installing other dependencies..."
    pip install "numpy<2" scipy ninja black flake8 mypy pytest

    # Set up environment variables for torch C++ extensions
    # This ensures CC/CXX are set every time the environment is activated
    if [[ "$(uname)" == "Darwin" ]]; then
        ACTIVATE_DIR="$CONDA_PREFIX/etc/conda/activate.d"
        DEACTIVATE_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"
        mkdir -p "$ACTIVATE_DIR" "$DEACTIVATE_DIR"

        cat > "$ACTIVATE_DIR/scorch_env.sh" << 'ACTIVATE_EOF'
#!/bin/bash
# Use system clang for torch C++ extensions (avoid Homebrew LLVM incompatibilities)
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
ACTIVATE_EOF

        cat > "$DEACTIVATE_DIR/scorch_env.sh" << 'DEACTIVATE_EOF'
#!/bin/bash
# Restore CC/CXX to defaults
unset CC
unset CXX
DEACTIVATE_EOF
        echo "Configured environment to use system clang"
    fi

    ENV_ACTIVATE_CMD="conda activate scorch"
else
    echo
    echo "Conda not found. Using virtual environment..."
    echo "Note: For best macOS compatibility, consider installing miniconda."

    # Check Python version
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    PYTHON_VERSION_OK=$(python3 -c 'import sys; print(sys.version_info[:2] >= (3, 9))')

    if [ "$PYTHON_VERSION_OK" == "False" ]; then
        echo "Error: Python >= 3.9 required, but found Python $PYTHON_VERSION"
        exit 1
    fi
    echo "Python version: $PYTHON_VERSION"

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

    ENV_ACTIVATE_CMD="source venv/bin/activate"
fi

# Install scorch in editable/develop mode
# Use --no-build-isolation since setup.py imports torch at the top level
echo
echo "Installing scorch in develop mode..."
pip install -e . --no-build-isolation

# Verify installation
echo
echo "Verifying installation..."
python3 -c "import scorch; print('scorch imported successfully')"

echo
echo "=== Setup complete! ==="
echo
echo "To activate the environment in the future, run:"
echo "  $ENV_ACTIVATE_CMD"
echo
echo "To run tests:"
echo "  pytest"
echo
echo "To run linting/formatting checks:"
echo "  bash pre-commit.sh"
