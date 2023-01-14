# Check Python version is at least 3.9
PYTHON_VERSION_MIN_MET=$(python3 -c 'import sys; print(sys.version_info[:2] >= (3, 9))')
if [ "$PYTHON_VERSION_MIN_MET" == "False" ]; then
  echo "Python version must be >=3.9, but is $(python3 --version 2>&1)"
  exit 1
fi

# Upgrade pip
python -m pip install --upgrade pip

# Pin dependencies
pip freeze -r requirements.txt > requirements.lock

# Code style and linting
black --check --diff .
mypy --install-types --non-interactive --show-error-codes --show-column-numbers --pretty .
flake8 .
