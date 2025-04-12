#!/bin/bash
# check_environment.sh - Script to check requirements for running bayesian_optimizer.py

echo "===== IMC Prosperity 3 Environment Check ====="
echo ""

# Check Python version
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON="python3"
    echo "Found $(python3 --version)"
elif command -v python &> /dev/null; then
    PYTHON="python"
    echo "Found $(python --version)"
else
    echo "ERROR: Python not found! Please install Python 3.6+"
    exit 1
fi

# Check for prosperity3bt
echo ""
echo "Checking prosperity3bt..."
if command -v prosperity3bt &> /dev/null; then
    echo "Found prosperity3bt: $(prosperity3bt --version 2>&1 | head -n 1)"
    echo "Located at: $(which prosperity3bt)"
else
    echo "ERROR: prosperity3bt not found in PATH!"
    echo "Make sure you have activated the correct environment or installed prosperity3bt."
    echo "Without prosperity3bt, the optimizer cannot run backtests."
    exit 1
fi

# Check Python packages
echo ""
echo "Checking required Python packages..."
PACKAGES=("optuna" "numpy" "pandas" "jsonpickle")

for package in "${PACKAGES[@]}"; do
    echo -n "Checking for $package... "
    $PYTHON -c "import $package" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "OK"
    else
        echo "MISSING"
        echo "  Please install with: pip install $package"
        MISSING_PKGS=1
    fi
done

# Check for required files
echo ""
echo "Checking for required files..."
FILES=("trader.py" "datamodel.py" "bayesian_optimizer.py")

for file in "${FILES[@]}"; do
    echo -n "Checking for $file... "
    if [ -f "$file" ]; then
        echo "OK"
    else
        echo "MISSING"
        echo "  ERROR: Required file $file not found!"
        MISSING_FILES=1
    fi
done

# Check data directory and files
echo ""
echo "Checking data directory..."
if [ -d "data" ]; then
    echo "Found data directory"
    
    echo "Checking for Round 2 data files..."
    DATA_FILES=("prices_round_2_day_0.csv" "trades_round_2_day_0.csv")
    
    for file in "${DATA_FILES[@]}"; do
        echo -n "Checking for data/$file... "
        if [ -f "data/$file" ]; then
            echo "OK"
        else
            echo "MISSING"
            echo "  WARNING: Data file data/$file not found!"
            echo "  Make sure you have downloaded the required data files for Round 2."
            MISSING_DATA=1
        fi
    done
else
    echo "WARNING: data directory not found!"
    echo "You need to create a data directory with required files:"
    echo "  mkdir -p data"
    echo "  # Then download/copy the required data files into the data directory"
    MISSING_DATA=1
fi

# Final summary
echo ""
echo "===== Summary ====="
if [ -n "$MISSING_PKGS" ] || [ -n "$MISSING_FILES" ] || [ -n "$MISSING_DATA" ]; then
    echo "There are issues that need to be addressed before running the optimizer."
    
    if [ -n "$MISSING_PKGS" ]; then
        echo "- Missing Python packages: Install with pip"
    fi
    
    if [ -n "$MISSING_FILES" ]; then
        echo "- Missing required source files: Make sure they exist in the current directory"
    fi
    
    if [ -n "$MISSING_DATA" ]; then
        echo "- Missing data files: Make sure to download/copy the data files to the data directory"
    fi
else
    echo "All checks passed! You should be able to run the Bayesian optimizer."
    echo "To run: python bayesian_optimizer.py"
fi