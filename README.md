# IMC Prosperity Trading Algorithm

This repository contains an algorithmic trading strategy for the IMC Prosperity challenge. The strategy implements market making and statistical arbitrage techniques for various products.

## Project Structure

- `trader.py`: Main trading algorithm with market making logic
- `datamodel.py`: Data structures and models used by the trading algorithm
- `test_runner.py`: Local testing harness for the trading algorithm
- `optimize_trader.py`: Parameter optimization using Optuna
- `bayesian_optimizer.py`: Bayesian optimization for Round 2 parameters
- `basket_analyzer.py`: Tool to analyze arbitrage opportunities in baskets
## Setup

1. Create and activate the conda environment:
```bash
conda create -n imc_prosperity python=3.8 -y
conda activate imc_prosperity
```

2. Install required packages:
```bash
conda install -y numpy pandas matplotlib plotly
conda install -y -c conda-forge jsonpickle optuna typing-extensions
```

3. For easy activation, use the provided script:
```bash
source activate_env.sh
```

## Running the Code

1. Run the local test harness:
```bash
python test_runner.py
```

2. Optimize parameters:
```bash
python optimize_trader.py
```

3. Analyze basket arbitrage opportunities:
```bash
python basket_analyzer.py --data_dir data --threshold 1.0
```

## Strategy Overview

The trading algorithm implements several strategies:

1. **Market Making**: Places bid and ask orders around a fair value, with spreads adjusted based on volatility.
2. **Statistical Arbitrage**: Captures temporary price deviations from fair value.
3. **Mean Reversion**: For RAINFOREST_RESIN, trades around an anchor price.
4. **Order Book Imbalance**: Adjusts fair value based on order book imbalance.
5. **Inventory Management**: Skews orders to reduce inventory risk.

Parameters are tuned using Bayesian optimization with Optuna.
