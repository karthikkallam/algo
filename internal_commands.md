# IMC Prosperity Trading Algorithm

This repository contains an algorithmic trading strategy for the IMC Prosperity challenge. The strategy implements market making and statistical arbitrage techniques for various products.

1. First activate the virtual environment:
   python -m venv venv && source venv/bin/activate

2. Run the parameter optimizer:
   python optimize_trader.py

3. Run the Bayesian optimizer:
   python bayesian_optimizer.py

4. Run basket analysis:
   python basket_analyzer.py --data_dir data --threshold 1.0

5. Run test runner:
   python test_runner.py

6. View optimization results dashboard:
   optuna-dashboard sqlite:///prosperity3_optimization.db
   optuna-dashboard sqlite:///prosperity3_round2_optimization.db

7. Run backtester:
   prosperity3bt trader.py 0 --merge-pnl

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
