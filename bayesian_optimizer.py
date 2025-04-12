# bayesian_optimizer.py
# Bayesian Optimization for IMC Prosperity 3 Round 2 Trader

import optuna
import subprocess
import json
import os
import shutil
import re
import tempfile
import traceback
import logging
from typing import Dict, Any, List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
TRADER_FILE_ORIGINAL = "trader.py"
DATAMODEL_FILE_ORIGINAL = "datamodel.py"
STORAGE_DB = "sqlite:///prosperity3_round2_optimization.db"
STUDY_NAME = "prosperity3-round2-optimizer"
N_TRIALS = 50  # Number of trials to run (adjust as needed)
ROUNDS_TO_TEST = ["2"]  # Only test Round 2
PARALLEL_JOBS = 1  # Set to >1 if your system supports parallel execution

# --- Product-Specific Configurations ---
PRODUCTS = [
    "RAINFOREST_RESIN", 
    "KELP",
    "CROISSANTS",
    "JAMS",
    "DJEMBES",
    "PICNIC_BASKET1",
    "PICNIC_BASKET2"
]

# --- Parameter Search Space Definition ---
def define_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Define the hyperparameter search space for all products."""
    params = {
        "shared": {
            "take_profit_threshold": trial.suggest_float("shared_take_profit", 0.2, 2.0, step=0.05),
            "max_history_length": trial.suggest_int("shared_max_hist", 60, 120, step=5),
            "arbitrage_threshold": trial.suggest_float("shared_arb_threshold", 0.5, 5.0, step=0.1),
            "conversion_threshold": trial.suggest_float("shared_conv_threshold", 0.5, 5.0, step=0.1)
        }
    }
    
    # RAINFOREST_RESIN parameters
    params["RAINFOREST_RESIN"] = {
        "fair_value_anchor": trial.suggest_float("rr_anchor", 9800.0, 10200.0, step=10.0),
        "anchor_blend_alpha": trial.suggest_float("rr_alpha", 0.05, 0.2, step=0.01),
        "min_spread": trial.suggest_int("rr_min_spread", 5, 10),
        "volatility_spread_factor": trial.suggest_float("rr_vol_spread", 0.1, 0.6, step=0.05),
        "inventory_skew_factor": trial.suggest_float("rr_skew", 0.0, 0.03, step=0.002),
        "base_order_qty": trial.suggest_int("rr_qty", 15, 35, step=1),
        "reversion_threshold": trial.suggest_int("rr_revert", 1, 8)
    }
    
    # KELP parameters
    params["KELP"] = {
        "ema_alpha": trial.suggest_float("k_ema", 0.02, 0.2, step=0.01),
        "min_spread": trial.suggest_int("k_min_spread", 1, 4),
        "volatility_spread_factor": trial.suggest_float("k_vol_spread", 0.5, 2.5, step=0.1),
        "inventory_skew_factor": trial.suggest_float("k_skew", 0.005, 0.03, step=0.001),
        "base_order_qty": trial.suggest_int("k_qty", 15, 40, step=1),
        "min_volatility_qty_factor": trial.suggest_float("k_min_qty_f", 1.0, 2.0, step=0.05),
        "max_volatility_for_qty_reduction": trial.suggest_float("k_max_vol", 2.0, 8.0, step=0.5),
        "imbalance_depth": trial.suggest_int("k_imb_depth", 3, 7),
        "imbalance_fv_adjustment_factor": trial.suggest_float("k_imb_adj", 0.2, 0.8, step=0.05)
    }
    
    # CROISSANT parameters
    params["CROISSANTS"] = {
        "ema_alpha": trial.suggest_float("c_ema", 0.05, 0.2, step=0.01),
        "min_spread": trial.suggest_int("c_min_spread", 1, 5),
        "volatility_spread_factor": trial.suggest_float("c_vol_spread", 0.3, 1.5, step=0.1),
        "inventory_skew_factor": trial.suggest_float("c_skew", 0.001, 0.01, step=0.001),
        "base_order_qty": trial.suggest_int("c_qty", 30, 80, step=5),
        "imbalance_depth": trial.suggest_int("c_imb_depth", 2, 5),
        "imbalance_fv_adjustment_factor": trial.suggest_float("c_imb_adj", 0.1, 0.5, step=0.05)
    }
    
    # JAM parameters
    params["JAMS"] = {
        "ema_alpha": trial.suggest_float("j_ema", 0.05, 0.2, step=0.01),
        "min_spread": trial.suggest_int("j_min_spread", 1, 5),
        "volatility_spread_factor": trial.suggest_float("j_vol_spread", 0.3, 1.5, step=0.1),
        "inventory_skew_factor": trial.suggest_float("j_skew", 0.001, 0.01, step=0.001),
        "base_order_qty": trial.suggest_int("j_qty", 40, 100, step=5),
        "imbalance_depth": trial.suggest_int("j_imb_depth", 2, 5),
        "imbalance_fv_adjustment_factor": trial.suggest_float("j_imb_adj", 0.1, 0.5, step=0.05)
    }
    
    # DJEMBE parameters
    params["DJEMBES"] = {
        "ema_alpha": trial.suggest_float("d_ema", 0.02, 0.1, step=0.01),
        "min_spread": trial.suggest_int("d_min_spread", 2, 6),
        "volatility_spread_factor": trial.suggest_float("d_vol_spread", 0.5, 2.0, step=0.1),
        "inventory_skew_factor": trial.suggest_float("d_skew", 0.01, 0.04, step=0.002),
        "base_order_qty": trial.suggest_int("d_qty", 5, 20, step=1),
        "imbalance_depth": trial.suggest_int("d_imb_depth", 3, 5),
        "imbalance_fv_adjustment_factor": trial.suggest_float("d_imb_adj", 0.2, 0.6, step=0.05)
    }
    
    # PICNIC_BASKET1 parameters
    params["PICNIC_BASKET1"] = {
        "ema_alpha": trial.suggest_float("pb1_ema", 0.02, 0.1, step=0.01),
        "min_spread": trial.suggest_int("pb1_min_spread", 3, 8),
        "volatility_spread_factor": trial.suggest_float("pb1_vol_spread", 0.5, 1.5, step=0.1),
        "inventory_skew_factor": trial.suggest_float("pb1_skew", 0.005, 0.025, step=0.001),
        "base_order_qty": trial.suggest_int("pb1_qty", 8, 20, step=1),
        "imbalance_depth": trial.suggest_int("pb1_imb_depth", 2, 4),
        "imbalance_fv_adjustment_factor": trial.suggest_float("pb1_imb_adj", 0.1, 0.5, step=0.05)
    }
    
    # PICNIC_BASKET2 parameters
    params["PICNIC_BASKET2"] = {
        "ema_alpha": trial.suggest_float("pb2_ema", 0.02, 0.1, step=0.01),
        "min_spread": trial.suggest_int("pb2_min_spread", 3, 7),
        "volatility_spread_factor": trial.suggest_float("pb2_vol_spread", 0.5, 1.5, step=0.1),
        "inventory_skew_factor": trial.suggest_float("pb2_skew", 0.005, 0.025, step=0.001),
        "base_order_qty": trial.suggest_int("pb2_qty", 10, 30, step=1),
        "imbalance_depth": trial.suggest_int("pb2_imb_depth", 2, 4),
        "imbalance_fv_adjustment_factor": trial.suggest_float("pb2_imb_adj", 0.1, 0.5, step=0.05)
    }
    
    return params

# --- Helper Functions ---
def modify_trader_file(original_trader_file: str, params: Dict[str, Any], output_file: str) -> bool:
    """Modify the trader.py file with new parameters."""
    try:
        with open(original_trader_file, 'r') as f:
            original_code = f.read()
            
        # Find PARAMS section in the code
        params_start_index = original_code.find("PARAMS = {")
        if params_start_index == -1:
            logger.error("Could not find 'PARAMS = {' in the trader file")
            return False
            
        # Find the end of PARAMS section
        brace_level = 0
        params_end_index = -1
        in_string = False
        string_delimiter = None
        escape_next = False
        
        for i in range(params_start_index + len("PARAMS = {") - 1, len(original_code)):
            char = original_code[i]
            
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\':
                escape_next = True
                continue
                
            if not in_string:
                if char in ('"', "'"):
                    in_string = True
                    string_delimiter = char
                elif char == '{':
                    brace_level += 1
                elif char == '}':
                    brace_level -= 1
                    if brace_level == 0:
                        params_end_index = i + 1
                        break
            else:
                if char == string_delimiter:
                    in_string = False
        
        if params_end_index == -1:
            logger.error("Could not find the end of PARAMS section")
            return False
            
        # Replace PARAMS with new parameters
        params_string = f"PARAMS = {json.dumps(params, indent=4)}"
        modified_code = original_code[:params_start_index] + params_string + original_code[params_end_index:]
        
        # Write the modified code to the output file
        with open(output_file, 'w') as f:
            f.write(modified_code)
            
        return True
    except Exception as e:
        logger.error(f"Error modifying trader file: {e}")
        traceback.print_exc()
        return False

def run_backtests(trader_file: str, rounds: List[str]) -> Tuple[float, Dict[str, float]]:
    """Run backtests for multiple rounds and return the total PnL."""
    total_pnl = 0.0
    round_pnls = {}
    
    for round_num in rounds:
        try:
            cmd = ["prosperity3bt", trader_file, round_num, "--merge-pnl", "--no-out"]
            logger.info(f"Running backtest for round {round_num}...")
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=300)
            
            if result.returncode == 0:
                # Extract PnL from the output
                match = re.search(r"Total profit:\s*([+-]?[\d,]+(?:\.\d+)?)", result.stdout)
                if match:
                    try:
                        pnl_str = match.group(1).replace(',', '')
                        pnl = float(pnl_str)
                        total_pnl += pnl
                        round_pnls[round_num] = pnl
                        logger.info(f"Round {round_num} PnL: {pnl}")
                    except ValueError:
                        logger.warning(f"Could not parse PnL float from '{match.group(1)}'")
                else:
                    logger.warning(f"Could not find PnL pattern in stdout for round {round_num}")
            else:
                logger.error(f"Backtest failed for round {round_num} with code {result.returncode}")
                stderr_snippet = result.stderr[:500] + ('...' if len(result.stderr) > 500 else '')
                logger.error(f"STDERR: {stderr_snippet}")
                
        except subprocess.TimeoutExpired:
            logger.error(f"Backtest for round {round_num} timed out after 300 seconds")
        except Exception as e:
            logger.error(f"Error running backtest for round {round_num}: {e}")
            
    return total_pnl, round_pnls

# --- Optuna Objective Function ---
def objective(trial: optuna.Trial) -> float:
    """Optuna objective function to maximize PnL across multiple rounds."""
    # Get parameters from search space
    params = define_search_space(trial)
    temp_dir = None
    
    try:
        # Create temporary directory for this trial
        temp_dir = tempfile.mkdtemp()
        temp_trader_file = os.path.join(temp_dir, os.path.basename(TRADER_FILE_ORIGINAL))
        temp_datamodel_file = os.path.join(temp_dir, os.path.basename(DATAMODEL_FILE_ORIGINAL))
        
        # Copy datamodel.py to temp directory
        if os.path.exists(DATAMODEL_FILE_ORIGINAL):
            shutil.copyfile(DATAMODEL_FILE_ORIGINAL, temp_datamodel_file)
        else:
            logger.error(f"Original datamodel file '{DATAMODEL_FILE_ORIGINAL}' not found")
            return -float('inf')
            
        # Modify trader.py with new parameters
        if not modify_trader_file(TRADER_FILE_ORIGINAL, params, temp_trader_file):
            logger.error("Failed to modify trader file")
            return -float('inf')
            
        # Run backtests for all specified rounds
        total_pnl, round_pnls = run_backtests(temp_trader_file, ROUNDS_TO_TEST)
        
        # Log results for this trial
        logger.info(f"Trial {trial.number} completed with total PnL: {total_pnl:.2f}")
        for round_num, pnl in round_pnls.items():
            trial.set_user_attr(f"round_{round_num}_pnl", pnl)
            
        return total_pnl
        
    except optuna.exceptions.TrialPruned as e:
        logger.info(f"Trial {trial.number} pruned: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in objective function for trial {trial.number}: {e}")
        traceback.print_exc()
        return -float('inf')
    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except OSError as e:
                logger.error(f"Error removing temporary directory {temp_dir}: {e}")

# --- Main Function ---
def main():
    """Main function to run the Bayesian optimizer."""
    logger.info("Starting Bayesian optimization for Round 2 trader")
    logger.info(f"Study Name: {STUDY_NAME}")
    logger.info(f"Storage: {STORAGE_DB}")
    logger.info(f"Trials: {N_TRIALS}")
    logger.info(f"Testing rounds: {', '.join(ROUNDS_TO_TEST)}")
    logger.info(f"Products: {', '.join(PRODUCTS)}")
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_DB,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)  # TPE sampler with fixed seed for reproducibility
    )
    
    # Add pruner to stop unpromising trials
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study.pruner = pruner
    
    try:
        # Run optimization with progress bar
        study.optimize(
            objective, 
            n_trials=N_TRIALS, 
            timeout=None,  # No timeout
            n_jobs=PARALLEL_JOBS,  # Parallel optimization if supported
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        logger.info("Optimization stopped manually")
    except Exception as e:
        logger.error(f"Optimization error: {e}")
        traceback.print_exc()
        
    # Report results
    logger.info("\n--- Optimization Results ---")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if completed_trials:
        best_trial = study.best_trial
        logger.info(f"Best trial number: {best_trial.number}")
        logger.info(f"Best trial value (Max PnL): {best_trial.value:.2f}")
        
        # Print round-specific PnLs for the best trial
        for round_num in ROUNDS_TO_TEST:
            round_pnl = best_trial.user_attrs.get(f"round_{round_num}_pnl", "N/A")
            logger.info(f"  - Round {round_num} PnL: {round_pnl}")
            
        # Print best parameters
        logger.info("\nBest parameters:")
        best_params = best_trial.params
        pretty_params = organize_params(best_params)
        logger.info(json.dumps(pretty_params, indent=4))
        
        # Save best parameters to file
        with open('best_params_round2.json', 'w') as f:
            json.dump(pretty_params, f, indent=4)
        logger.info("Best parameters saved to 'best_params_round2.json'")
        
        # Generate visualizations if available
        if optuna.visualization.is_available():
            logger.info("\nGenerating visualizations...")
            try:
                fig_history = optuna.visualization.plot_optimization_history(study)
                fig_history.write_html("optimization_history.html")
                
                fig_importance = optuna.visualization.plot_param_importances(study)
                fig_importance.write_html("param_importances.html")
                
                fig_slice = optuna.visualization.plot_slice(study)
                fig_slice.write_html("param_slices.html")
                
                logger.info("Visualizations saved as HTML files")
                logger.info("You can also explore interactively with: optuna-dashboard sqlite:///prosperity3_round2_optimization.db")
            except Exception as ve:
                logger.error(f"Error generating visualizations: {ve}")
    else:
        logger.info("No trials completed successfully")

def organize_params(flat_params: Dict[str, Any]) -> Dict[str, Any]:
    """Reorganize flat parameter dict into nested structure by product."""
    organized = {"shared": {}, "RAINFOREST_RESIN": {}, "KELP": {}, "CROISSANTS": {}, "JAMS": {}, 
                "DJEMBES": {}, "PICNIC_BASKET1": {}, "PICNIC_BASKET2": {}}
    
    prefixes = {
        "shared_": "shared",
        "rr_": "RAINFOREST_RESIN",
        "k_": "KELP",
        "c_": "CROISSANTS",
        "j_": "JAMS",
        "d_": "DJEMBES",
        "pb1_": "PICNIC_BASKET1",
        "pb2_": "PICNIC_BASKET2"
    }
    
    param_mapping = {
        "take_profit": "take_profit_threshold",
        "max_hist": "max_history_length",
        "arb_threshold": "arbitrage_threshold",
        "conv_threshold": "conversion_threshold",
        "anchor": "fair_value_anchor",
        "alpha": "anchor_blend_alpha",
        "min_spread": "min_spread",
        "vol_spread": "volatility_spread_factor",
        "skew": "inventory_skew_factor",
        "qty": "base_order_qty",
        "revert": "reversion_threshold",
        "ema": "ema_alpha",
        "min_qty_f": "min_volatility_qty_factor",
        "max_vol": "max_volatility_for_qty_reduction",
        "imb_depth": "imbalance_depth",
        "imb_adj": "imbalance_fv_adjustment_factor"
    }
    
    for param_name, value in flat_params.items():
        # Find the prefix that matches this parameter
        matched_prefix = None
        for prefix, product in prefixes.items():
            if param_name.startswith(prefix):
                matched_prefix = prefix
                param_key = param_name[len(prefix):]  # Remove prefix
                
                # Map to the full parameter name if available
                if param_key in param_mapping:
                    param_key = param_mapping[param_key]
                    
                # Add to the correct product dict
                organized[product][param_key] = value
                break
                
        if not matched_prefix:
            # Parameter doesn't match any known prefix, add to unknown section
            organized.setdefault("unknown", {})[param_name] = value
            
    return organized

# --- Create Best Trader File ---
def create_best_trader_file(study: optuna.Study) -> None:
    """Create a trader file with the best parameters found."""
    best_trial = study.best_trial
    best_params = best_trial.params
    organized_params = organize_params(best_params)
    
    output_file = "best_trader_round2.py"
    
    if modify_trader_file(TRADER_FILE_ORIGINAL, organized_params, output_file):
        logger.info(f"Best trader saved to '{output_file}'")
    else:
        logger.error("Failed to create best trader file")

if __name__ == "__main__":
    main()
    
    # Load study and create best trader file after optimization
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE_DB)
        create_best_trader_file(study)
    except Exception as e:
        logger.error(f"Error creating best trader file: {e}")
        traceback.print_exc()