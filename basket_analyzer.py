# basket_analyzer.py
# A tool to analyze pricing relationships between picnic baskets and their components

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from typing import Dict, List, Tuple, Set, Optional
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants for Round 2 ---
BASKET_CONTENTS = {
    'PICNIC_BASKET1': {'CROISSANT': 6, 'JAM': 3, 'DJEMBE': 1},
    'PICNIC_BASKET2': {'CROISSANT': 4, 'JAM': 2}
}

class BasketAnalyzer:
    def __init__(self, data_dir: str):
        """Initialize analyzer with data directory path."""
        self.data_dir = data_dir
        self.prices_files = []
        self.trades_files = []
        self.day_data = {}  # Dictionary to store data for each day
        
        # Find all price and trade files in the directory
        self._find_data_files()
        
    def _find_data_files(self) -> None:
        """Find all price and trade CSV files in the data directory."""
        if not os.path.exists(self.data_dir):
            logger.error(f"Data directory {self.data_dir} does not exist.")
            return
            
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.csv'):
                filepath = os.path.join(self.data_dir, filename)
                # Check if it's a price or trade file
                if 'price' in filename.lower():
                    self.prices_files.append(filepath)
                elif 'trade' in filename.lower():
                    self.trades_files.append(filepath)
                    
        logger.info(f"Found {len(self.prices_files)} price files and {len(self.trades_files)} trade files.")
    
    def _parse_day_from_filename(self, filename: str) -> Optional[int]:
        """Extract day number from filename."""
        match = re.search(r'day_(\d+)', filename)
        if match:
            return int(match.group(1))
        return None
    
    def load_data(self) -> None:
        """Load price and trade data from all files."""
        # Process price files
        for price_file in self.prices_files:
            day = self._parse_day_from_filename(os.path.basename(price_file))
            if day is None:
                logger.warning(f"Could not parse day from {price_file}, skipping.")
                continue
                
            logger.info(f"Loading price data for day {day} from {os.path.basename(price_file)}")
            try:
                # First check the file format
                with open(price_file, 'r') as f:
                    first_line = f.readline().strip()
                
                # Determine delimiter based on first line
                if ';' in first_line:
                    delimiter = ';'
                else:
                    delimiter = ','
                
                # Read the file with the appropriate delimiter
                df = pd.read_csv(price_file, delimiter=delimiter)
                
                # Initialize day data if not exists
                if day not in self.day_data:
                    self.day_data[day] = {'prices': None, 'trades': None}
                
                self.day_data[day]['prices'] = df
                logger.info(f"Loaded {len(df)} price records for day {day}")
            except Exception as e:
                logger.error(f"Error loading price file {price_file}: {e}")
        
        # Process trade files
        for trade_file in self.trades_files:
            day = self._parse_day_from_filename(os.path.basename(trade_file))
            if day is None:
                logger.warning(f"Could not parse day from {trade_file}, skipping.")
                continue
                
            logger.info(f"Loading trade data for day {day} from {os.path.basename(trade_file)}")
            try:
                # First check the file format
                with open(trade_file, 'r') as f:
                    first_line = f.readline().strip()
                
                # Determine delimiter based on first line
                if ';' in first_line:
                    delimiter = ';'
                else:
                    delimiter = ','
                
                # Read the file with the appropriate delimiter
                df = pd.read_csv(trade_file, delimiter=delimiter)
                
                # Initialize day data if not exists
                if day not in self.day_data:
                    self.day_data[day] = {'prices': None, 'trades': None}
                
                self.day_data[day]['trades'] = df
                logger.info(f"Loaded {len(df)} trade records for day {day}")
            except Exception as e:
                logger.error(f"Error loading trade file {trade_file}: {e}")
    
    def preprocess_data(self) -> None:
        """Preprocess the loaded data for analysis."""
        for day, data in self.day_data.items():
            prices_df = data.get('prices')
            if prices_df is not None:
                # Ensure column names are normalized
                # This step depends on the actual format of your data
                # For example, renaming columns if necessary
                if 'product' not in prices_df.columns and 'symbol' in prices_df.columns:
                    prices_df.rename(columns={'symbol': 'product'}, inplace=True)
                
                # Calculate mid prices if not already present
                if 'mid_price' not in prices_df.columns:
                    # This assumes you have bid and ask price columns
                    if 'bid_price_1' in prices_df.columns and 'ask_price_1' in prices_df.columns:
                        prices_df['mid_price'] = (prices_df['bid_price_1'] + prices_df['ask_price_1']) / 2
                
                # Store the preprocessed data back
                self.day_data[day]['prices'] = prices_df
                logger.info(f"Preprocessed price data for day {day}")
    
    def analyze_basket_pricing(self) -> Dict[int, Dict[str, pd.DataFrame]]:
        """Analyze the pricing relationship between baskets and their components."""
        results = {}
        
        for day, data in self.day_data.items():
            prices_df = data.get('prices')
            if prices_df is None:
                logger.warning(f"No price data for day {day}, skipping analysis.")
                continue
            
            # Ensure we have all required products
            required_products = set(['PICNIC_BASKET1', 'PICNIC_BASKET2', 'CROISSANT', 'JAM', 'DJEMBE'])
            available_products = set(prices_df['product'].unique())
            missing_products = required_products - available_products
            
            if missing_products:
                logger.warning(f"Missing required products for day {day}: {missing_products}")
                if 'PICNIC_BASKET1' in missing_products or 'PICNIC_BASKET2' in missing_products:
                    logger.warning(f"Skipping basket analysis for day {day} due to missing basket products.")
                    continue
            
            # Group by timestamp and product to get price data at each point in time
            prices_by_time = {}
            timestamps = prices_df['timestamp'].unique()
            
            for ts in timestamps:
                ts_data = prices_df[prices_df['timestamp'] == ts]
                prices_by_time[ts] = {row['product']: row for _, row in ts_data.iterrows()}
            
            # Analyze each basket
            basket_analysis = {}
            
            for basket, contents in BASKET_CONTENTS.items():
                if basket not in available_products:
                    continue
                
                # Prepare data for this basket
                analysis_data = []
                
                for ts, products in prices_by_time.items():
                    if basket not in products:
                        continue
                    
                    basket_data = products[basket]
                    basket_mid = basket_data.get('mid_price')
                    if pd.isna(basket_mid):
                        # Try to calculate mid price from best bid/ask if available
                        bid = basket_data.get('bid_price_1')
                        ask = basket_data.get('ask_price_1')
                        if not pd.isna(bid) and not pd.isna(ask):
                            basket_mid = (bid + ask) / 2
                        else:
                            continue
                    
                    # Calculate theoretical basket value from components
                    component_value = 0
                    missing_components = False
                    
                    for component, qty in contents.items():
                        if component not in products:
                            missing_components = True
                            break
                        
                        comp_data = products[component]
                        comp_mid = comp_data.get('mid_price')
                        if pd.isna(comp_mid):
                            # Try to calculate mid price from best bid/ask if available
                            bid = comp_data.get('bid_price_1')
                            ask = comp_data.get('ask_price_1')
                            if not pd.isna(bid) and not pd.isna(ask):
                                comp_mid = (bid + ask) / 2
                            else:
                                missing_components = True
                                break
                        
                        component_value += comp_mid * qty
                    
                    if missing_components:
                        continue
                    
                    # Calculate arbitrage opportunity metrics
                    basket_discount = basket_mid - component_value
                    basket_discount_pct = (basket_mid - component_value) / component_value * 100
                    
                    analysis_data.append({
                        'timestamp': ts,
                        'basket_mid': basket_mid,
                        'theoretical_value': component_value,
                        'discount': basket_discount,
                        'discount_pct': basket_discount_pct
                    })
                
                # Convert to DataFrame and store
                if analysis_data:
                    basket_analysis[basket] = pd.DataFrame(analysis_data)
                    logger.info(f"Analyzed {len(analysis_data)} data points for {basket} on day {day}")
            
            results[day] = basket_analysis
        
        return results
    
    def plot_basket_analysis(self, analysis_results: Dict[int, Dict[str, pd.DataFrame]]) -> None:
        """Plot the basket pricing analysis results."""
        for day, baskets in analysis_results.items():
            logger.info(f"Plotting results for day {day}")
            
            for basket, df in baskets.items():
                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
                
                # Plot 1: Basket price vs. Theoretical Value
                ax1.plot(df['timestamp'], df['basket_mid'], label=f'{basket} Price', color='blue')
                ax1.plot(df['timestamp'], df['theoretical_value'], label='Theoretical Value', color='green')
                ax1.set_title(f'{basket} Price vs. Theoretical Value - Day {day}')
                ax1.set_ylabel('Price')
                ax1.legend()
                ax1.grid(True)
                
                # Plot 2: Discount/Premium
                ax2.plot(df['timestamp'], df['discount'], color='red')
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax2.set_title(f'{basket} Discount/Premium - Day {day}')
                ax2.set_xlabel('Timestamp')
                ax2.set_ylabel('Discount/Premium')
                ax2.grid(True)
                
                # Add text annotations for mean, max, min
                mean_discount = df['discount'].mean()
                max_discount = df['discount'].max()
                min_discount = df['discount'].min()
                std_discount = df['discount'].std()
                
                ax2.text(0.02, 0.95, 
                        f'Mean: {mean_discount:.2f}\nMax: {max_discount:.2f}\nMin: {min_discount:.2f}\nStd: {std_discount:.2f}',
                        transform=ax2.transAxes, bbox=dict(facecolor='white', alpha=0.8))
                
                # Save the figure
                plt.tight_layout()
                plt.savefig(f'{basket}_analysis_day_{day}.png')
                plt.close()
                
                logger.info(f"Created plot for {basket} on day {day}")
                
                # Also create a histogram of discount/premium
                plt.figure(figsize=(10, 6))
                plt.hist(df['discount'], bins=50, alpha=0.7)
                plt.axvline(x=0, color='red', linestyle='--')
                plt.title(f'{basket} Discount/Premium Distribution - Day {day}')
                plt.xlabel('Discount/Premium')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                plt.savefig(f'{basket}_histogram_day_{day}.png')
                plt.close()
    
    def identify_arbitrage_opportunities(self, analysis_results: Dict[int, Dict[str, pd.DataFrame]], 
                                      threshold: float = 1.0) -> Dict[int, Dict[str, pd.DataFrame]]:
        """Identify potential arbitrage opportunities based on a threshold."""
        opportunities = {}
        
        for day, baskets in analysis_results.items():
            day_opps = {}
            
            for basket, df in baskets.items():
                # Find opportunities where the absolute discount exceeds the threshold
                buy_basket_opps = df[df['discount'] < -threshold].copy()
                buy_basket_opps['type'] = 'buy_basket_sell_components'
                buy_basket_opps['profit_potential'] = -buy_basket_opps['discount']
                
                sell_basket_opps = df[df['discount'] > threshold].copy()
                sell_basket_opps['type'] = 'buy_components_sell_basket'
                sell_basket_opps['profit_potential'] = sell_basket_opps['discount']
                
                # Combine all opportunities
                all_opps = pd.concat([buy_basket_opps, sell_basket_opps])
                
                if not all_opps.empty:
                    day_opps[basket] = all_opps
                    logger.info(f"Found {len(all_opps)} arbitrage opportunities for {basket} on day {day}")
                    logger.info(f"  - Buy basket opportunities: {len(buy_basket_opps)}")
                    logger.info(f"  - Sell basket opportunities: {len(sell_basket_opps)}")
                    logger.info(f"  - Average profit potential: {all_opps['profit_potential'].mean():.2f}")
            
            if day_opps:
                opportunities[day] = day_opps
        
        return opportunities
    
    def generate_summary_report(self, analysis_results: Dict[int, Dict[str, pd.DataFrame]], 
                              opportunities: Dict[int, Dict[str, pd.DataFrame]]) -> None:
        """Generate a summary report of the analysis."""
        with open('basket_analysis_summary.txt', 'w') as f:
            f.write("=== Basket Analysis Summary ===\n\n")
            
            # Overall statistics
            f.write("=== Overall Statistics ===\n")
            all_discounts = []
            all_opportunities = 0
            
            for day, baskets in analysis_results.items():
                for basket, df in baskets.items():
                    all_discounts.extend(df['discount'].tolist())
                    
                    # Count opportunities
                    if day in opportunities and basket in opportunities[day]:
                        all_opportunities += len(opportunities[day][basket])
            
            f.write(f"Total data points: {len(all_discounts)}\n")
            f.write(f"Mean discount/premium: {np.mean(all_discounts):.4f}\n")
            f.write(f"Standard deviation: {np.std(all_discounts):.4f}\n")
            f.write(f"Min discount/premium: {min(all_discounts):.4f}\n")
            f.write(f"Max discount/premium: {max(all_discounts):.4f}\n")
            f.write(f"Total arbitrage opportunities: {all_opportunities}\n\n")
            
            # Day-by-day statistics
            for day, baskets in analysis_results.items():
                f.write(f"=== Day {day} Analysis ===\n")
                
                for basket, df in baskets.items():
                    f.write(f"  {basket}:\n")
                    f.write(f"    Data points: {len(df)}\n")
                    f.write(f"    Mean discount/premium: {df['discount'].mean():.4f}\n")
                    f.write(f"    Standard deviation: {df['discount'].std():.4f}\n")
                    f.write(f"    Min discount/premium: {df['discount'].min():.4f}\n")
                    f.write(f"    Max discount/premium: {df['discount'].max():.4f}\n")
                    
                    # Opportunity statistics if available
                    if day in opportunities and basket in opportunities[day]:
                        opps = opportunities[day][basket]
                        buy_basket = opps[opps['type'] == 'buy_basket_sell_components']
                        sell_basket = opps[opps['type'] == 'buy_components_sell_basket']
                        
                        f.write(f"    Arbitrage opportunities: {len(opps)}\n")
                        f.write(f"      - Buy basket opportunities: {len(buy_basket)}\n")
                        f.write(f"      - Sell basket opportunities: {len(sell_basket)}\n")
                        f.write(f"      - Mean profit potential: {opps['profit_potential'].mean():.4f}\n")
                        f.write(f"      - Max profit potential: {opps['profit_potential'].max():.4f}\n")
                    else:
                        f.write("    No arbitrage opportunities found.\n")
                    
                    f.write("\n")
                
                f.write("\n")
            
            logger.info(f"Generated summary report: basket_analysis_summary.txt")
    
    def run_analysis(self, threshold: float = 1.0) -> None:
        """Run the complete analysis pipeline."""
        logger.info("Starting basket analysis pipeline")
        
        # Load data from files
        self.load_data()
        
        # Preprocess the data
        self.preprocess_data()
        
        # Analyze basket pricing
        analysis_results = self.analyze_basket_pricing()
        
        # Identify arbitrage opportunities
        opportunities = self.identify_arbitrage_opportunities(analysis_results, threshold)
        
        # Plot the results
        self.plot_basket_analysis(analysis_results)
        
        # Generate summary report
        self.generate_summary_report(analysis_results, opportunities)
        
        logger.info("Analysis pipeline completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze basket pricing for IMC Prosperity Round 2")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing price and trade CSV files")
    parser.add_argument("--threshold", type=float, default=1.0, help="Threshold for identifying arbitrage opportunities")
    
    args = parser.parse_args()
    
    analyzer = BasketAnalyzer(args.data_dir)
    analyzer.run_analysis(args.threshold)