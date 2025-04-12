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
    'PICNIC_BASKET1': {'CROISSANTS': 6, 'JAMS': 3, 'DJEMBES': 1},
    'PICNIC_BASKET2': {'CROISSANTS': 4, 'JAMS': 2}
}

class BasketAnalyzer:
    def __init__(self, data_dir: str, output_dir: str = "basket_analysis_output"):
        """Initialize analyzer with data directory path and output directory."""
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.prices_files = []
        self.trades_files = []
        self.day_data = {}  # Dictionary to store data for each day
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
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
        match = re.search(r'day_(-?\d+)', filename)
        if match:
            return int(match.group(1))
        return None
    
    def _detect_delimiter(self, file_path: str) -> str:
        """Detect the delimiter used in a CSV file."""
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()
        
        if ';' in first_line:
            return ';'
        return ','
    
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
                # Detect delimiter
                delimiter = self._detect_delimiter(price_file)
                
                # Read the file with the appropriate delimiter
                df = pd.read_csv(price_file, delimiter=delimiter)
                
                # Check if 'product' column exists, if not but 'symbol' does, rename it
                if 'product' not in df.columns and 'symbol' in df.columns:
                    df.rename(columns={'symbol': 'product'}, inplace=True)
                
                # Initialize day data if not exists
                if day not in self.day_data:
                    self.day_data[day] = {'prices': None, 'trades': None}
                
                self.day_data[day]['prices'] = df
                logger.info(f"Loaded {len(df)} price records for day {day}")
                
                # Print sample to debug
                logger.info(f"Sample price data columns: {df.columns.tolist()}")
                if len(df) > 0:
                    logger.info(f"First row sample: {df.iloc[0].to_dict()}")
                
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
                # Detect delimiter
                delimiter = self._detect_delimiter(trade_file)
                
                # Read the file with the appropriate delimiter
                df = pd.read_csv(trade_file, delimiter=delimiter)
                
                # Check if 'symbol' column exists, if not but 'product' does, rename it
                if 'symbol' not in df.columns and 'product' in df.columns:
                    df.rename(columns={'product': 'symbol'}, inplace=True)
                
                # Initialize day data if not exists
                if day not in self.day_data:
                    self.day_data[day] = {'prices': None, 'trades': None}
                
                self.day_data[day]['trades'] = df
                logger.info(f"Loaded {len(df)} trade records for day {day}")
                
                # Print sample to debug
                logger.info(f"Sample trade data columns: {df.columns.tolist()}")
                if len(df) > 0:
                    logger.info(f"First row sample: {df.iloc[0].to_dict()}")
                
            except Exception as e:
                logger.error(f"Error loading trade file {trade_file}: {e}")
    
    def preprocess_data(self) -> None:
        """Preprocess the loaded data for analysis."""
        for day, data in self.day_data.items():
            prices_df = data.get('prices')
            if prices_df is not None:
                # Ensure column names are normalized
                if 'product' not in prices_df.columns and 'symbol' in prices_df.columns:
                    prices_df.rename(columns={'symbol': 'product'}, inplace=True)
                
                # Calculate mid prices if not already present
                if 'mid_price' not in prices_df.columns:
                    # Check for bid_price_1 and ask_price_1 columns
                    bid_col = None
                    ask_col = None
                    
                    # Try different column naming patterns
                    if 'bid_price_1' in prices_df.columns and 'ask_price_1' in prices_df.columns:
                        bid_col = 'bid_price_1'
                        ask_col = 'ask_price_1'
                    elif 'bid_price1' in prices_df.columns and 'ask_price1' in prices_df.columns:
                        bid_col = 'bid_price1'
                        ask_col = 'ask_price1'
                    
                    if bid_col and ask_col:
                        # Ensure prices are numeric
                        prices_df[bid_col] = pd.to_numeric(prices_df[bid_col], errors='coerce')
                        prices_df[ask_col] = pd.to_numeric(prices_df[ask_col], errors='coerce')
                        
                        # Calculate mid price
                        prices_df['mid_price'] = (prices_df[bid_col] + prices_df[ask_col]) / 2
                    else:
                        logger.warning(f"Could not find bid_price_1 and ask_price_1 columns for day {day}")
                
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
            
            # Log the unique products in this day's data
            available_products = set(prices_df['product'].unique())
            logger.info(f"Available products for day {day}: {available_products}")
            
            # Ensure we have all required products
            required_products = set(['PICNIC_BASKET1', 'PICNIC_BASKET2', 'CROISSANTS', 'JAMS', 'DJEMBES'])
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
                    logger.warning(f"Basket {basket} not available for day {day}")
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
                        bid_price_col = next((col for col in basket_data.index if 'bid_price_1' in col or 'bid_price1' in col), None)
                        ask_price_col = next((col for col in basket_data.index if 'ask_price_1' in col or 'ask_price1' in col), None)
                        
                        if bid_price_col and ask_price_col:
                            bid = basket_data[bid_price_col]
                            ask = basket_data[ask_price_col]
                            if not pd.isna(bid) and not pd.isna(ask):
                                basket_mid = (bid + ask) / 2
                            else:
                                continue
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
                            bid_price_col = next((col for col in comp_data.index if 'bid_price_1' in col or 'bid_price1' in col), None)
                            ask_price_col = next((col for col in comp_data.index if 'ask_price_1' in col or 'ask_price1' in col), None)
                            
                            if bid_price_col and ask_price_col:
                                bid = comp_data[bid_price_col]
                                ask = comp_data[ask_price_col]
                                if not pd.isna(bid) and not pd.isna(ask):
                                    comp_mid = (bid + ask) / 2
                                else:
                                    missing_components = True
                                    break
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
                else:
                    logger.warning(f"No valid analysis data points for {basket} on day {day}")
            
            results[day] = basket_analysis
        
        return results
    
    def plot_basket_analysis(self, analysis_results: Dict[int, Dict[str, pd.DataFrame]]) -> None:
        """Plot the basket pricing analysis results."""
        for day, baskets in analysis_results.items():
            # Create day-specific folder
            day_folder = os.path.join(self.output_dir, f"day_{day}")
            os.makedirs(day_folder, exist_ok=True)
            
            logger.info(f"Plotting results for day {day} to {day_folder}")
            
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
                
                # Save the figure to the day folder
                file_path = os.path.join(day_folder, f'{basket}_analysis.png')
                plt.tight_layout()
                plt.savefig(file_path)
                plt.close()
                
                logger.info(f"Created plot for {basket} on day {day} at {file_path}")
                
                # Also create a histogram of discount/premium
                plt.figure(figsize=(10, 6))
                plt.hist(df['discount'], bins=50, alpha=0.7)
                plt.axvline(x=0, color='red', linestyle='--')
                plt.title(f'{basket} Discount/Premium Distribution - Day {day}')
                plt.xlabel('Discount/Premium')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                hist_path = os.path.join(day_folder, f'{basket}_histogram.png')
                plt.savefig(hist_path)
                plt.close()
                
                logger.info(f"Created histogram for {basket} on day {day} at {hist_path}")
                
                # Save the raw data to CSV
                csv_path = os.path.join(day_folder, f'{basket}_data.csv')
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved raw data for {basket} on day {day} to {csv_path}")
    
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
        # Create path for summary report
        report_path = os.path.join(self.output_dir, 'basket_analysis_summary.txt')
        
        with open(report_path, 'w') as f:
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
            
            if all_discounts:
                f.write(f"Total data points: {len(all_discounts)}\n")
                f.write(f"Mean discount/premium: {np.mean(all_discounts):.4f}\n")
                f.write(f"Standard deviation: {np.std(all_discounts):.4f}\n")
                f.write(f"Min discount/premium: {min(all_discounts):.4f}\n")
                f.write(f"Max discount/premium: {max(all_discounts):.4f}\n")
                f.write(f"Total arbitrage opportunities: {all_opportunities}\n\n")
            else:
                f.write("Total data points: 0\n")
                f.write("Mean discount/premium: nan\n")
                f.write("Standard deviation: nan\n")
            
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
            
            logger.info(f"Generated summary report: {report_path}")
            
            # Also save identified opportunities to CSV files
            for day, baskets in opportunities.items():
                day_folder = os.path.join(self.output_dir, f"day_{day}")
                os.makedirs(day_folder, exist_ok=True)
                
                for basket, df in baskets.items():
                    opps_path = os.path.join(day_folder, f'{basket}_opportunities.csv')
                    df.to_csv(opps_path, index=False)
                    logger.info(f"Saved opportunity data for {basket} on day {day} to {opps_path}")
    
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
    parser.add_argument("--output_dir", type=str, default="basket_analysis_output", help="Directory to save output files")
    parser.add_argument("--threshold", type=float, default=1.0, help="Threshold for identifying arbitrage opportunities")
    
    args = parser.parse_args()
    
    analyzer = BasketAnalyzer(args.data_dir, args.output_dir)
    analyzer.run_analysis(args.threshold)