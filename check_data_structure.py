#!/usr/bin/env python
# check_data_structure.py - A diagnostic script to check CSV data structure

import os
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_delimiter(file_path):
    """Detect the delimiter used in a CSV file by checking the first line."""
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        
    if ';' in first_line:
        return ';'
    elif ',' in first_line:
        return ','
    else:
        return None

def check_file(file_path):
    """Check and report the structure of a CSV file."""
    logger.info(f"Checking file: {file_path}")
    
    # Detect delimiter
    delimiter = detect_delimiter(file_path)
    if not delimiter:
        logger.error(f"Could not detect delimiter in {file_path}")
        return
    
    logger.info(f"Detected delimiter: '{delimiter}'")
    
    # Read the file
    try:
        df = pd.read_csv(file_path, delimiter=delimiter)
        
        # Report basic file stats
        row_count = len(df)
        col_count = len(df.columns)
        logger.info(f"Rows: {row_count}, Columns: {col_count}")
        
        # Report column names
        logger.info(f"Columns: {', '.join(df.columns.tolist())}")
        
        # Check for common column names
        for common_col in ['timestamp', 'product', 'symbol', 'buyer', 'seller', 
                          'bid_price_1', 'bid_volume_1', 'ask_price_1', 'ask_volume_1',
                          'price', 'quantity', 'currency']:
            if common_col in df.columns:
                logger.info(f"Column '{common_col}' is present")
                
                # Show sample values
                unique_values = df[common_col].nunique()
                logger.info(f"  - Unique values: {unique_values}")
                
                if unique_values > 0 and unique_values <= 10:
                    values = df[common_col].unique()
                    logger.info(f"  - Values: {values}")
                elif unique_values > 0:
                    sample_values = df[common_col].sample(min(5, len(df))).tolist()
                    logger.info(f"  - Sample values: {sample_values}")
        
        # Show first row as sample
        if row_count > 0:
            logger.info("Sample first row:")
            for col, val in df.iloc[0].items():
                logger.info(f"  {col}: {val}")
        
    except Exception as e:
        logger.error(f"Error analyzing file: {e}")

def main():
    """Main function to check data files."""
    data_dir = "data"
    if not os.path.exists(data_dir):
        logger.error(f"Data directory '{data_dir}' not found.")
        return
    
    logger.info(f"Scanning directory: {data_dir}")
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    logger.info(f"Found {len(csv_files)} CSV files")
    
    # Process price files first, then trade files
    price_files = sorted([f for f in csv_files if 'price' in f.lower()])
    trade_files = sorted([f for f in csv_files if 'trade' in f.lower()])
    
    logger.info(f"Price files: {len(price_files)}")
    for file in price_files:
        check_file(os.path.join(data_dir, file))
        logger.info("-" * 80)
    
    logger.info(f"Trade files: {len(trade_files)}")
    for file in trade_files:
        check_file(os.path.join(data_dir, file))
        logger.info("-" * 80)

if __name__ == "__main__":
    main()