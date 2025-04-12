#!/usr/bin/env python3
# download_data.py - Script to help download IMC Prosperity data files

import os
import sys
import requests
import argparse

def main():
    parser = argparse.ArgumentParser(description="Download IMC Prosperity data files")
    parser.add_argument('--round', type=int, default=2, help='Round number (default: 2)')
    parser.add_argument('--day', type=int, default=0, help='Day number (default: 0)')
    args = parser.parse_args()
    
    round_num = args.round
    day_num = args.day
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        print("Creating data directory...")
        os.makedirs('data')
    
    # Define file names
    prices_file = f"prices_round_{round_num}_day_{day_num}.csv"
    trades_file = f"trades_round_{round_num}_day_{day_num}.csv"
    
    price_path = os.path.join('data', prices_file)
    trade_path = os.path.join('data', trades_file)
    
    # Check if files already exist
    if os.path.exists(price_path) and os.path.exists(trade_path):
        print(f"Files for Round {round_num} Day {day_num} already exist in data directory.")
        overwrite = input("Do you want to re-download them? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Download canceled.")
            return
    
    # Instructions for manual download
    print(f"\n=== Instructions for downloading Round {round_num} Day {day_num} data ===")
    print("Since the IMC Prosperity data files require login/authentication, you need to:")
    print("1. Log in to the IMC Prosperity platform")
    print("2. Navigate to the Round 2 page")
    print("3. Download the following files:")
    print(f"   - {prices_file}")
    print(f"   - {trades_file}")
    print("4. Save these files to the 'data' directory in your project folder\n")
    
    open_browser = input("Would you like help opening your web browser to the IMC Prosperity site? (y/n): ").strip().lower()
    if open_browser == 'y':
        try:
            import webbrowser
            print("Opening IMC Prosperity platform in your browser...")
            webbrowser.open("https://prosperity.imc.com/")
        except Exception as e:
            print(f"Failed to open browser: {e}")
            print("Please manually navigate to: https://prosperity.imc.com/")
    
    print("\nAfter downloading the files, run the check_environment.sh script to verify your setup.")

if __name__ == "__main__":
    main()