import pandas as pd
from datamodel import * # Import all classes from datamodel.py
from trader import Trader # Import your Trader class
import time # To measure execution time

# --- Configuration ---
# Update with correct file paths based on your directory structure
DATA_DIR = 'data/'
ORDER_BOOK_CSV_PATH = DATA_DIR + 'prices_round_2_day_0.csv'  # Change day number as needed
TRADE_CSV_PATH = DATA_DIR + 'trades_round_2_day_0.csv'       # Change day number as needed

# --- Helper Function to Parse Order Book Data ---
def parse_order_depth_from_row(row, product: str) -> OrderDepth:
    depth = OrderDepth()
    # Parse up to 3 levels of depth provided
    for i in range(1, 4):
        bid_price_col = f'bid_price_{i}'
        bid_vol_col = f'bid_volume_{i}'
        ask_price_col = f'ask_price_{i}'
        ask_vol_col = f'ask_volume_{i}'

        if bid_price_col in row and pd.notna(row[bid_price_col]) and row[bid_vol_col] > 0:
            depth.buy_orders[int(row[bid_price_col])] = int(row[bid_vol_col])
        if ask_price_col in row and pd.notna(row[ask_price_col]) and row[ask_vol_col] > 0:
            # Remember sell orders use negative quantity in datamodel
            depth.sell_orders[int(row[ask_price_col])] = -int(row[ask_vol_col])
            
    # Ensure bids are sorted high to low, asks low to high
    depth.buy_orders = dict(sorted(depth.buy_orders.items(), reverse=True))
    depth.sell_orders = dict(sorted(depth.sell_orders.items()))
    return depth

# --- Main Simulation Loop ---
def run_simulation():
    print(f"Loading data from {ORDER_BOOK_CSV_PATH}...")
    try:
        # Use semicolon as delimiter based on your data file format
        market_data_df = pd.read_csv(ORDER_BOOK_CSV_PATH, delimiter=';')
        print(f"Data loaded. Columns: {market_data_df.columns.tolist()}")
    except FileNotFoundError:
        print(f"ERROR: File not found at {ORDER_BOOK_CSV_PATH}")
        return
    except Exception as e:
        print(f"ERROR loading CSV: {e}")
        return

    # Get unique products and timestamps from the data
    products = market_data_df['product'].unique()
    timestamps = sorted(market_data_df['timestamp'].unique())
    
    print(f"Found Products: {products}")
    print(f"Simulating {len(timestamps)} timestamps...")

    # Initialize trader and state variables
    trader = Trader()
    traderData = "" # Initial empty state string
    current_positions: Dict[Product, Position] = {p: 0 for p in products} # Start with zero positions

    # --- Loop through each timestamp in the sample data ---
    for i, ts in enumerate(timestamps):
        start_time = time.time()
        print(f"\n--- Timestamp: {ts} ---")
        
        # Filter data for the current timestamp
        ts_data = market_data_df[market_data_df['timestamp'] == ts]

        # Construct the TradingState for this timestamp
        listings: Dict[Symbol, Listing] = {}
        order_depths: Dict[Symbol, OrderDepth] = {}
        
        for product in products:
             product_row = ts_data[ts_data['product'] == product]
             if not product_row.empty:
                 row = product_row.iloc[0] # Get the first row for this product/timestamp
                 listings[product] = Listing(symbol=product, product=product, denomination="SEASHELLS")
                 order_depths[product] = parse_order_depth_from_row(row, product)
             else:
                  # Handle cases where a product might not have data at a specific timestamp
                  listings[product] = Listing(symbol=product, product=product, denomination="SEASHELLS")
                  order_depths[product] = OrderDepth() # Empty book

        # Initialize empty trade lists
        own_trades = {p: [] for p in products}
        market_trades = {p: [] for p in products}

        # Create empty observations
        observations = Observation(plainValueObservations={}, conversionObservations={})

        # Create the state object
        state = TradingState(
            traderData=traderData,
            timestamp=ts,
            listings=listings,
            order_depths=order_depths,
            own_trades=own_trades,
            market_trades=market_trades,
            position=current_positions.copy(), # Pass a copy
            observations=observations
        )

        # --- Call the Trader's run method ---
        try:
            result, conversions, traderData = trader.run(state)
            # Check execution time
            end_time = time.time()
            duration = (end_time - start_time) * 1000 # ms
            print(f"Trader.run() executed in {duration:.2f} ms")
            if duration > 900:
                print("WARNING: Execution time exceeded 900ms!")

            # Print intended orders and conversions
            print(f"Intended Orders: {result}")
            print(f"Intended Conversions: {conversions}")

            # Basic position update (simplified)
            for product, orders in result.items():
                 for order in orders:
                      current_positions[product] = current_positions.get(product, 0) + order.quantity
                      
            print(f"Approx Updated Positions: {current_positions}")

        except Exception as e:
            print(f"ERROR during trader.run() at timestamp {ts}: {e}")
            import traceback
            traceback.print_exc()
            break # Stop simulation on error

    print("\n--- Simulation Finished ---")

# --- Run the main simulation function ---
if __name__ == "__main__":
    run_simulation()