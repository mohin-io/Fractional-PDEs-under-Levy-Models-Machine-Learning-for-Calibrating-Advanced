"""
Module for cleaning and filtering raw option market data.

Ensures data quality by removing illiquid options, handling bid-ask spreads,
and checking for basic arbitrage violations.
"""
import pandas as pd

# Define liquidity and data quality thresholds
MIN_VOLUME = 10
MIN_OPEN_INTEREST = 100
MAX_BID_ASK_SPREAD_RATIO = 0.3 # Spread shouldn't be more than 30% of mid-price

def clean_option_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans a raw option chain DataFrame according to predefined rules.

    Args:
        raw_df: DataFrame returned by the data acquisition functions.

    Returns:
        A cleaned and filtered DataFrame ready for further processing.
    """
    df = raw_df.copy()

    # 1. Calculate mid-price and bid-ask spread
    df['mid_price'] = (df['bid'] + df['ask']) / 2.0
    df['spread'] = df['ask'] - df['bid']

    # 2. Filter out zero-price or negative-price options
    df = df[df['mid_price'] > 0]

    # 3. Filter based on liquidity (volume and open interest)
    df = df[df['volume'] >= MIN_VOLUME]
    df = df[df['open_interest'] >= MIN_OPEN_INTEREST]

    # 4. Filter based on bid-ask spread
    # Avoid division by zero for options with no price
    df = df[(df['spread'] / df['mid_price']) < MAX_BID_ASK_SPREAD_RATIO]

    # 5. Filter for basic arbitrage violations (intrinsic value)
    # For calls, price must be >= S - K. For puts, >= K - S.
    df['intrinsic_value'] = 0.0
    if not df.empty:
        call_mask = df['option_type'] == 'call'
        put_mask = df['option_type'] == 'put'
        df.loc[call_mask, 'intrinsic_value'] = (df.loc[call_mask, 'underlying_price'] - df.loc[call_mask, 'strike']).clip(lower=0)
        df.loc[put_mask, 'intrinsic_value'] = (df.loc[put_mask, 'strike'] - df.loc[put_mask, 'underlying_price']).clip(lower=0)

        df = df[df['mid_price'] >= df['intrinsic_value']]

    # 6. Calculate time to maturity in years
    df['time_to_maturity'] = (df['maturity'] - pd.Timestamp.now()).dt.days / 365.25
    df = df[df['time_to_maturity'] > 0] # Filter out expired options

    # Select and reorder columns for a clean output
    final_cols = [
        'strike', 'maturity', 'option_type', 'mid_price', 'time_to_maturity',
        'underlying_price', 'volume', 'open_interest'
    ]
    return df[final_cols].reset_index(drop=True)
