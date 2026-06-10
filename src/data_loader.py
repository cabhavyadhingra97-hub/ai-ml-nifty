import yfinance as yf
import pandas as pd
import os

def load_data(start_date="2018-01-01", end_date=None, save_path="data/raw_data.csv"):
    """
    Fetches Nifty 50, India VIX, and USDINR data from Yahoo Finance.
    """
    tickers = {
        "Nifty50": "^NSEI",
        "VIX": "^INDIAVIX",
        "USDINR": "USDINR=X"
    }
    
    dfs = []
    for name, ticker in tickers.items():
        print(f"Downloading {name} ({ticker})...")
        data = yf.download(ticker, start=start_date, end=end_date)
        
        # Flatten multi-index columns if any
        if isinstance(data.columns, pd.MultiIndex):
            # For yfinance versions that return multi-level columns
            data.columns = [col[0] for col in data.columns]
            
        if name == "Nifty50":
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            data.columns = [f"Nifty_{col}" for col in data.columns]
        else:
            data = data[['Close']]
            data.columns = [f"{name}_Close"]
            
        dfs.append(data)
        
    # Merge all dataframes on index (Date)
    merged_df = pd.concat(dfs, axis=1)
    
    # Forward fill missing values (e.g. currency market open on stock market holiday)
    merged_df.ffill(inplace=True)
    
    # Drop any remaining NaNs (usually at the very beginning)
    merged_df.dropna(inplace=True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        merged_df.to_csv(save_path)
        print(f"Data saved to {save_path}")
        
    return merged_df

if __name__ == "__main__":
    df = load_data()
    print("Data load complete.")
    print(df.head())
    print(df.info())
