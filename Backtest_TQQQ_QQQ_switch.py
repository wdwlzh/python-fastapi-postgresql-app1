import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "data_cache"  # directory to save CSV files
os.makedirs(DATA_DIR, exist_ok=True)

def get_price_data(ticker, start="2005-01-01"):
    """Load price data from local CSV if exists, otherwise download from Yahoo Finance."""
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")

    if os.path.exists(file_path):
        print(f"Loading {ticker} from local CSV...")
        df = pd.read_csv(file_path, index_col=0, parse_dates=True, date_format="%Y-%m-%d")


        # Ensure numeric columns are floats
        df = df.apply(pd.to_numeric, errors="coerce")
    else:
        print(f"Downloading {ticker} from Yahoo Finance...")
        df = yf.download(ticker, start=start, auto_adjust=True)

        # Save only needed columns to avoid parsing issues
        df = df[["Close"]]
        df.to_csv(file_path)

    return df


def backtest(strategy_func, tickers=("QQQ", "TQQQ"), start="2005-01-01", name="strategy"):
    """Run a backtest on given strategy function, save equity curve plot and CSV results."""
    qqq = get_price_data(tickers[0], start)
    tqqq = get_price_data(tickers[1], start)

    # Use Close (already adjusted if auto_adjust=True)
    prices = pd.DataFrame({
        "QQQ": qqq["Close"],
        "TQQQ": tqqq["Close"]
    }).dropna()

    # Run strategy
    signals = strategy_func(prices)
    daily_returns = (signals.shift(1) * prices.pct_change()).sum(axis=1)
    portfolio = daily_returns.cumsum().apply(lambda x: (1 + x))

    # Save portfolio equity curve CSV
    csv_filename = f"{name}_equity_curve.csv"
    portfolio.to_csv(csv_filename, header=["Equity"])
    print(f"Saved equity data to {csv_filename}")

    # Plot and save PNG
    plt.figure(figsize=(10, 6))
    portfolio.plot(title=f"{name} Equity Curve")
    png_filename = f"{name}_equity_curve.png"
    plt.savefig(png_filename)
    plt.close()
    print(f"Saved plot to {png_filename}")

    return portfolio



# === Example strategies ===

def sma_crossover(prices, short=50, long=200):
    """Switch between QQQ and TQQQ using SMA crossover filter."""
    sma_short = prices["QQQ"].rolling(short).mean()
    sma_long = prices["QQQ"].rolling(long).mean()

    signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    signals.loc[sma_short > sma_long, "TQQQ"] = 1  # uptrend → TQQQ
    signals.loc[sma_short <= sma_long, "QQQ"] = 1  # downtrend/sideways → QQQ
    return signals

def rsi_filter(prices, period=14, threshold=50):
    """Switch based on RSI (momentum)."""
    delta = prices["QQQ"].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    signals.loc[rsi > threshold, "TQQQ"] = 1
    signals.loc[rsi <= threshold, "QQQ"] = 1
    return signals

def vol_adjusted(prices, lookback=20):
    """Adjust allocation: more TQQQ when volatility is low."""
    vol = prices["QQQ"].pct_change().rolling(lookback).std()
    signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    signals["QQQ"] = 0.5
    signals["TQQQ"] = 0.5

    # overweight TQQQ when vol is below median
    signals.loc[vol < vol.median(), "TQQQ"] = 0.8
    signals.loc[vol < vol.median(), "QQQ"] = 0.2

    # overweight QQQ when vol is high
    signals.loc[vol >= vol.median(), "QQQ"] = 0.8
    signals.loc[vol >= vol.median(), "TQQQ"] = 0.2
    return signals


if __name__ == "__main__":
    print("Running SMA crossover strategy...")
    backtest(sma_crossover, name="sma_crossover")

    print("Running RSI filter strategy...")
    backtest(rsi_filter, name="rsi_filter")

    print("Running Volatility-adjusted strategy...")
    backtest(vol_adjusted, name="vol_adjusted")

