import yfinance as yf
from pathlib import Path


def download_recent_futures(symbol: str, filename: str) -> None:
    """
    Downloads the last 7 days of 1-minute historical data from Yahoo Finance.

    Args:
        symbol (str): The Yahoo Finance ticker (e.g., 'ES=F' for E-Mini S&P 500, 'NQ=F' for Nasdaq).
        filename (str): The name of the output CSV file.
    """
    print(f"Downloading 1-minute data for {symbol}...")

    # Initialize the yfinance Ticker
    ticker = yf.Ticker(symbol)

    # Yahoo Finance strictly limits 1m data to the last 7 days
    # We use period="7d" and interval="1m"
    df = ticker.history(period="7d", interval="1m")

    if df.empty:
        print(f"Failed to retrieve data for {symbol}. The market might be closed or the symbol is incorrect.")
        return

    # yfinance returns columns like 'Open', 'High', 'Low', 'Close', 'Volume'
    # We will reset the index to make the Datetime a standard column before saving
    df = df.reset_index()

    # Ensure the output directory exists
    output_dir = Path("C:/Users/Кирил/PycharmProjects/ICT_Backtester/ICT_Backtester/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    # Save raw data to CSV
    df.to_csv(output_path, index=False)
    print(f"Successfully saved {len(df)} rows to {output_path.resolve()}")


if __name__ == "__main__":
    # Execute the downloads for both ES and NQ
    download_recent_futures(symbol="ES=F", filename="es_1m_sample.csv")
    download_recent_futures(symbol="NQ=F", filename="nq_1m_sample.csv")