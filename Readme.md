"8: Common Beginner Mistakes to Avoid
Lookahead Bias: This is the deadliest sin in quantitative finance. If your 1-hour candle logic uses the "Close" price to make a decision at 10:15 AM, you are peeking into the future. You only know the 1H close at 11:00 AM. Your engine must enforce temporal strictness.

The CSV Trap: 10 years of 1-minute data for two assets is roughly 7 million rows. Loading this from CSV every time you run a test will waste hours of your life. Convert your raw CSVs to Parquet format immediately. It is strongly typed, highly compressed, and loads 10x to 50x faster in Pandas.

Rigid Coupling: Tying your strategy logic directly to your broker simulation. If you want to change how a BOS is calculated, you shouldn't have to touch the TradeManager.

Premature Optimization: Trying to make the backtest run in milliseconds before verifying it is mathematically correct. Build it accurately first; make it fast later.

9: 18-Month Development Roadmap
Data & Primitives
Months 1-3
Implement Parquet data loading, strict timeframe resampling (1m to 4H), and write comprehensive Pytest coverage to ensure data integrity.

Domain Logic Translation
Months 4-6
Translate subjective concepts (Swings, Liquidity, BOS, CHoCH) into pure, stateless Python functions. Output these markers visually using Plotly to verify they match human discretion.

Event-Driven Engine
Months 7-9
Build the broker.py and portfolio.py. Create the time-stepping loop that feeds data to the strategy minute-by-minute without leaking future data.

Strategy Implementation
Months 10-12
Combine the multi-timeframe signals. Implement the specific entry models, stop-loss logic, and take-profit targets based on the domain definitions.

Analytics & Reporting
Months 13-15
Develop the tear sheets. Calculate Sharpe ratio, maximum drawdown, win rate, and profit factor. Implement logging to trace every decision the engine makes.

Optimization & UI
Months 16-18
Parameterize the strategy for grid searching (e.g., testing different required distances for a liquidity sweep). Optionally integrate PySide6 for a desktop control panel and write the final thesis documentation.