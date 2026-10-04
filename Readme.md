# Quantitative ICT Backtesting Engine

An event-driven, high-precision backtesting framework designed specifically for NQ and ES futures. This project translates highly subjective Inner Circle Trader (ICT) concepts—such as Market Structure Shifts (CHoCH), Break of Structure (BOS), and Macro Session Liquidity—into strict, stateless algorithmic logic. 

Built entirely in Python, the engine processes 1-minute tick data to execute gap-aware structural analysis. It is engineered to bridge the gap between discretionary price action concepts and rigorous quantitative verification.

## Core Design Principles

*   **Zero Lookahead Bias:** The deadliest sin in quantitative finance is peeking into the future. This engine enforces strict temporal boundaries; all structural and entry logic is evaluated strictly on closed candle states to ensure mathematically valid test results.
*   **High-Performance Data Pipeline:** Intraday testing across years of 1-minute data generates millions of rows. The engine bypasses traditional CSV loading in favor of Parquet, providing highly compressed, strongly typed data that loads exponentially faster in Pandas.
*   **Modular Architecture:** Strategy logic is decoupled from broker simulation. Modifications to how a BOS or swing point is calculated do not require re-engineering the core portfolio or execution manager.
*   **Accuracy Over Speed:** The primary objective is absolute mathematical precision and visual chart alignment over raw execution speed. Optimization and microsecond processing are deferred until the foundational logic is perfectly mapped to human discretion.

## Development Roadmap

This project follows an 18-month staged development lifecycle, currently actively executing Phase 2.

### Phase 1: Data & Primitives (Months 1-3)
*   [x] Implement highly optimized Parquet data loading infrastructure.
*   [x] Establish strict timeframe resampling engines (1m up to 4H).
*   [ ] Write comprehensive Pytest coverage to ensure baseline data integrity.

### Phase 2: Domain Logic Translation (Months 4-6)
*   [x] Translate subjective swings into pure, stateless Python functions.
*   [x] Develop gap-aware structural break detection (BOS / CHoCH).
*   [x] Implement macro session liquidity pooling (Asia / London).
*   [ ] Output domain markers visually using Plotly to verify visual chart alignment.

### Phase 3: Event-Driven Engine (Months 7-9)
*   [ ] Build the localized `broker.py` and `portfolio.py` execution modules.
*   [ ] Create the time-stepping loop that feeds data to the strategy minute-by-minute without leaking future data.

### Phase 4: Strategy Implementation (Months 10-12)
*   [ ] Combine multi-timeframe signals into actionable trade logic.
*   [ ] Implement programmatic entry models (e.g., Fair Value Gaps, Order Blocks).
*   [ ] Construct dynamic stop-loss logic and take-profit targeting based on structural domain definitions.

### Phase 5: Analytics & Reporting (Months 13-15)
*   [ ] Develop comprehensive tear sheets for performance review.
*   [ ] Calculate core quantitative metrics: Sharpe ratio, maximum drawdown, win rate, and profit factor.
*   [ ] Implement deep-trace logging to record every execution decision the engine makes.

### Phase 6: Optimization & UI (Months 16-18)
*   [ ] Parameterize the strategy for grid searching (e.g., testing optimal required tick distances for liquidity sweeps).
*   [ ] Optionally integrate PySide6 for a local desktop control panel.
*   [ ] Finalize thesis documentation and methodology reporting.
