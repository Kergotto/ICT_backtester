# Programmatic Backtesting System for Trading Strategies on Financial Time Series with Adaptive Parameter Tuning and LLM-Based Result Explanations

An event-driven, high-precision backtesting framework designed specifically for NQ and ES futures. This project translates highly subjective Inner Circle Trader (ICT) concepts—such as Market Structure Shifts, Liquidity Pools, Fair Value Gaps, Single-Leg Equilibrium, and Multi-Swing Fibonacci Retracements—into strict, stateless algorithmic logic.

Built entirely in Python, the engine processes historical data to execute structural analysis and bridge the gap between discretionary price action concepts and rigorous quantitative verification.

## Core Design Principles

-   **Zero Lookahead Bias:** The deadliest sin in quantitative finance is peeking into the future. This engine enforces strict temporal boundaries; all structural and entry logic is evaluated strictly on closed candle states to ensure mathematically valid test results.
-   **Modular Architecture:** Strategy logic is decoupled from broker simulation. Modifications to how a swing point or Fibonacci level is calculated do not require re-engineering the core portfolio or execution manager.
-   **Accuracy Over Speed:** The primary objective is absolute mathematical precision and visual chart alignment over raw execution speed. 

## Development Roadmap (Academic Schedule)

This project follows an 8-month academic development lifecycle aligned with bachelor's thesis requirements.

### Stage 1: Simulation Engine & Core Primitives (Months 1–2)
- [x] Implement historical data loading and timeframe resampling engines.
- [x] Translate subjective swings and market structure into stateless Python functions.
- [x] Implement spatial indexing mechanism for liquidity pool tracking and Finite State Machines for pool lifecycles.
- [x] Build Fair Value Gap (FVG) and SMT Divergence engines.
- [x] Implement Single-Leg Equilibrium (0 - 0.5 - 1) and Multi-Swing Fibonacci Retracement & OTE (0.62 - 0.705 - 0.79) tracking.
- [x] Output domain markers visually using Plotly to verify visual chart alignment.
- [ ] Build localized `broker.py` and `portfolio.py` execution modules with step-by-step event loops preventing look-ahead bias.

### Stage 2: Performance Metrics & Baseline Models (Months 3–4)
- [ ] Integrate quantitative performance metrics (Total return, Sharpe ratio, Maximum drawdown, Win rate, Profit factor).
- [ ] Implement baseline benchmark strategies: Random Entries (with identical risk parameters) and Buy-and-Hold.
- [ ] Run comparative backtests to evaluate whether formal ICT setups exhibit statistical advantage over random entries.

### Stage 3: Optimization & Adaptivity (Month 5)
- [ ] Parameterize system constants (liquidity tolerances, macro time windows) into configuration files.
- [ ] Implement Walk-Forward optimization (In-Sample / Out-of-Sample windows) for adaptive parameter tuning.

### Stage 4: LLM-Based Analytics & Interpretation (Month 6)
- [ ] Integrate a structured LLM explanation module to interpret quantified backtest metrics, performance summaries, and identified patterns in natural language.

### Stage 5: Advanced Pattern Mining & Finalization (Months 7–8)
- [ ] (Optional) Apply Association Rule Mining to extract multi-factor pattern combinations.
- [ ] Finalize thesis documentation, performance reports, and defense presentation.