# Programmatic Backtesting System for Trading Strategies on Financial Time Series with Adaptive Parameter Tuning and LLM-Based Result Explanations

An event-driven, high-precision backtesting framework designed specifically for NQ and ES futures. This project translates highly subjective Inner Circle Trader (ICT) concepts—such as Market Structure Shifts, Liquidity Pools, Fair Value Gaps, Single-Leg Equilibrium, and Multi-Swing Fibonacci Retracements—into strict, stateless algorithmic logic.

Built entirely in Python, the engine processes historical data to execute structural analysis and bridge the gap between discretionary price action concepts and rigorous quantitative verification via a **Hybrid Quant Architecture** (combining deterministic logic, Machine Learning, and LLM translation).

### Core Design Principles

- **Zero Lookahead Bias:** The deadliest sin in quantitative finance is peeking into the future. This engine enforces strict temporal boundaries; all structural and entry logic is evaluated strictly on closed candle states to ensure mathematically valid test results.
- **Modular Architecture:** Strategy logic is decoupled from broker simulation. Modifications to how a swing point or Fibonacci level is calculated do not require re-engineering the core portfolio or execution manager.
- **Accuracy Over Speed:** The primary objective is absolute mathematical precision and visual chart alignment over raw execution speed.
- **Deterministic Guardrails:** The system hardcodes "veteran trader" experience (e.g., blocking counter-trend shorts into All-Time Highs, preventing entries after exhausted 200+ point expansion legs) to prevent machine learning models from curve-fitting to random noise.

---

### Development Roadmap (Academic Schedule)

This project follows an 8-month academic development lifecycle aligned with bachelor's thesis requirements.

#### Stage 1: Simulation Engine & Core Primitives (Months 1–2)
- [x] Implement historical data loading and timeframe resampling engines.
- [x] Translate subjective swings and market structure into stateless Python functions.
- [x] Implement spatial indexing mechanism for liquidity pool tracking and Finite State Machines for pool lifecycles.
- [x] Build Fair Value Gap (FVG) and SMT Divergence engines.
- [x] Implement Single-Leg Equilibrium (0 - 0.5 - 1) and Multi-Swing Fibonacci Retracement & OTE (0.62 - 0.705 - 0.79) tracking.
- [x] Output domain markers visually using Plotly to verify visual chart alignment.
- [x] Build localized `broker.py` and `portfolio.py` execution modules with step-by-step event loops preventing look-ahead bias.

#### Stage 2: Performance Metrics & Baseline Models (Months 3–4)
- [x] Integrate quantitative performance metrics (Total return, Sharpe ratio, Maximum drawdown, Win rate, Profit factor).
- [x] Implement baseline benchmark strategies: Random Entries (with identical risk parameters) and Buy-and-Hold.
- [x] Run comparative backtests to evaluate whether formal ICT setups exhibit statistical advantage over random entries.
- [x] Generate an Automated Trade Audit Log with precise execution timestamps, multi-tier partial take-profits, and exact PnL dollar figures.
- [x] Implement dynamic Micro-contract (MES/MNQ) fallback sizing to enforce strict dollar-risk limits ($800–$1,100) on highly volatile setups.

#### Stage 3: Machine Learning Optimization & Adaptivity (Month 5)
- [ ] Parameterize system constants (liquidity tolerances, macro time windows) into configuration files.
- [ ] Implement Walk-Forward optimization (In-Sample / Out-of-Sample windows) for adaptive parameter tuning.
- [ ] **Machine Learning Optimization:** Feed historical execution data through an ML optimizer to analyze thousands of executed trades. Bounded by the engine's deterministic guardrails, the ML model will identify valid statistical edges and hidden correlations (e.g., time-of-day, volatility, stop-loss widths) without curve-fitting.

#### Stage 4: LLM-Based Analytics & Translation (Month 6)
- [ ] **LLM Translation Agent:** Integrate a structured LLM explanation module to act as a bridge between the complex quantitative output and human readability.
- [ ] Interpret quantified backtest metrics, ML optimization matrices, and performance summaries into plain-English, actionable trading insights (e.g., translating raw Sharpe ratios and variance into clear strategy adjustments).

#### Stage 5: Advanced Pattern Mining & Finalization (Months 7–8)
- [ ] (Optional) Apply Association Rule Mining to extract multi-factor pattern combinations.
- [ ] Finalize thesis documentation, performance reports, and defense presentation.
