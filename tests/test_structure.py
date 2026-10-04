import pytest
import pandas as pd
import numpy as np
import domain.structure

def test_choch_and_bos_detection():
    # 1. Create synthetic data representing a clean trend cycle
    dates = pd.date_range(start='2026-07-08 09:30', periods=6, freq='1min')

    # Aligned Candle 0 open/high downwards so body_high (95) < SwingHigh (98.0)
    df = pd.DataFrame({
        'open':  [95, 90, 80,  95, 108, 105],
        'high':  [100, 95, 85, 110, 112, 120],
        'low':   [85,  85, 75,  90, 100, 100],
        'close': [90,  80, 85, 105, 105, 115]
    }, index=dates)

    # 2. Inject synthetic swings that our engine should react to
    mock_swings = [
        {'type': 'SwingHigh', 'price': 98.0, 'time': dates[0]},
        {'type': 'SwingLow',  'price': 78.0, 'time': dates[2]},
        {'type': 'SwingHigh', 'price': 112.0, 'time': dates[4]},
    ]

    # 3. Run the engine
    result_df = domain.structure.detect_structure(df, mock_swings)

    # 4. Assertions to prove the logic works
    assert result_df['choch_bullish'].iloc[3] == True, "Engine failed to detect Bullish CHoCH"
    assert result_df['trend'].iloc[3] == 1, "Trend state did not flip to 1 (Uptrend)"

    assert result_df['bos_bullish'].iloc[5] == True, "Engine failed to detect Bullish BOS"
    assert result_df['trend'].iloc[5] == 1, "Trend state should remain 1 (Uptrend)"

    assert not result_df['choch_bearish'].any(), "False Bearish CHoCH detected"
    assert not result_df['bos_bearish'].any(), "False Bearish BOS detected"