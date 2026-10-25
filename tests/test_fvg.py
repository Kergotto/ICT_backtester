import pytest
import pandas as pd
from domain.fvg import FairValueGap, detect_fvgs


def create_df(data: list) -> pd.DataFrame:
    """Helper to generate a 1-minute OHLC DataFrame."""
    return pd.DataFrame(
        data,
        columns=['open', 'high', 'low', 'close'],
        index=pd.date_range('2026-01-01 09:30', periods=len(data), freq='1min')
    )


def test_detect_bisi_and_sibi():
    # Test BISI Bullish FVG Detection
    # Candle 1 High = 100, Candle 3 Low = 105 -> Gap: 100 to 105
    df_bisi = create_df([
        [90, 100, 80, 95],  # C1
        [95, 120, 90, 115],  # C2
        [115, 130, 105, 125]  # C3
    ])
    fvgs = detect_fvgs(df_bisi)
    assert len(fvgs) == 1
    assert fvgs[0].fvg_type == 'BISI'
    assert fvgs[0].bottom == 100.0
    assert fvgs[0].top == 105.0

    # Test Bearish FVG Detection
    # Candle 1 Low = 200, Candle 3 High = 195 -> Gap: 195 to 200
    df_sibi = create_df([
        [210, 220, 200, 205],  # C1
        [205, 210, 180, 185],  # C2
        [185, 195, 170, 175]  # C3
    ])
    fvgs2 = detect_fvgs(df_sibi)
    assert len(fvgs2) == 1
    assert fvgs2[0].fvg_type == 'SIBI'
    assert fvgs2[0].bottom == 195.0
    assert fvgs2[0].top == 200.0


def test_bisi_lifecycle():
    ts = pd.Timestamp('2026-01-01 10:00')
    fvg = FairValueGap(fvg_type='BISI', top=105.0, bottom=100.0, creation_time=ts)

    # Ignore candles from before or during creation time
    fvg.evaluate_candle(high=110, low=106, close=108, timestamp=ts)
    assert fvg.status == "Active"

    # 1. Partial Mitigation: Wick drops to 102, but body closes above gap at 106
    fvg.evaluate_candle(high=110, low=102, close=106, timestamp=pd.Timestamp('2026-01-01 10:01'))
    assert fvg.status == "Partially_Mitigated"
    assert fvg.unmitigated_top == 102.0

    # 2. Filled: Body closes inside the gap at 101
    fvg.evaluate_candle(high=105, low=99, close=101, timestamp=pd.Timestamp('2026-01-01 10:02'))
    assert fvg.status == "Filled"
    assert fvg.unmitigated_top == 99.0

    # 3. Inverted: Body closes below the gap's bottom
    fvg.evaluate_candle(high=100, low=90, close=95, timestamp=pd.Timestamp('2026-01-01 10:03'))
    assert fvg.status == "Inverted"


def test_sibi_lifecycle():
    ts = pd.Timestamp('2026-01-01 10:00')
    fvg = FairValueGap(fvg_type='SIBI', top=200.0, bottom=195.0, creation_time=ts)

    # 1. Partial Mitigation: Wick spikes to 198, body closes below gap at 194
    fvg.evaluate_candle(high=198, low=190, close=194, timestamp=pd.Timestamp('2026-01-01 10:01'))
    assert fvg.status == "Partially_Mitigated"
    assert fvg.unmitigated_bottom == 198.0

    # 2. Inverted: Sudden news spike closes body above the gap at 202 (skips "Filled")
    fvg.evaluate_candle(high=205, low=195, close=202, timestamp=pd.Timestamp('2026-01-01 10:02'))
    assert fvg.status == "Inverted"