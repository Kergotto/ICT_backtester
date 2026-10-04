import pandas as pd


def detect_swings(df):
    """
    Scans the dataframe for strict 2-candle structural swings.
    Swing Low = 1 Bearish Close -> 1 Bullish Close (minimum wick of the two).
    Swing High = 1 Bullish Close -> 1 Bearish Close (maximum wick of the two).
    Handles dojis by evaluating their close relative to the previous candle's close.
    """
    swings = []

    # 1. Evaluate synthetic colors for the entire DataFrame upfront
    prev_closes = df['close'].shift(1)

    is_green = df['close'] > df['open']
    is_red = df['close'] < df['open']
    is_doji = df['close'] == df['open']

    # Resolve dojis: Bullish if closed >= prev close, Bearish if closed < prev close
    bullish_mask = is_green | (is_doji & (df['close'] >= prev_closes))
    bearish_mask = is_red | (is_doji & (df['close'] < prev_closes))

    # 2. Scan for 2-candle sequences
    # Start at index 1 so we can always look back at exactly 1 previous candle
    for i in range(1, len(df)):

        # Wicks for extreme isolation
        prev_low = df['low'].iloc[i - 1]
        prev_high = df['high'].iloc[i - 1]
        curr_low = df['low'].iloc[i]
        curr_high = df['high'].iloc[i]

        # Use our pre-calculated synthetic colors
        prev_is_bearish = bearish_mask.iloc[i - 1]
        prev_is_bullish = bullish_mask.iloc[i - 1]
        curr_is_bearish = bearish_mask.iloc[i]
        curr_is_bullish = bullish_mask.iloc[i]

        # LOGIC 1: Detect Swing Low (Red -> Green)
        if prev_is_bearish and curr_is_bullish:
            # Isolate the extreme low between ONLY these two candles
            actual_low = min(prev_low, curr_low)
            swings.append({
                'type': 'SwingLow',
                'price': actual_low,
                'time': df.index[i]  # Anchors to the transition candle
            })

        # LOGIC 2: Detect Swing High (Green -> Red)
        elif prev_is_bullish and curr_is_bearish:
            # Isolate the extreme high between ONLY these two candles
            actual_high = max(prev_high, curr_high)
            swings.append({
                'type': 'SwingHigh',
                'price': actual_high,
                'time': df.index[i]  # Anchors to the transition candle
            })

    return swings