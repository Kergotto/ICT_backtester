import pandas as pd

def print_structure_events(df: pd.DataFrame) -> None:
    """
    Filters the DataFrame for structural breaks and prints a clean chronological log.
    """
    # Create a mask for any row that has a true flag
    mask = (
            df['choch_bullish'] |
            df['bos_bullish'] |
            df['choch_bearish'] |
            df['bos_bearish']
    )

    events = df[mask]

    print("\n" + "=" * 50)
    print("📈 STRUCTURAL EVENTS LOG")
    print("=" * 50)

    if events.empty:
        print("No structural events detected in this timeframe.")
        return

    for time, row in events.iterrows():
        event_type = ""
        if row['choch_bullish']:
            event_type = "🟢 CHoCH (Bullish Reversal)"
        elif row['bos_bullish']:
            event_type = "🟩 BOS (Bullish Continuation)"
        elif row['choch_bearish']:
            event_type = "🔴 CHoCH (Bearish Reversal)"
        elif row['bos_bearish']:
            event_type = "🟥 BOS (Bearish Continuation)"

        print(f"Time: {time} | Event: {event_type:<28} | Break Price: {row['close']:.2f} | Trend State: {row['trend']}")

    print("=" * 50 + "\n")