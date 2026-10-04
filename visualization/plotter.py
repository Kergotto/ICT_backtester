import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Any
from domain.liquidity import LiquidityPool


def plot_liquidity_chart(df: pd.DataFrame, pools: List[LiquidityPool], swings: List[Dict[str, Any]],
                         title: str = "Liquidity Sweep Analysis") -> None:
    """
    Renders an interactive candlestick chart with overlaid liquidity pools and swing markers.
    """
    # 1. Convert the DatetimeIndex to strict strings.
    # Plotly treats strings as categories, automatically placing them side-by-side with no gaps!
    str_index = df.index.strftime('%Y-%m-%d %H:%M')

    fig = go.Figure(data=[go.Candlestick(
        x=str_index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Price Action"
    )])

    # Format swing times to exactly match the string index formatting
    high_times = [s['time'].strftime('%Y-%m-%d %H:%M') for s in swings if
                  s['type'] == 'SwingHigh' and s['time'] in df.index]
    high_prices = [s['price'] for s in swings if s['type'] == 'SwingHigh' and s['time'] in df.index]

    low_times = [s['time'].strftime('%Y-%m-%d %H:%M') for s in swings if
                 s['type'] == 'SwingLow' and s['time'] in df.index]
    low_prices = [s['price'] for s in swings if s['type'] == 'SwingLow' and s['time'] in df.index]

    # Overlay Swing Highs
    fig.add_trace(go.Scatter(
        x=high_times, y=high_prices, mode='markers',
        marker=dict(symbol='triangle-down', color='red', size=14, line=dict(color='black', width=1)),
        name='Engine Swing High'
    ))

    # Overlay Swing Lows
    fig.add_trace(go.Scatter(
        x=low_times, y=low_prices, mode='markers',
        marker=dict(symbol='triangle-up', color='green', size=14, line=dict(color='black', width=1)),
        name='Engine Swing Low'
    ))

    chart_start_time = df.index[0]

    for pool in pools:
        if not pool.is_active:
            color = "rgba(128, 128, 128, 0.3)"
            line_color = "gray"
        elif pool.has_been_swept:
            color = "rgba(255, 0, 0, 0.3)"
            line_color = "red"
        else:
            color = "rgba(0, 255, 0, 0.3)"
            line_color = "green"

        if getattr(pool, 'source', '') == "Macro_Session":
            if getattr(pool, 'session_origin', '') == "Asia":
                # Gold for Asia Session
                color = "rgba(255, 215, 0, 0.4)"
                line_color = "rgba(255, 215, 0, 1.0)"
            elif getattr(pool, 'session_origin', '') == "London":
                # Deep Sky Blue for London Session
                color = "rgba(0, 191, 255, 0.4)"
                line_color = "rgba(0, 191, 255, 1.0)"
            elif getattr(pool, 'session_origin', '') == "New York":
                # Purple / Orchid for NY Session
                color = "rgba(218, 112, 214, 0.4)"
                line_color = "rgba(218, 112, 214, 1.0)"

        end_time = df.index[-1]

        if not pool.is_active:
            # If the pool is dead, draw it all the way out to the exact BREAK event
            for event, timestamp in pool.events:
                if event == "BREAK":
                    end_time = timestamp - pd.Timedelta(minutes=1)
                    break

        draw_start = max(pool.creation_time, chart_start_time)

        idx_start = df.index.searchsorted(draw_start)
        idx_end = df.index.searchsorted(end_time)

        idx_start = max(0, min(idx_start, len(df) - 1))
        idx_end = max(0, min(idx_end, len(df) - 1))

        str_start = df.index[idx_start].strftime('%Y-%m-%d %H:%M')
        str_end = df.index[idx_end].strftime('%Y-%m-%d %H:%M')

        visual_lower = pool.lower_bound
        visual_upper = pool.upper_bound

        if visual_upper == visual_lower:
            visual_offset = 1.00
            if pool.pool_type == 'BSL':
                visual_upper += visual_offset
            elif pool.pool_type == 'SSL':
                visual_lower -= visual_offset

        session_name = getattr(pool, 'session_origin', 'Unknown')
        status = "Active" if pool.is_active else "Broken"

        # Strictly differentiate the title based on the engine source
        if getattr(pool, 'source', '') == "Macro_Session":
            title_text = f"{session_name} Macro {pool.pool_type}"
        else:
            title_text = f"Structural {pool.pool_type} (in {session_name})"

        hover_text = (
            f"<b>{title_text}</b><br>"
            f"Range: {visual_lower:.2f} - {visual_upper:.2f}<br>"
            f"Status: {status}<br>"
            f"Touches: {pool.touches}<br>"
            f"Time: {pool.creation_time.strftime('%Y-%m-%d %H:%M')}"
        )

        fig.add_trace(go.Scatter(
            x=[str_start, str_end, str_end, str_start, str_start],
            y=[visual_lower, visual_lower, visual_upper, visual_upper, visual_lower],
            fill='toself',
            fillcolor=color,
            opacity=0.6,
            line=dict(color=line_color, width=1),
            mode='lines',
            hoverinfo='text',
            text=hover_text,
            showlegend=False,
            hoverlabel=dict(bgcolor="black", bordercolor=line_color, font_size=12)
        ))

    fig.update_layout(
        title=title,
        yaxis_title="Price",
        xaxis_title="Time (NY)",
        xaxis_type='category',  # Forces Plotly to place categories side-by-side
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=800
    )

    fig.show()