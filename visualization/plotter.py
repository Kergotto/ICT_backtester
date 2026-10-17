import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Any
from domain.liquidity import LiquidityPool
import time
import os
import time
import webbrowser


def plot_liquidity_chart(df: pd.DataFrame, pools: List[LiquidityPool], swings: List[Dict[str, Any]],
                         fvgs: List[Any] = None, title: str = "Liquidity Sweep Analysis") -> None:
    """
    Renders an interactive candlestick chart with overlaid liquidity pools, swing markers, and FVGs.
    """
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
    chart_end_time = df.index[-1]
    chart_max_high = df['high'].max()
    chart_min_low = df['low'].min()

    for pool in pools:
        # Only check events that happened within the visible chart timeframe
        break_time = next((ts for ev, ts in pool.events if ev == "BREAK" and ts <= chart_end_time), None)
        swept_in_window = any(ev == "SWEEP" and ts <= chart_end_time for ev, ts in pool.events)

        # 1. Never draw pools that were created in the future (after the right edge of the chart)
        if pool.creation_time > chart_end_time:
            continue

        # 2. Skip pools from previous days that sit outside the visible chart's price range
        if pool.creation_time < chart_start_time:
            if pool.lower_bound > chart_max_high or pool.upper_bound < chart_min_low:
                continue

        if break_time is not None:
            color = "rgba(128, 128, 128, 0.3)"
            line_color = "gray"
            status = "Broken"
            end_time = break_time - pd.Timedelta(minutes=1)
        elif swept_in_window:
            color = "rgba(255, 0, 0, 0.3)"
            line_color = "red"
            status = "Swept"
            end_time = chart_end_time
        else:
            color = "rgba(0, 255, 0, 0.3)"
            line_color = "green"
            status = "Active"
            end_time = chart_end_time

        if getattr(pool, 'source', '') == "Macro_Session":
            if getattr(pool, 'session_origin', '') == "Asia":
                color = "rgba(255, 215, 0, 0.4)"
                line_color = "rgba(255, 215, 0, 1.0)"
            elif getattr(pool, 'session_origin', '') == "London":
                color = "rgba(0, 191, 255, 0.4)"
                line_color = "rgba(0, 191, 255, 1.0)"
            elif getattr(pool, 'session_origin', '') == "New York":
                color = "rgba(218, 112, 214, 0.4)"
                line_color = "rgba(218, 112, 214, 1.0)"

        draw_start = max(pool.creation_time, chart_start_time)

        idx_start = df.index.searchsorted(draw_start)
        idx_end = df.index.searchsorted(end_time)

        idx_start = max(0, min(idx_start, len(df) - 1))
        idx_end = max(0, min(idx_end, len(df) - 1))

        str_start = df.index[idx_start].strftime('%Y-%m-%d %H:%M')
        str_end = df.index[idx_end].strftime('%Y-%m-%d %H:%M')

        # Get the exact touches and bounds of the pool at the end of the visible chart window
        valid_history = [h for h in getattr(pool, 'touch_history', []) if h[0] <= end_time]
        if valid_history:
            _, visible_touches, visual_lower, visual_upper = valid_history[-1]
        else:
            visible_touches = pool.touches
            visual_lower = pool.lower_bound
            visual_upper = pool.upper_bound

        if visual_upper - visual_lower > 1.00:
            if pool.pool_type == 'BSL':
                visual_lower = visual_upper - 1.00
            elif pool.pool_type == 'SSL':
                visual_upper = visual_lower + 1.00
        elif visual_upper == visual_lower:
            visual_offset = 0.25
            if pool.pool_type == 'BSL':
                visual_lower -= visual_offset
            elif pool.pool_type == 'SSL':
                visual_upper += visual_offset

        session_name = getattr(pool, 'session_origin', 'Unknown')

        if getattr(pool, 'source', '') == "Macro_Session":
            title_text = f"{session_name} Macro {pool.pool_type}"
        else:
            title_text = f"Structural {pool.pool_type} (in {session_name})"

        hover_text = (
            f"<b>{title_text}</b><br>"
            f"Range: {visual_lower:.2f} - {visual_upper:.2f}<br>"
            f"Status: {status}<br>"
            f"Touches: {visible_touches}<br>"
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

    if fvgs:
        for fvg in fvgs:
            if fvg.creation_time > chart_end_time or fvg.creation_time < chart_start_time:
                continue

            # Reconstruct the visible status and end time based ONLY on events within the chart window
            end_time = chart_end_time
            display_status = "Active"

            for ev, ts, _ in fvg.events:
                if ts <= chart_end_time:
                    if ev == "INVERTED":
                        end_time = ts
                        display_status = "Inverted"
                        break
                    elif ev in ["FILLED_BY_CLOSE", "FILLED_BY_WICK"]:
                        display_status = "Filled"
                    elif ev == "MITIGATED" and display_status not in ["Filled", "Inverted"]:
                        display_status = "Partially_Mitigated"

            idx_start = df.index.searchsorted(fvg.creation_time)
            idx_end = df.index.searchsorted(end_time)
            idx_start = max(0, min(idx_start, len(df) - 1))
            idx_end = max(0, min(idx_end, len(df) - 1))

            str_start = df.index[idx_start].strftime('%Y-%m-%d %H:%M')
            str_end = df.index[idx_end].strftime('%Y-%m-%d %H:%M')

            color = "rgba(0, 150, 255, 0.15)" if fvg.fvg_type == 'BISI' else "rgba(255, 100, 100, 0.15)"
            line_color = "rgba(0, 150, 255, 0.5)" if fvg.fvg_type == 'BISI' else "rgba(255, 100, 100, 0.5)"

            hover_text = (
                f"<b>{fvg.fvg_type} FVG</b><br>"
                f"Range: {fvg.bottom:.2f} - {fvg.top:.2f}<br>"
                f"CE (50%): {fvg.ce:.2f}<br>"
                f"Status: {display_status}<br>"
                f"Time: {fvg.creation_time.strftime('%Y-%m-%d %H:%M')}"
            )

            # Draw the FVG Box
            fig.add_trace(go.Scatter(
                x=[str_start, str_end, str_end, str_start, str_start],
                y=[fvg.bottom, fvg.bottom, fvg.top, fvg.top, fvg.bottom],
                fill='toself',
                fillcolor=color,
                opacity=0.7,
                line=dict(color=line_color, width=1, dash='dot'),
                mode='lines',
                hoverinfo='text',
                text=hover_text,
                showlegend=False,
                hoverlabel=dict(bgcolor="black", bordercolor=line_color, font_size=12)
            ))

            # Draw the CE (50%) Dashed Line inside the FVG
            fig.add_trace(go.Scatter(
                x=[str_start, str_end],
                y=[fvg.ce, fvg.ce],
                mode='lines',
                line=dict(color=line_color, width=1, dash='dash'),
                hoverinfo='skip',
                showlegend=False
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

    filename = "es_chart.html" if "ES" in title else "nq_chart.html"
    filepath = os.path.abspath(filename)

    fig.write_html(filepath, auto_open=False)

    webbrowser.open(f"file:///{filepath}")
    time.sleep(1)