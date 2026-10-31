import os
import time
import webbrowser
from typing import List, Dict, Any
import pandas as pd
import plotly.graph_objects as go
from domain.liquidity import LiquidityPool


def plot_liquidity_chart(
    df: pd.DataFrame,
    pools: List[LiquidityPool],
    swings: List[Dict[str, Any]],
    fvgs: List[Any] = None,
    ranges: List[Any] = None,
    fibs: List[Any] = None,
    title: str = "Liquidity Sweep Analysis"
) -> None:
    """
    Renders an interactive candlestick chart with overlaid liquidity pools,
    swing markers, FVGs, single-leg Equilibrium boxes, and multi-swing Fibonacci OTE zones.
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
        break_time = next((ts for ev, ts in pool.events if ev == "BREAK" and ts <= chart_end_time), None)
        swept_in_window = any(ev == "SWEEP" and ts <= chart_end_time for ev, ts in pool.events)

        if pool.creation_time > chart_end_time:
            continue

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
        is_macro = getattr(pool, 'source', '') == "Macro_Session"

        if is_macro:
            title_text = f"{session_name} Macro {pool.pool_type}"
            border_width = 2.0
            trace_opacity = 0.75
        else:
            title_text = f"Structural {pool.pool_type} (in {session_name})"
            border_width = 1.0
            trace_opacity = 0.6

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
            opacity=trace_opacity,
            line=dict(color=line_color, width=border_width),
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

            fig.add_trace(go.Scatter(
                x=[str_start, str_end],
                y=[fvg.ce, fvg.ce],
                mode='lines',
                line=dict(color=line_color, width=1, dash='dash'),
                hoverinfo='skip',
                showlegend=False
            ))

    # Overlay Single-Leg Equilibrium Boxes (Strictly from Anchor Swing to Next Swing)
    if ranges:
        shown_eq_legend = False
        for dr in ranges:
            if dr.creation_time < chart_start_time or dr.start_time > chart_end_time:
                continue

            # Left border = Anchor Swing (dr.start_time), Right border = Completing Swing (dr.creation_time)
            draw_start = max(dr.start_time, chart_start_time)
            draw_end = min(dr.creation_time, chart_end_time)

            idx_start = max(0, min(df.index.searchsorted(draw_start), len(df) - 1))
            idx_end = max(0, min(df.index.searchsorted(draw_end), len(df) - 1))
            if idx_end <= idx_start and idx_start < len(df) - 1:
                idx_end = idx_start + 1

            str_start = df.index[idx_start].strftime('%Y-%m-%d %H:%M')
            str_end = df.index[idx_end].strftime('%Y-%m-%d %H:%M')

            if dr.break_time is not None and dr.break_time <= chart_end_time:
                vis_status = "Broken"
            elif dr.eq_mitigation_time is not None and dr.eq_mitigation_time <= chart_end_time:
                vis_status = "EQ_Mitigated"
            else:
                vis_status = "Active"

            box_color = "rgba(0, 255, 255, 0.08)"
            border_color = "rgba(0, 255, 255, 0.85)"
            eq_color = "rgba(0, 255, 255, 1.0)"

            hover_text = (
                f"<b>{dr.direction} Equilibrium (Single Leg)</b><br>"
                f"1.0 High (Top Border): {dr.high:.2f}<br>"
                f"0.5 Equilibrium (50%): {dr.eq:.2f}<br>"
                f"0.0 Low (Bottom Border): {dr.low:.2f}<br>"
                f"Status: {vis_status}<br>"
                f"Start Swing: {dr.start_time.strftime('%Y-%m-%d %H:%M')}<br>"
                f"End Swing: {dr.creation_time.strftime('%Y-%m-%d %H:%M')}"
            )

            # 1. Equilibrium Box (Swing 1 to Swing 2)
            fig.add_trace(go.Scatter(
                x=[str_start, str_end, str_end, str_start, str_start],
                y=[dr.low, dr.low, dr.high, dr.high, dr.low],
                fill='toself',
                fillcolor=box_color,
                line=dict(color=border_color, width=1.5, dash='solid'),
                mode='lines',
                hoverinfo='text',
                text=hover_text,
                legendgroup="equilibrium",
                showlegend=False,
                hoverlabel=dict(bgcolor="black", bordercolor="cyan", font_size=12)
            ))

            # 2. 0.5 Equilibrium Center Line
            fig.add_trace(go.Scatter(
                x=[str_start, str_end],
                y=[dr.eq, dr.eq],
                mode='lines',
                line=dict(color=eq_color, width=2, dash='dashdot'),
                hoverinfo='text',
                text=hover_text,
                name="Dealing Range & EQ (50%)",
                legendgroup="equilibrium",
                showlegend=not shown_eq_legend,
                hoverlabel=dict(bgcolor="black", bordercolor="cyan", font_size=12)
            ))
            shown_eq_legend = True

    # Overlay Multi-Swing Fibonacci Retracements & OTE (62% - 70.5% - 79%)
    if fibs:
        shown_fib_legend = False
        for fib in fibs:
            if fib.extreme_time < chart_start_time or fib.start_time > chart_end_time:
                continue

            draw_start = max(fib.start_time, chart_start_time)
            # Extend Fib retracement across the pullback until broken or end of window
            if fib.break_time is not None and fib.break_time <= chart_end_time:
                draw_end = fib.break_time
            else:
                draw_end = chart_end_time

            idx_start = max(0, min(df.index.searchsorted(draw_start), len(df) - 1))
            idx_end = max(0, min(df.index.searchsorted(draw_end), len(df) - 1))

            str_start = df.index[idx_start].strftime('%Y-%m-%d %H:%M')
            str_end = df.index[idx_end].strftime('%Y-%m-%d %H:%M')

            lvls = fib.levels
            ote_low, ote_high = fib.ote_bounds

            hover_text = (
                f"<b>{fib.direction} Fibonacci Retracement ({fib.swing_count} Swings)</b><br>"
                f"0.0 (Extreme): {lvls[0.0]:.2f}<br>"
                f"0.382: {lvls[0.382]:.2f}<br>"
                f"0.500: {lvls[0.5]:.2f}<br>"
                f"0.620 (OTE): {lvls[0.62]:.2f}<br>"
                f"0.705 (OTE Sweet Spot): {lvls[0.705]:.2f}<br>"
                f"0.790 (OTE): {lvls[0.79]:.2f}<br>"
                f"1.0 (Anchor): {lvls[1.0]:.2f}<br>"
                f"Status: {fib.status}<br>"
                f"Trend Start: {fib.start_time.strftime('%Y-%m-%d %H:%M')}<br>"
                f"Extreme Time: {fib.extreme_time.strftime('%Y-%m-%d %H:%M')}"
            )

            # 1. Full Fib Range Outer Frame (0.0 to 1.0)
            fig.add_trace(go.Scatter(
                x=[str_start, str_end, str_end, str_start, str_start],
                y=[fib.low, fib.low, fib.high, fib.high, fib.low],
                mode='lines',
                line=dict(color="rgba(255, 215, 0, 0.5)", width=1, dash='dot'),
                hoverinfo='skip',
                legendgroup="fibonacci",
                showlegend=False,
            ))

            # 2. OTE Zone (0.62 - 0.79) Shaded Box
            fig.add_trace(go.Scatter(
                x=[str_start, str_end, str_end, str_start, str_start],
                y=[ote_low, ote_low, ote_high, ote_high, ote_low],
                fill='toself',
                fillcolor="rgba(255, 80, 80, 0.16)",
                line=dict(color="rgba(255, 90, 90, 0.85)", width=1.5, dash='solid'),
                mode='lines',
                hoverinfo='text',
                text=hover_text,
                legendgroup="fibonacci",
                showlegend=False,
                hoverlabel=dict(bgcolor="black", bordercolor="gold", font_size=12)
            ))

            # 3. 0.705 OTE Sweet Spot Line (Green)
            fig.add_trace(go.Scatter(
                x=[str_start, str_end],
                y=[lvls[0.705], lvls[0.705]],
                mode='lines',
                line=dict(color="rgba(50, 220, 100, 0.95)", width=2, dash='dash'),
                hoverinfo='text',
                text=hover_text,
                name="Fibonacci & OTE (62%-79%)",
                legendgroup="fibonacci",
                showlegend=not shown_fib_legend,
                hoverlabel=dict(bgcolor="black", bordercolor="gold", font_size=12)
            ))
            shown_fib_legend = True

    fig.update_layout(
        title=title,
        yaxis_title="Price",
        xaxis_title="Time (NY)",
        xaxis_type='category',
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=800
    )

    filename = "es_chart.html" if "ES" in title else "nq_chart.html"
    filepath = os.path.abspath(filename)

    fig.write_html(filepath, auto_open=False)

    webbrowser.open(f"file:///{filepath}")
    time.sleep(1)