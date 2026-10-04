import pandas as pd

def extract_session_liquidity(df_1m: pd.DataFrame) -> list:
    """
    Extracts the absolute ONE high and ONE low per trading session
    (Asia, London, New York) using exact 1-minute precision.
    """
    session_pools = []
    grouped = df_1m.groupby(pd.Grouper(freq='24h', offset='17h'))

    for name, group in grouped:
        if group.empty:
            continue

        # 1. ASIA (18:00 to 02:59)
        asia_data = group.between_time('18:00', '02:59')
        if not asia_data.empty:
            session_pools.append({'session': 'Asia', 'type': 'BSL', 'price': asia_data['high'].max(), 'time': asia_data['high'].idxmax()})
            session_pools.append({'session': 'Asia', 'type': 'SSL', 'price': asia_data['low'].min(), 'time': asia_data['low'].idxmin()})

        # 2. LONDON (03:00 to 09:29)
        london_data = group.between_time('03:00', '09:29')
        if not london_data.empty:
            session_pools.append({'session': 'London', 'type': 'BSL', 'price': london_data['high'].max(), 'time': london_data['high'].idxmax()})
            session_pools.append({'session': 'London', 'type': 'SSL', 'price': london_data['low'].min(), 'time': london_data['low'].idxmin()})

        # 3. NEW YORK (09:30 to 16:59)
        ny_data = group.between_time('09:30', '16:59')
        if not ny_data.empty:
            session_pools.append({'session': 'New York', 'type': 'BSL', 'price': ny_data['high'].max(), 'time': ny_data['high'].idxmax()})
            session_pools.append({'session': 'New York', 'type': 'SSL', 'price': ny_data['low'].min(), 'time': ny_data['low'].idxmin()})

    return session_pools