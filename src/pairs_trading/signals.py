import pandas as pd


def rolling_zscore(spread, lookback):
    """Compute z-score using rolling statistics known before the current bar."""
    shifted = spread.shift(1)
    mean = shifted.rolling(lookback).mean()
    std = shifted.rolling(lookback).std()
    zscore = (spread - mean) / std
    zscore.name = "zscore"
    return zscore


def generate_positions(
    zscore,
    entry_z,
    exit_z,
    stop_z,
    max_holding_days=30,
):
    """Generate desired close-of-day spread positions.

    1 means long spread, -1 means short spread, and 0 means flat.
    """
    positions = []
    position = 0
    holding_days = 0

    for z in zscore:
        if pd.isna(z):
            positions.append(position)
            if position != 0:
                holding_days += 1
            continue

        if position == 0:
            if z <= -entry_z:
                position = 1
                holding_days = 0
            elif z >= entry_z:
                position = -1
                holding_days = 0
        else:
            holding_days += 1
            should_exit = abs(z) <= exit_z or abs(z) >= stop_z
            if max_holding_days is not None:
                should_exit = should_exit or holding_days >= max_holding_days
            if should_exit:
                position = 0
                holding_days = 0

        positions.append(position)

    return pd.Series(positions, index=zscore.index, name="position", dtype="int64")
