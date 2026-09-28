"""
Feature engineering module for IPL ball-by-ball momentum, form, and match context.
"""

from typing import List
import numpy as np
import pandas as pd
from src.bowler_mapping import map_bowler_type


MODEL_FEATURES: List[str] = [
    "over",
    "ball",
    "bat_pos",
    "bowler",
    "current_sr",
    "runs_last_5",
    "runs_last_10",
    "prev_ball_runs",
    "boundaries_last_5",
    "dotballs_last_5",
    "balls_since_boundary",
    "over_phase",
    "runs_total",
]

NUMERIC_FEATURES: List[str] = [
    "over",
    "ball",
    "current_sr",
    "runs_last_5",
    "runs_last_10",
    "prev_ball_runs",
    "boundaries_last_5",
    "dotballs_last_5",
    "balls_since_boundary",
]

CATEGORICAL_FEATURES: List[str] = [
    "bat_pos",
    "bowler",
    "over_phase",
]


def create_dataset(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Computes situational, momentum, and historical features per match:
    - current_sr: Progressive strike rate before current delivery
    - runs_last_5, runs_last_10: Rolling runs in previous 5 and 10 balls
    - prev_ball_runs: Runs scored on the immediate previous ball
    - boundaries_last_5: Boundaries struck in previous 5 balls
    - dotballs_last_5: Dot balls faced in previous 5 balls
    - balls_since_boundary: Deliveries elapsed since batter's last boundary
    """
    df = df_in.sort_values(["match_id", "over", "ball"]).reset_index(drop=True).copy()

    # Current Strike Rate (before current ball)
    df["current_sr"] = (
        df.groupby("match_id")["runs_batter"]
        .transform(lambda x: (x.shift(1).cumsum() / np.maximum(np.arange(len(x)), 1)) * 100)
    )

    # Runs in previous 5 balls
    df["runs_last_5"] = (
        df.groupby("match_id")["runs_batter"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    )

    # Runs in previous 10 balls
    df["runs_last_10"] = (
        df.groupby("match_id")["runs_batter"]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).sum())
    )

    # Previous ball runs
    df["prev_ball_runs"] = (
        df.groupby("match_id")["runs_batter"]
        .shift(1)
        .fillna(0)
    )

    # Boundary flag
    df["boundary"] = df["runs_batter"].isin([4, 6]).astype(int)

    # Boundaries in previous 5 balls
    df["boundaries_last_5"] = (
        df.groupby("match_id")["boundary"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    )

    # Dot ball flag
    df["dot_ball"] = (df["runs_batter"] == 0).astype(int)

    # Dot balls in previous 5 balls
    df["dotballs_last_5"] = (
        df.groupby("match_id")["dot_ball"]
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    )

    # Balls since last boundary
    def balls_since_boundary(boundary_series):
        ans = []
        last = -1
        for i, b in enumerate(boundary_series):
            if last == -1:
                ans.append(-1)
            else:
                ans.append(i - last)
            if b == 1:
                last = i
        return ans

    df["balls_since_boundary"] = (
        df.groupby("match_id")["boundary"]
        .transform(balls_since_boundary)
    )

    # Target
    df["target"] = df["boundary"]
    return df


def prepare_engineered_dataset(df_innings: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature engineering, extracts target features, maps bowler to pace/spin,
    and fills initial NaN values with 0.
    """
    df = create_dataset(df_innings)
    df = df[MODEL_FEATURES].copy()
    df = df.fillna(0)
    df["bowler"] = df["bowler"].apply(map_bowler_type)
    return df
