"""
Data loading and basic preprocessing utilities for IPL boundary prediction.
"""

from typing import Tuple
import pandas as pd
import numpy as np
from src.bowler_mapping import map_bowler_type


def get_phase(over: int) -> str:
    """Classifies match over into game phase."""
    if over <= 6:
        return "Powerplay"
    elif over <= 15:
        return "Middle"
    else:
        return "Death"


def load_raw_ipl_data(filepath: str = "IPL.csv") -> pd.DataFrame:
    """Loads raw IPL ball-by-ball dataset."""
    return pd.read_csv(filepath, low_memory=False)


def prepare_kohli_dataset(
    filepath: str = "IPL.csv",
    batter: str = "V Kohli",
    max_year: int = 2023
) -> pd.DataFrame:
    """
    Filters raw IPL data for specified batter up to max_year,
    adds over_phase, cleans 7-run edge cases, and sets boundary target.
    """
    df = pd.read_csv(filepath, low_memory=False)
    df = df[df["batter"] == batter].copy()

    df["date"] = pd.to_datetime(df["date"])
    if max_year is not None:
        df = df[df["date"].dt.year <= max_year].copy()

    df["over_phase"] = df["over"].apply(get_phase)

    # Drop rare 7-run balls (overthrow edge-cases) and create binary boundary target
    df = df.drop(df[df["runs_total"] == 7].index)
    df["runs_total"] = np.where(df["runs_total"] > 3, 1, 0)

    return df


def split_innings(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits dataset into Innings 1 and Innings 2 copies.
    """
    dfi1 = df[df["innings"] == 1].copy()
    dfi2 = df[df["innings"] == 2].copy()
    return dfi1, dfi2
