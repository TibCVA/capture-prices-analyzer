"""Time and timezone utilities for Capture Prices Analyzer v3.0."""

from __future__ import annotations

import pandas as pd
from src.constants import TZ_UTC


def to_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convertit un index datetime en UTC tz-aware."""

    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("to_utc_index attend un pd.DatetimeIndex")
    if idx.tz is None:
        return idx.tz_localize(TZ_UTC)
    return idx.tz_convert(TZ_UTC)


def as_local(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Retourne une copie vue locale (index converti sans modifier les valeurs)."""

    out = df.copy(deep=False)
    if not isinstance(out.index, pd.DatetimeIndex):
        raise TypeError("as_local requiert un index DatetimeIndex")
    out.index = to_utc_index(out.index).tz_convert(tz)
    return out


def is_weekday_local(ts: pd.Timestamp) -> bool:
    """Renvoie True pour lundi-vendredi en timezone locale du timestamp."""

    if not isinstance(ts, pd.Timestamp):
        raise TypeError("is_weekday_local attend un pd.Timestamp")
    if ts.tz is None:
        raise ValueError("is_weekday_local requiert un timestamp tz-aware")
    return ts.weekday() < 5


def peak_mask(df_index_utc: pd.DatetimeIndex, tz: str) -> pd.Series:
    """Masque peak local: heures 08..19 inclus, jours ouvres uniquement."""

    idx_utc = to_utc_index(df_index_utc)
    idx_local = idx_utc.tz_convert(tz)
    mask = (idx_local.weekday < 5) & (idx_local.hour >= 8) & (idx_local.hour <= 19)
    return pd.Series(mask, index=idx_utc)
