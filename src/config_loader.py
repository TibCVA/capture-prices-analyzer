"""Configuration loaders for Capture Prices Analyzer v3.0."""

from __future__ import annotations

from pathlib import Path
import datetime as _dt
import yaml

COUNTRY_CODE_PERIODS_KEY = "__country_code_periods__"


def _read_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration introuvable: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML invalide (objet racine attendu): {p}")
    return data


def load_countries_config(path: str = "config/countries.yaml") -> dict:
    """Charge et valide `countries.yaml`.

    Retourne un dict indexe par code pays (`FR`, `DE`, ...).
    Un bloc optionnel de periodes ENTSO-E est stocke sous `COUNTRY_CODE_PERIODS_KEY`.
    """

    data = _read_yaml(path)
    countries = data.get("countries")
    if not isinstance(countries, dict) or not countries:
        raise ValueError("countries.yaml invalide: bloc `countries` requis")

    required_top = {"entsoe_code", "timezone", "must_run", "flex", "thermal"}
    validated: dict[str, dict] = {}

    for country_key, cfg in countries.items():
        if not isinstance(cfg, dict):
            raise ValueError(f"countries.yaml invalide: config non-dict pour {country_key}")
        missing = sorted(required_top - set(cfg.keys()))
        if missing:
            raise ValueError(
                f"countries.yaml invalide: {country_key} sans champs requis {missing}"
            )

        must_run = cfg.get("must_run", {})
        if must_run.get("mode") not in {"observed", "floor"}:
            raise ValueError(
                f"countries.yaml invalide: {country_key}.must_run.mode doit etre observed|floor"
            )

        flex = cfg.get("flex", {})
        if flex.get("model_mode") not in {"observed", "capacity"}:
            raise ValueError(
                f"countries.yaml invalide: {country_key}.flex.model_mode doit etre observed|capacity"
            )

        validated[country_key] = cfg

    periods = data.get("country_code_periods")
    if periods is not None:
        if not isinstance(periods, dict):
            raise ValueError("countries.yaml invalide: `country_code_periods` doit etre un dict")
        validated[COUNTRY_CODE_PERIODS_KEY] = periods

    return validated


def load_thresholds(path: str = "config/thresholds.yaml") -> dict:
    """Charge et valide `thresholds.yaml`."""

    data = _read_yaml(path)
    required = {"phase_thresholds", "model_params", "coherence_params"}
    missing = sorted(required - set(data.keys()))
    if missing:
        raise ValueError(f"thresholds.yaml invalide: champs manquants {missing}")
    return data


def load_scenarios(path: str = "config/scenarios.yaml") -> dict:
    """Charge et retourne le bloc `scenarios`."""

    data = _read_yaml(path)
    scenarios = data.get("scenarios")
    if not isinstance(scenarios, dict):
        raise ValueError("scenarios.yaml invalide: bloc `scenarios` requis")
    return scenarios


def resolve_entsoe_code(country_key: str, year: int, countries_cfg: dict) -> str:
    """Resout le code ENTSO-E selon les periodes configurables."""

    if country_key not in countries_cfg:
        raise KeyError(f"Pays inconnu: {country_key}")

    periods = countries_cfg.get(COUNTRY_CODE_PERIODS_KEY) or countries_cfg.get("country_code_periods")
    date_ref = _dt.date(year, 1, 1)

    if isinstance(periods, dict) and country_key in periods:
        for item in periods[country_key]:
            try:
                start = _dt.date.fromisoformat(item["start"])
                end = _dt.date.fromisoformat(item["end"])
                if start <= date_ref <= end:
                    return item["code"]
            except Exception as exc:  # pragma: no cover - defensive parsing
                raise ValueError(
                    f"Spec ambigue : periode ENTSO-E invalide pour {country_key} ({item})"
                ) from exc

    return countries_cfg[country_key]["entsoe_code"]
