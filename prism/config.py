"""Configuration loading: config.yaml for settings, .env for API keys."""
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


def load_config(path="config.yaml"):
    """Load config.yaml and return (tickers, cfg) in the shape the engine expects."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    weights = raw.get("weights", {})
    cfg = {
        "weight_growth": weights.get("growth", 0.35),
        "weight_value": weights.get("value", 0.25),
        "weight_momentum": weights.get("momentum", 0.20),
        "weight_resilience": weights.get("resilience", 0.20),
        "blend_2026": raw.get("blend_2026", 0.75),
        "blend_2027": raw.get("blend_2027", 0.25),
        "growth_cap": raw.get("growth_cap", 2.00),
        "sector_relative": raw.get("sector_relative", False),
        "edge_size": raw.get("edge_size", 10),
        "fragile_threshold": raw.get("fragile_threshold", 3.0),
        "growth_overrides": {
            k: tuple(v) for k, v in (raw.get("growth_overrides") or {}).items()
        },
        "force_include": raw.get("force_include") or [],
        "force_exclude": raw.get("force_exclude") or [],
        "output_dir": raw.get("output_dir", "outputs"),
    }
    return raw["tickers"], cfg


def load_api_keys(env_path=None):
    """Load API keys from .env (or the environment). Returns (fmp_key, finnhub_key)."""
    load_dotenv(env_path or Path(".env"))
    return os.getenv("FMP_API_KEY", ""), os.getenv("FINNHUB_API_KEY", "")
