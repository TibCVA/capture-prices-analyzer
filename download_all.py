"""
Script de telechargement batch de toutes les donnees ENTSO-E.
Lance en standalone, pas via Streamlit.
"""
import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('download_all.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("download_all")

from src.data_fetcher import fetch_country_year, load_commodity_prices
from src.data_loader import (load_raw, load_processed, save_processed,
                              load_country_config, load_metrics, save_metrics,
                              load_diagnostics, save_diagnostics, load_thresholds)
from src.nrl_engine import compute_nrl
from src.metrics import compute_annual_metrics
from src.phase_diagnostics import diagnose_phase


def main():
    api_key = os.environ.get('ENTSOE_API_KEY')
    if not api_key:
        logger.error("ENTSOE_API_KEY manquante dans .env")
        sys.exit(1)

    countries = ['FR', 'DE', 'ES', 'PL', 'DK']
    years = list(range(2015, 2025))
    mode = 'observed'

    # Identifier les taches manquantes
    tasks = []
    for ck in countries:
        for y in years:
            processed = load_processed(ck, y, mode)
            if processed is None:
                tasks.append((ck, y))
            else:
                logger.info(f"SKIP {ck}/{y} -- deja en cache")

    if not tasks:
        logger.info("Tout est deja en cache. Rien a telecharger.")
        return

    logger.info(f"=== {len(tasks)} pays/annees a telecharger ===")
    for ck, y in tasks:
        logger.info(f"  - {ck}/{y}")

    # Precharger les configs et commodities
    configs = {}
    for ck in countries:
        try:
            configs[ck] = load_country_config(ck)
        except ValueError as e:
            logger.error(f"Config manquante pour {ck}: {e}")

    commodity_prices = load_commodity_prices()
    thresholds = load_thresholds()

    # Stocker toutes les metriques pour le diagnostic inter-annuel
    all_metrics = {}
    # Precharger les metriques existantes
    for ck in countries:
        for y in years:
            m = load_metrics(ck, y, mode)
            if m:
                all_metrics[(ck, y)] = m

    done = 0
    errors = []
    start_time = time.time()

    def _fetch_one(ck, y):
        """Fetch + process un pays/annee."""
        if ck not in configs:
            raise ValueError(f"Config manquante pour {ck}")

        t0 = time.time()
        raw = load_raw(ck, y)
        if raw is None:
            raw = fetch_country_year(ck, y, api_key)

        processed = compute_nrl(raw, configs[ck], ck, y, commodity_prices, mode)
        save_processed(processed, ck, y, mode)
        metrics = compute_annual_metrics(processed, y, ck)
        save_metrics(metrics, ck, y, mode)
        elapsed = time.time() - t0
        return ck, y, metrics, elapsed

    # Telecharger SEQUENTIELLEMENT (1 a la fois pour eviter les 504 ENTSO-E)
    for ck, y in tasks:
        try:
            ck, y, metrics, elapsed = _fetch_one(ck, y)
            all_metrics[(ck, y)] = metrics
            done += 1
            total_elapsed = time.time() - start_time
            remaining = (total_elapsed / done) * (len(tasks) - done) if done > 0 else 0
            logger.info(f"OK {ck}/{y} ({elapsed:.0f}s) -- "
                       f"{done}/{len(tasks)} -- "
                       f"total: {total_elapsed:.0f}s -- "
                       f"ETA: {remaining:.0f}s")
        except Exception as e:
            errors.append(f"{ck}/{y}: {e}")
            done += 1
            logger.error(f"FAIL {ck}/{y}: {e}")
        # Pause entre chaque appel pour ne pas surcharger l'API
        if done < len(tasks):
            time.sleep(5)

    # Calculer les diagnostics (necessite l'ordre chronologique)
    logger.info("=== Calcul des diagnostics ===")
    for ck in countries:
        for y in sorted(years):
            if (ck, y) not in all_metrics:
                continue
            prev_m = all_metrics.get((ck, y - 1))
            diag = diagnose_phase(all_metrics[(ck, y)], thresholds, prev_m)
            save_diagnostics(diag, ck, y, mode)
            logger.info(f"Diag {ck}/{y}: Stage {diag['phase_number']} "
                       f"(confiance {diag.get('confidence', 0):.0%})")

    total_time = time.time() - start_time
    logger.info(f"=== TERMINE === {done}/{len(tasks)} en {total_time:.0f}s")
    if errors:
        logger.warning(f"{len(errors)} erreurs:")
        for e in errors:
            logger.warning(f"  {e}")
    else:
        logger.info("Aucune erreur.")


if __name__ == "__main__":
    main()
