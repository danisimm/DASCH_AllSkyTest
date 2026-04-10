#!/usr/bin/env python3
"""
scripts/dasch_download.py
--------------------------
Helper script to query the DASCH DR7 API for plates overlapping a given sky region
and download calibrated plate FITS and mask files into a local directory.

Usage:
    python scripts/dasch_download.py --ra 10.684583 --dec 41.269167 \
        --radius_deg 1.3 --outdir ./data/m31_plates

    python scripts/dasch_download.py --ra 10.684583 --dec 41.269167 \
        --box_deg 2.5 --outdir ./data/m31_plates

Environment variables:
    DASCH_API_BASE  : override default API base URL
    DASCH_TIMEOUT   : HTTP request timeout in seconds (default: 30)
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

# DASCH DR7 public API base URL (update if the endpoint changes)
DASCH_API_BASE = os.environ.get(
    "DASCH_API_BASE",
    "https://dasch.cfa.harvard.edu/dr7",
)
DASCH_TIMEOUT = int(os.environ.get("DASCH_TIMEOUT", "30"))
MAX_RETRIES = 3
RETRY_BACKOFF = 5  # seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _request_with_retry(url: str, params: dict | None = None, stream: bool = False):
    """GET request with simple exponential backoff on transient failures."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=DASCH_TIMEOUT, stream=stream)
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 429:
                wait = RETRY_BACKOFF * attempt
                log.warning("Rate-limited (429). Waiting %ds before retry %d/%d …", wait, attempt, MAX_RETRIES)
                time.sleep(wait)
                last_exc = exc
            else:
                raise
        except requests.exceptions.ConnectionError as exc:
            wait = RETRY_BACKOFF * attempt
            log.warning("Connection error: %s. Waiting %ds before retry %d/%d …", exc, wait, attempt, MAX_RETRIES)
            time.sleep(wait)
            last_exc = exc
        except requests.exceptions.Timeout as exc:
            wait = RETRY_BACKOFF * attempt
            log.warning("Timeout. Waiting %ds before retry %d/%d …", wait, attempt, MAX_RETRIES)
            time.sleep(wait)
            last_exc = exc

    raise RuntimeError(f"Request to {url} failed after {MAX_RETRIES} attempts") from last_exc


# ---------------------------------------------------------------------------
# Plate query
# ---------------------------------------------------------------------------

def query_plates(ra: float, dec: float, radius_deg: float | None = None, box_deg: float | None = None) -> list[dict]:
    """
    Query the DASCH API for plates overlapping a given sky region.

    Returns a list of plate metadata dicts with at least:
        plate_id, fits_url, mask_url (when available)

    If the API is unreachable, raises an exception with a helpful message.
    """
    # Build query parameters
    params: dict = {"ra": ra, "dec": dec, "format": "json"}
    if radius_deg is not None:
        params["radius"] = radius_deg
    elif box_deg is not None:
        half = box_deg / 2.0
        params["ra_min"] = ra - half
        params["ra_max"] = ra + half
        params["dec_min"] = dec - half
        params["dec_max"] = dec + half
    else:
        raise ValueError("Either radius_deg or box_deg must be specified")

    search_url = f"{DASCH_API_BASE}/search"
    log.info("Querying DASCH API: %s  params=%s", search_url, params)

    try:
        resp = _request_with_retry(search_url, params=params)
    except Exception as exc:
        raise RuntimeError(
            f"DASCH API unreachable at {search_url}.\n"
            "If the API is down, please download plate FITS files manually from:\n"
            "  https://dasch.cfa.harvard.edu/dr7/data/\n"
            "and pass the directory via --outdir (input mode) or INPUT_DIR env var.\n"
            f"Original error: {exc}"
        ) from exc

    data = resp.json()
    plates = data.get("plates", data.get("results", data if isinstance(data, list) else []))
    log.info("API returned %d plates", len(plates))
    return plates


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_file(url: str, dest: Path) -> bool:
    """Download a single file from url to dest. Returns True on success."""
    if dest.exists():
        log.debug("Already exists, skipping: %s", dest)
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = _request_with_retry(url, stream=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        log.info("Downloaded: %s -> %s", url, dest)
        return True
    except Exception as exc:
        log.warning("Failed to download %s: %s", url, exc)
        if dest.exists():
            dest.unlink()
        return False


def download_plates_for_tile(cfg: dict) -> list:
    """
    Query DASCH and download all plates overlapping the tile defined in cfg.
    Returns a list of (plate_id, data, wcs, mask, header) tuples (same as run_coadd expects).

    Intended to be imported by run_coadd.py.
    """
    import math
    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS

    ra = cfg["center_ra_deg"]
    dec = cfg["center_dec_deg"]
    radius = math.sqrt(cfg["area_sqdeg"] / math.pi) * 1.2  # add 20% margin

    plates_meta = query_plates(ra, dec, radius_deg=radius)

    outdir = Path(os.environ.get("INPUT_DIR", "/work/data/plates"))
    outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for meta in plates_meta:
        plate_id = meta.get("plate_id", meta.get("id", "unknown"))
        fits_url = meta.get("fits_url", meta.get("url", ""))
        mask_url = meta.get("mask_url", "")

        if not fits_url:
            log.warning("No FITS URL for plate %s, skipping", plate_id)
            continue

        fits_dest = outdir / f"{plate_id}.fits"
        if not download_file(fits_url, fits_dest):
            continue

        # Optional mask
        mask_dest = outdir / f"{plate_id}_mask.fits" if mask_url else None
        if mask_url:
            download_file(mask_url, outdir / f"{plate_id}_mask.fits")

        try:
            with fits.open(fits_dest) as hdul:
                hdr = hdul[0].header
                data = hdul[0].data.astype(np.float32)
                wcs = WCS(hdr, naxis=2)

            mask = np.zeros(data.shape, dtype=np.uint8)
            if mask_dest and mask_dest.exists():
                with fits.open(mask_dest) as mhdul:
                    mask = mhdul[0].data.astype(np.uint8)

            results.append((plate_id, data, wcs, mask, hdr))
        except Exception as exc:
            log.warning("Error loading downloaded plate %s: %s", plate_id, exc)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Download DASCH DR7 plates for a sky region")
    p.add_argument("--ra", type=float, required=True, help="Center RA (deg)")
    p.add_argument("--dec", type=float, required=True, help="Center Dec (deg)")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--radius_deg", type=float, help="Search radius (deg)")
    group.add_argument("--box_deg", type=float, help="Half-size of RA/Dec box (deg)")
    p.add_argument("--outdir", required=True, help="Output directory for downloaded files")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        plates = query_plates(
            args.ra, args.dec,
            radius_deg=args.radius_deg,
            box_deg=args.box_deg,
        )
    except RuntimeError as exc:
        log.error("%s", exc)
        sys.exit(1)

    if not plates:
        log.warning("No plates found for the specified region")
        sys.exit(0)

    ok = failed = 0
    for meta in plates:
        plate_id = meta.get("plate_id", meta.get("id", "unknown"))
        fits_url = meta.get("fits_url", meta.get("url", ""))
        mask_url = meta.get("mask_url", "")

        if not fits_url:
            log.warning("No FITS URL for plate %s", plate_id)
            failed += 1
            continue

        if download_file(fits_url, outdir / f"{plate_id}.fits"):
            ok += 1
        else:
            failed += 1

        if mask_url:
            download_file(mask_url, outdir / f"{plate_id}_mask.fits")

    log.info("Download complete: %d succeeded, %d failed", ok, failed)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
