#!/usr/bin/env python3
"""
pipeline/run_coadd.py
---------------------
Main entry point for the DASCH median coadd prototype pipeline.

Usage:
    python pipeline/run_coadd.py --config config/m31.json [--input-dir /path/to/fits]

Environment variables:
    RUN_MODE     : 'test' (default, limits to 6 plates / 2 deg^2) or 'full'
    INPUT_DIR    : directory containing pre-downloaded plate FITS files
                   (overrides any API download attempt)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("dasch_coadd")


# ---------------------------------------------------------------------------
# Helper: read config
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = json.load(f)
    required = [
        "name", "center_ra_deg", "center_dec_deg",
        "area_sqdeg", "projection", "pixel_scale_arcsec", "coadd_method",
    ]
    for key in required:
        if key not in cfg:
            raise ValueError(f"Config missing required key: {key}")
    return cfg


# ---------------------------------------------------------------------------
# Helper: build WCS for output grid
# ---------------------------------------------------------------------------
def make_output_wcs(cfg: dict):
    """Return an astropy WCS for the output TAN grid."""
    import math

    ra0 = cfg["center_ra_deg"]
    dec0 = cfg["center_dec_deg"]
    scale_deg = cfg["pixel_scale_arcsec"] / 3600.0

    # For a square tile side = sqrt(area)
    run_mode = os.environ.get("RUN_MODE", "test")
    area = cfg["area_sqdeg"]
    if run_mode == "test":
        area = min(area, 2.0)
        log.info("RUN_MODE=test: limiting tile area to %.1f deg^2", area)

    side_deg = math.sqrt(area)
    n_pix = int(side_deg / scale_deg)
    if n_pix % 2 != 0:
        n_pix += 1

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [n_pix / 2 + 0.5, n_pix / 2 + 0.5]
    wcs.wcs.cdelt = [-scale_deg, scale_deg]
    wcs.wcs.crval = [ra0, dec0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs, n_pix, n_pix


# ---------------------------------------------------------------------------
# Helper: build simple synthetic plate for test mode
# ---------------------------------------------------------------------------
def _make_synthetic_plate(cfg: dict, plate_id: int, seed: int):
    """Return (data, wcs, mask) for a synthetic test plate centered near M31."""
    rng = np.random.default_rng(seed)
    nx, ny = 512, 512
    scale_deg = cfg["pixel_scale_arcsec"] / 3600.0 * 2  # coarser plate scale

    # Small random offset so plates don't perfectly overlap
    ra0 = cfg["center_ra_deg"] + rng.uniform(-0.3, 0.3)
    dec0 = cfg["center_dec_deg"] + rng.uniform(-0.3, 0.3)

    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [nx / 2 + 0.5, ny / 2 + 0.5]
    wcs.wcs.cdelt = [-scale_deg, scale_deg]
    wcs.wcs.crval = [ra0, dec0]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # Fake sky background + noise + a few "stars"
    data = rng.normal(1000.0, 30.0, (ny, nx)).astype(np.float32)
    n_stars = rng.integers(20, 60)
    for _ in range(n_stars):
        sx = rng.integers(5, nx - 5)
        sy = rng.integers(5, ny - 5)
        flux = rng.uniform(500, 5000)
        yy, xx = np.mgrid[0:ny, 0:nx]
        data += (flux * np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * 2.5 ** 2))).astype(np.float32)

    # Simple mask: flag a random rectangle as bad
    mask = np.zeros((ny, nx), dtype=np.uint8)
    bx, by = rng.integers(0, nx // 2), rng.integers(0, ny // 2)
    mask[by : by + 30, bx : bx + 30] = 1

    return data, wcs, mask


# ---------------------------------------------------------------------------
# Plate ingestion
# ---------------------------------------------------------------------------
def load_plates(cfg: dict, input_dir: str | None):
    """
    Return a list of (plate_id, data_array, wcs, mask_array, header) tuples.

    In test mode: generate synthetic plates.
    Otherwise: load from input_dir (pre-downloaded FITS) or attempt DASCH API.
    """
    run_mode = os.environ.get("RUN_MODE", "test")
    plates = []

    if run_mode == "test":
        n_plates = int(os.environ.get("TEST_N_PLATES", "6"))
        log.info("RUN_MODE=test: generating %d synthetic plates", n_plates)
        for i in range(n_plates):
            data, wcs, mask = _make_synthetic_plate(cfg, i, seed=42 + i)
            hdr = fits.Header()
            hdr["PLATEID"] = f"SYN{i:04d}"
            hdr["COMMENT"] = "Synthetic test plate"
            plates.append((f"SYN{i:04d}", data, wcs, mask, hdr))
        return plates

    # Full mode: try input_dir first, then API
    if input_dir and Path(input_dir).is_dir():
        fits_files = sorted(Path(input_dir).glob("*.fits")) + sorted(Path(input_dir).glob("*.fit"))
        if not fits_files:
            log.warning("No FITS files found in %s", input_dir)
        for fpath in fits_files:
            try:
                pid, data, wcs, mask, hdr = _load_fits_plate(fpath)
                plates.append((pid, data, wcs, mask, hdr))
                log.info("Loaded plate %s from %s", pid, fpath)
            except Exception as exc:
                log.warning("Failed to load %s: %s", fpath, exc)
        if plates:
            return plates

    # Fall back to DASCH API download
    log.info("Attempting to download plates from DASCH API …")
    try:
        from scripts.dasch_download import download_plates_for_tile
        plates = download_plates_for_tile(cfg)
    except Exception as exc:
        log.error(
            "DASCH API download failed: %s\n"
            "Please pre-download plate FITS files and pass --input-dir, "
            "or set RUN_MODE=test for synthetic data.",
            exc,
        )
        sys.exit(1)

    return plates


def _load_fits_plate(fpath: Path):
    """Load a single plate FITS; return (plate_id, data, wcs, mask, header)."""
    with fits.open(fpath) as hdul:
        # Primary extension is the image
        hdr = hdul[0].header
        data = hdul[0].data.astype(np.float32)
        wcs = WCS(hdr, naxis=2)
        plate_id = hdr.get("PLATEID", hdr.get("PLATE_ID", fpath.stem))

        # Look for a mask extension (common names: MASK, FLAGS, DQ)
        mask = None
        for ext_name in ("MASK", "FLAGS", "DQ", "WEIGHT"):
            if ext_name in hdul:
                mask = hdul[ext_name].data.astype(np.uint8)
                break
        if mask is None and len(hdul) > 1:
            try:
                mask = hdul[1].data.astype(np.uint8)
            except Exception:
                pass

        if mask is None:
            mask = _auto_mask(data)

    return plate_id, data, wcs, mask, hdr


def _auto_mask(data: np.ndarray) -> np.ndarray:
    """Generate a simple mask by flagging extreme pixel values."""
    from scipy.ndimage import binary_dilation

    finite = np.isfinite(data)
    lo, hi = np.nanpercentile(data[finite], [0.1, 99.9])
    bad = ~finite | (data < lo) | (data > hi)
    # Expand bad regions slightly
    bad = binary_dilation(bad, iterations=2)
    return bad.astype(np.uint8)


# ---------------------------------------------------------------------------
# Weight map computation
# ---------------------------------------------------------------------------
def compute_weight(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Simple inverse-variance weight map.
    Background RMS is estimated from unmasked pixels using a sigma-clipped median.
    """
    good = mask == 0
    if good.sum() < 100:
        return np.zeros_like(data)
    vals = data[good]
    med = np.nanmedian(vals)
    rms = np.nanmedian(np.abs(vals - med)) * 1.4826  # MAD estimator
    if rms <= 0:
        return np.zeros_like(data)
    weight = np.where(good, 1.0 / rms ** 2, 0.0).astype(np.float32)
    return weight


# ---------------------------------------------------------------------------
# Reprojection
# ---------------------------------------------------------------------------
def reproject_plate(data, wcs_in, mask, weight, output_wcs, shape_out, plate_id, tmp_dir: Path):
    """
    Reproject data, mask, and weight onto output_wcs grid.
    Returns (repr_data, repr_mask, repr_weight).
    Saves intermediate files to tmp_dir.
    """
    from reproject import reproject_interp

    # Mask invalid pixels as NaN before reprojection
    data_nan = data.copy().astype(np.float64)
    data_nan[mask != 0] = np.nan

    try:
        # order=1: bilinear interpolation; order=0: nearest-neighbour (for masks)
        repr_data, footprint = reproject_interp(
            (data_nan, wcs_in), output_wcs, shape_out=shape_out, order=1
        )
        repr_mask_f, _ = reproject_interp(
            (mask.astype(np.float64), wcs_in), output_wcs, shape_out=shape_out, order=0
        )
        repr_weight_f, _ = reproject_interp(
            (weight.astype(np.float64), wcs_in), output_wcs, shape_out=shape_out, order=1
        )
    except Exception as exc:
        log.warning("Reprojection failed for plate %s: %s", plate_id, exc)
        empty = np.full(shape_out, np.nan)
        return empty, np.ones(shape_out, dtype=np.uint8), np.zeros(shape_out)

    repr_mask = (repr_mask_f > 0.5).astype(np.uint8)
    repr_mask[footprint < 0.5] = 1  # outside footprint = masked
    repr_data[repr_mask != 0] = np.nan

    repr_weight = repr_weight_f.astype(np.float32)
    repr_weight[repr_mask != 0] = 0.0

    # Save to tmp
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out_hdr = output_wcs.to_header()
    fits.writeto(tmp_dir / f"{plate_id}_repr.fits", repr_data.astype(np.float32), out_hdr, overwrite=True)
    fits.writeto(tmp_dir / f"{plate_id}_mask.fits", repr_mask, out_hdr, overwrite=True)

    return repr_data, repr_mask, repr_weight


# ---------------------------------------------------------------------------
# Coaddition
# ---------------------------------------------------------------------------
def median_coadd(repr_stack, mask_stack, weight_stack):
    """
    Produce median coadd and auxiliary maps.
    Returns (coadd, weight_map, mask_map, contributors_map).
    """
    # Stack shape: (N, ny, nx)
    arr = np.array(repr_stack, dtype=np.float64)   # (N, ny, nx)
    wts = np.array(weight_stack, dtype=np.float64)  # (N, ny, nx)
    msk = np.array(mask_stack, dtype=np.uint8)       # (N, ny, nx)

    # Mask out bad pixels in arr
    arr[msk != 0] = np.nan

    # Median ignoring NaNs
    coadd = np.nanmedian(arr, axis=0).astype(np.float32)

    # Number of valid contributors per pixel
    contributors = np.sum(np.isfinite(arr), axis=0).astype(np.int16)

    # Sum of weights (only from good pixels)
    wts[msk != 0] = 0.0
    weight_map = np.sum(wts, axis=0).astype(np.float32)

    # Logical OR of masks: pixel is masked if ALL inputs are masked
    mask_map = (contributors == 0).astype(np.uint8)

    return coadd, weight_map, mask_map, contributors


# ---------------------------------------------------------------------------
# Source detection
# ---------------------------------------------------------------------------
def detect_sources(coadd: np.ndarray, output_wcs: WCS, threshold_sigma: float = 3.0):
    """Run SEP source detection on coadd. Returns an astropy Table."""
    import sep
    from astropy.table import Table

    data = coadd.copy().astype(np.float64)
    finite = np.isfinite(data)
    if finite.sum() < 100:
        log.warning("Too few finite pixels for source detection")
        return Table(names=["x", "y", "ra", "dec", "flux_auto", "fluxerr_auto"])

    # Fill NaNs for background estimation
    data[~finite] = np.nanmedian(data[finite])

    bkg = sep.Background(data)
    data_sub = data - bkg

    try:
        objects = sep.extract(data_sub, threshold_sigma, err=bkg.globalrms)
    except Exception as exc:
        log.warning("SEP extraction failed: %s", exc)
        return Table(names=["x", "y", "ra", "dec", "flux_auto", "fluxerr_auto"])

    if len(objects) == 0:
        log.info("No sources detected above threshold")
        return Table(names=["x", "y", "ra", "dec", "flux_auto", "fluxerr_auto"])

    # Aperture photometry
    flux, fluxerr, flag = sep.sum_circle(
        data_sub, objects["x"], objects["y"], r=3.0, err=bkg.globalrms
    )

    # Sky coordinates
    sky = output_wcs.pixel_to_world(objects["x"], objects["y"])
    ra = sky.ra.deg
    dec = sky.dec.deg

    tbl = Table(
        {
            "x": objects["x"],
            "y": objects["y"],
            "ra": ra,
            "dec": dec,
            "flux_auto": flux,
            "fluxerr_auto": fluxerr,
        }
    )
    log.info("Detected %d sources", len(tbl))
    return tbl


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------
def write_outputs(
    coadd, weight_map, mask_map, contributors, source_catalog, output_wcs, cfg, plate_ids, out_dir: Path
):
    """Write all output FITS files and provenance log."""
    out_dir.mkdir(parents=True, exist_ok=True)
    hdr = output_wcs.to_header()
    hdr["TILE"] = cfg["name"]
    hdr["RA_CTR"] = (cfg["center_ra_deg"], "Tile center RA (deg)")
    hdr["DEC_CTR"] = (cfg["center_dec_deg"], "Tile center Dec (deg)")
    hdr["COADD_M"] = (cfg["coadd_method"], "Coaddition method")
    hdr["PIXSCALE"] = (cfg["pixel_scale_arcsec"], "Pixel scale arcsec")
    hdr["N_PLATES"] = (len(plate_ids), "Number of contributing plates")

    # Embed plate IDs (truncated to 80-char FITS header values)
    for i, pid in enumerate(plate_ids[:64]):  # cap at 64 cards
        hdr[f"PLATEID{i:02d}"] = pid

    fits.writeto(out_dir / "coadd.fits", coadd, hdr, overwrite=True)
    fits.writeto(out_dir / "weight_map.fits", weight_map, hdr, overwrite=True)
    fits.writeto(out_dir / "mask_map.fits", mask_map, hdr, overwrite=True)
    fits.writeto(out_dir / "contributors_map.fits", contributors, hdr, overwrite=True)

    if source_catalog is not None and len(source_catalog) > 0:
        source_catalog.write(out_dir / "source_catalog.fits", overwrite=True)

    # Provenance log
    log_path = out_dir / "provenance.log"
    with open(log_path, "w") as f:
        f.write(f"DASCH Coadd Pipeline — Provenance Log\n")
        f.write(f"Tile: {cfg['name']}\n")
        f.write(f"RA/Dec center: {cfg['center_ra_deg']}, {cfg['center_dec_deg']}\n")
        f.write(f"Area: {cfg['area_sqdeg']} deg^2\n")
        f.write(f"Coadd method: {cfg['coadd_method']}\n")
        f.write(f"Contributing plates ({len(plate_ids)}):\n")
        for pid in plate_ids:
            f.write(f"  {pid}\n")

    log.info("Output files written to %s", out_dir)


# ---------------------------------------------------------------------------
# QA plots
# ---------------------------------------------------------------------------
def write_qa_plots(coadd, contributors, plate_ids, out_dir: Path):
    """Generate QA PNG plots."""
    try:
        from scripts.qa_plots import (
            plot_coadd_thumbnail,
            plot_contributor_map,
            plot_zp_histogram,
        )
        plot_coadd_thumbnail(coadd, out_dir / "coadd_thumbnail.png")
        plot_contributor_map(contributors, out_dir / "contributors_map.png")
        plot_zp_histogram(plate_ids, out_dir / "zp_histogram.png")
    except Exception as exc:
        log.warning("QA plots failed (non-fatal): %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DASCH median coadd pipeline")
    parser.add_argument("--config", default="config/m31.json", help="Path to JSON tile config")
    parser.add_argument(
        "--input-dir",
        default=os.environ.get("INPUT_DIR"),
        help="Directory with pre-downloaded plate FITS files",
    )
    parser.add_argument("--out-dir", default="/work/output", help="Output directory for coadd products")
    parser.add_argument("--tmp-dir", default="/work/tmp", help="Temporary directory for intermediate files")
    args = parser.parse_args()

    cfg = load_config(args.config)
    log.info("Loaded config for tile: %s", cfg["name"])

    # Build output WCS / grid
    output_wcs, ny, nx = make_output_wcs(cfg)
    log.info("Output grid: %d x %d pixels (%.4f arcsec/pix)", nx, ny, cfg["pixel_scale_arcsec"])

    # Load plates
    plates = load_plates(cfg, args.input_dir)
    log.info("Total plates to process: %d", len(plates))

    tmp_dir = Path(args.tmp_dir)
    repr_stack, mask_stack, weight_stack, plate_ids = [], [], [], []

    for plate_id, data, wcs_in, mask, hdr in plates:
        log.info("Processing plate %s …", plate_id)
        weight = compute_weight(data, mask)
        r_data, r_mask, r_weight = reproject_plate(
            data, wcs_in, mask, weight, output_wcs, (ny, nx), plate_id, tmp_dir
        )
        repr_stack.append(r_data)
        mask_stack.append(r_mask)
        weight_stack.append(r_weight)
        plate_ids.append(plate_id)

    log.info("Coadding %d plates …", len(repr_stack))
    coadd, weight_map, mask_map, contributors = median_coadd(repr_stack, mask_stack, weight_stack)

    # Source detection
    catalog = detect_sources(coadd, output_wcs)

    # Write outputs
    out_dir = Path(args.out_dir)
    write_outputs(coadd, weight_map, mask_map, contributors, catalog, output_wcs, cfg, plate_ids, out_dir)

    # QA plots
    write_qa_plots(coadd, contributors, plate_ids, out_dir)

    log.info("Pipeline complete. Outputs in %s", out_dir)


if __name__ == "__main__":
    main()
