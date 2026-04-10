#!/usr/bin/env python3
"""
scripts/qa_plots.py
--------------------
Generate QA PNG plots used by the DASCH coadd workflow.

Functions:
    plot_coadd_thumbnail  -- small image thumbnail of the coadd
    plot_contributor_map  -- heatmap of per-pixel contributor counts
    plot_zp_histogram     -- zero-point histogram per plate (placeholder when ZP metadata absent)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, bbox_inches="tight")
    log.info("Saved QA plot: %s", path)


def plot_coadd_thumbnail(coadd: np.ndarray, out_path: Path):
    """Save a PNG thumbnail of the coadd image with a simple z-scale stretch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))

    finite = np.isfinite(coadd)
    if finite.sum() == 0:
        ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes, ha="center")
    else:
        vmin, vmax = np.nanpercentile(coadd[finite], [1, 99])
        im = ax.imshow(
            coadd,
            origin="lower",
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, label="Intensity (ADU)")

    ax.set_title("Coadd thumbnail")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    _save_fig(fig, out_path)
    plt.close(fig)


def plot_contributor_map(contributors: np.ndarray, out_path: Path):
    """Save a heatmap of the per-pixel contributor (good plate) count."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(contributors, origin="lower", cmap="viridis", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="N contributors")
    ax.set_title("Contributor map")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    _save_fig(fig, out_path)
    plt.close(fig)


def plot_zp_histogram(
    plate_ids: list[str],
    out_path: Path,
    zero_points: list[float] | None = None,
):
    """
    Plot a histogram of per-plate photometric zero-points.

    If zero_points is None (DASCH zero-point metadata not yet loaded),
    produce a placeholder plot indicating that ZP metadata is not available.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))

    if zero_points and len(zero_points) == len(plate_ids):
        ax.hist(zero_points, bins=max(5, len(zero_points) // 3), edgecolor="black")
        ax.set_xlabel("Zero-point (mag)")
        ax.set_ylabel("N plates")
        ax.set_title(f"Zero-point distribution ({len(plate_ids)} plates)")
    else:
        ax.text(
            0.5,
            0.5,
            "Zero-point metadata not available\n(placeholder — will be populated\nwhen DASCH API returns ZP metadata)",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            style="italic",
        )
        ax.set_title(f"Zero-point histogram ({len(plate_ids)} plates)")

    _save_fig(fig, out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI convenience — regenerate plots from existing FITS outputs
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from astropy.io import fits

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Regenerate QA plots from coadd FITS outputs")
    parser.add_argument("--out-dir", default="/work/output", help="Directory containing coadd FITS files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    coadd_path = out_dir / "coadd.fits"
    contrib_path = out_dir / "contributors_map.fits"

    if coadd_path.exists():
        coadd = fits.getdata(coadd_path).astype(np.float32)
        plot_coadd_thumbnail(coadd, out_dir / "coadd_thumbnail.png")
    else:
        log.warning("coadd.fits not found at %s", coadd_path)

    if contrib_path.exists():
        contributors = fits.getdata(contrib_path)
        plot_contributor_map(contributors, out_dir / "contributors_map.png")
    else:
        log.warning("contributors_map.fits not found at %s", contrib_path)

    plot_zp_histogram(["placeholder"], out_dir / "zp_histogram.png")
