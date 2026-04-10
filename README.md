# DASCH_AllSkyTest — Coadd Prototype Pipeline

This repository contains a **reproducible prototype pipeline** for building a
median image coadd from [DASCH](https://dasch.cfa.harvard.edu/) plate mosaics.
The initial target is a **5 deg² tile centred on M31** (Andromeda Galaxy).

The pipeline uses DASCH-provided WCS and photometry metadata as-is — no
astrometric re-solving or ubercal recalibration is performed in this initial
prototype, consistent with the SDSS approach of building on existing calibrations
before adding further refinements.

---

## Quick-start (Docker — recommended)

```bash
# 1. Clone the repo
git clone https://github.com/danisimm/DASCH_AllSkyTest.git
cd DASCH_AllSkyTest

# 2. Build the Docker image (takes ~5 min the first time; cached after)
docker build -f docker/Dockerfile -t dasch-coadd:latest .

# 3. Run in test mode (synthetic plates, fast, ~1 min)
mkdir -p workspace
docker run --rm \
  -e RUN_MODE=test \
  -v "$(pwd)/workspace:/work/output_host" \
  -v "$(pwd)/workspace/tmp:/work/tmp" \
  dasch-coadd:latest \
  python /work/pipeline/run_coadd.py \
    --config /work/config/m31.json \
    --out-dir /work/output_host

# Outputs appear in ./workspace/
```

### Run with real DASCH plates (full mode)

```bash
# Pre-download plates for the M31 tile (requires network access to DASCH DR7)
mkdir -p data/m31_plates
python scripts/dasch_download.py \
  --ra 10.684583 --dec 41.269167 \
  --radius_deg 1.3 \
  --outdir data/m31_plates

# Run full pipeline
docker run --rm \
  -e RUN_MODE=full \
  -e INPUT_DIR=/work/data/m31_plates \
  -v "$(pwd)/data:/work/data" \
  -v "$(pwd)/workspace:/work/output_host" \
  -v "$(pwd)/workspace/tmp:/work/tmp" \
  dasch-coadd:latest \
  python /work/pipeline/run_coadd.py \
    --config /work/config/m31.json \
    --out-dir /work/output_host \
    --input-dir /work/data/m31_plates
```

### Local Python (no Docker)

```bash
pip install -r docker/requirements.txt
RUN_MODE=test python pipeline/run_coadd.py \
  --config config/m31.json \
  --out-dir /tmp/dasch_output \
  --tmp-dir /tmp/dasch_tmp
```

---

## Repository layout

```
.
├── config/
│   └── m31.json                  Tile definition (M31, 5 deg²)
├── docker/
│   ├── Dockerfile                Reproducible Python 3.10 image
│   └── requirements.txt          Python dependencies
├── pipeline/
│   ├── run_coadd.py              Main pipeline script
│   └── notebooks/
│       └── coadd_prototype.ipynb Interactive demo notebook
├── scripts/
│   ├── dasch_download.py         DASCH API plate downloader
│   └── qa_plots.py               QA PNG plot generation
├── .github/
│   └── workflows/
│       └── coadd.yml             GitHub Actions CI workflow
└── README.md                     This file
```

---

## Pipeline outputs

| File | Description |
|------|-------------|
| `coadd.fits` | Median coadd image |
| `weight_map.fits` | Sum of per-plate inverse-variance weights |
| `mask_map.fits` | Logical OR of all plate masks (1 = masked) |
| `contributors_map.fits` | Number of valid plates per pixel |
| `source_catalog.fits` | SEP source catalog (RA, Dec, flux, fluxerr) |
| `provenance.log` | Plate IDs and pipeline provenance |
| `coadd_thumbnail.png` | Quick-look PNG of the coadd |
| `contributors_map.png` | Heatmap of contributor counts |
| `zp_histogram.png` | Per-plate zero-point distribution (placeholder if ZP metadata absent) |

---

## GitHub Actions workflow

The workflow (`.github/workflows/coadd.yml`) triggers on:
- Every pull-request open / update (`RUN_MODE=test`, synthetic plates, ≤ 6 plates)
- Manual dispatch (`workflow_dispatch`), where you can choose `run_mode=full`

Artifacts (FITS + PNG) are uploaded and retained for 14 days.

### Running in full mode via the Actions UI

1. Go to **Actions → DASCH Coadd Pipeline → Run workflow**
2. Set `run_mode` to `full`
3. Trigger — note this requires network access to the DASCH DR7 API and may
   produce larger artifacts (several hundred MB for many plates)

For full-scale runs with many DASCH plates consider running on a self-hosted
runner with adequate disk/memory and optionally routing outputs to S3 (secrets
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `S3_BUCKET` can be added later).

---

## Roadmap / future steps

- [ ] Astrometric refinement to Gaia DR3 using SCAMP
- [ ] Ubercal-style per-plate photometric homogenization (sparse linear solver)
- [ ] PSF characterization with PSFEx and kernel convolution to common PSF
- [ ] Multi-epoch / decade coadds
- [ ] Large-scale parallelism on HPC / cloud
