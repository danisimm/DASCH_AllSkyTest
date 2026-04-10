FROM python:3.10-slim

# System dependencies needed for numpy, sep, reproject, and optional SWarp
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libjpeg-dev \
        zlib1g-dev \
        swarp \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

# Install Python dependencies
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy pipeline code and scripts into the image
COPY pipeline/ /work/pipeline/
COPY scripts/ /work/scripts/
COPY config/ /work/config/

# Make scripts executable
RUN chmod +x /work/pipeline/run_coadd.py /work/scripts/dasch_download.py /work/scripts/qa_plots.py

ENV PYTHONUNBUFFERED=1

CMD ["python", "/work/pipeline/run_coadd.py", "--config", "/work/config/m31.json"]
