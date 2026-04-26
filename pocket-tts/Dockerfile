FROM dustynv/pytorch:2.6-r36.4.0-cu128

# The base image uses pypi.jetson-ai-lab.dev (self-signed cert) as primary index.
# We need PyPI for pocket-tts and its deps, and --trusted-host for the Jetson index.
ENV PIP_EXTRA_INDEX_URL=https://pypi.org/simple/
ENV PIP_TRUSTED_HOST=pypi.jetson-ai-lab.dev

# Install pocket-tts without deps (base image provides GPU PyTorch)
RUN pip install --no-cache-dir pocket-tts --no-deps

# Install pocket-tts runtime dependencies (everything except torch)
RUN pip install --no-cache-dir \
    "beartype>=0.22.5" \
    "einops>=0.4.0" \
    "huggingface-hub>=0.10" \
    "numpy>=2" \
    "pydantic>=2" \
    "python-multipart>=0.0.21" \
    "requests>=2.20.0" \
    "safetensors>=0.4.0" \
    "scipy>=1.5.0" \
    "sentencepiece>=0.2.1" \
    "typer>=0.10.0" \
    "typing-extensions>=4.0.0" \
    "soundfile>=0.12.0"

# Install server dependencies (flask, waitress)
# --ignore-installed needed because base image has distutils-installed blinker
COPY server/requirements-jetson.txt /tmp/requirements-jetson.txt
RUN pip install --no-cache-dir --ignore-installed -r /tmp/requirements-jetson.txt

# Copy server code and voices
COPY server/ /app/
COPY voices/ /app/voices/

WORKDIR /app

ENV POCKET_TTS_VOICES_DIR=/app/voices
ENV POCKET_TTS_HOST=0.0.0.0
ENV POCKET_TTS_PORT=8880

EXPOSE 8880

CMD ["python3", "server.py", "--host", "0.0.0.0", "--port", "8880", "--voices-dir", "/app/voices"]
