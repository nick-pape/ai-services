# Build whisper.cpp with CUDA on Jetson Orin
FROM nvidia/cuda:12.6.1-devel-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git cmake g++ make wget \
    && rm -rf /var/lib/apt/lists/*

# Clone whisper.cpp
WORKDIR /build
RUN git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git

# Build with CUDA support for Orin (SM 8.7)
WORKDIR /build/whisper.cpp
# Create libcuda.so.1 symlink from stubs (real driver is mounted at runtime by nvidia runtime)
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1

ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:${LIBRARY_PATH}

RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="87" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--allow-shlib-undefined" \
    && cmake --build build --target whisper-server -j4

# Download base.en model
RUN mkdir -p /models && \
    wget -q -O /models/ggml-base.en.bin \
    https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin

# --- Runtime stage ---
FROM nvidia/cuda:12.6.1-runtime-ubuntu22.04

# Install ffmpeg for audio conversion and curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# Copy built binary and model
COPY --from=builder /build/whisper.cpp/build/bin/whisper-server /usr/local/bin/whisper-server
COPY --from=builder /build/whisper.cpp/build/src/libwhisper.so* /usr/local/lib/
COPY --from=builder /build/whisper.cpp/build/ggml/src/libggml*.so* /usr/local/lib/
COPY --from=builder /build/whisper.cpp/build/ggml/src/ggml-cuda/libggml-cuda*.so* /usr/local/lib/
COPY --from=builder /models /models

RUN ldconfig

EXPOSE 8000

ENTRYPOINT ["whisper-server"]
CMD ["--model", "/models/ggml-base.en.bin", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--inference-path", "/v1/audio/transcriptions", \
     "--convert"]
