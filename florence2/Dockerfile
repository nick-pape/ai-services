FROM dustynv/pytorch:2.6-r36.4.0-cu128

WORKDIR /app

RUN pip install --quiet --index-url https://pypi.org/simple/ \
    transformers==4.40.0 einops Pillow fastapi uvicorn[standard] && \
    pip install --quiet --no-deps --index-url https://pypi.org/simple/ timm

COPY server.py /app/server.py

CMD ["python3", "/app/server.py"]
