# AI Services Stack for Jetson Nano

This project provides a Docker Compose configuration that spins up a complete AI services stack optimized for NVIDIA Jetson Nano. The stack includes local language models, text-to-speech, speech processing, and a web interface for easy interaction with AI services.

## Services Included

- **Ollama** - Local language model server for running various LLMs (Large Language Models) with API endpoints for chat completion and embeddings
- **Open WebUI** - Modern web interface for interacting with AI models, providing an intuitive chat interface
- **Speaches** - Speech processing and recognition service providing speech-to-text functionality
- **Kokoro** - Fast and high-quality text-to-speech service using Kokoro TTS models

## Prerequisites

- **NVIDIA Jetson Nano** with JetPack installed
- **Docker** and **Docker Compose** installed
- **NVIDIA Container Runtime** configured
- Sufficient storage for AI models (recommended: at least 32GB free space)

## Quick Start

1. Clone this repository:
   ```bash
   # Replace <your-username>/<your-repo> with the actual repository path if using a fork
   git clone https://github.com/<your-username>/<your-repo>.git
   cd ai-services
   ```

2. Start all services:
   ```bash
   docker-compose up -d
   ```

3. Access the web interface:
   - Open WebUI: http://localhost:8080 (when using host networking)
   - Ollama API: http://localhost:11434
   - Kokoro TTS API: http://localhost:8880
   - Speaches API: http://localhost:8000

## Configuration

The stack uses environment variables for port configuration. You can customize ports by creating a `.env` file:

```env
OLLAMA_PORT=11434
KOKORO_PORT=8880
SPEACHES_PORT=8000
```

## Service Details

### Ollama
- **Purpose**: Runs local language models for chat, completion, and embeddings
- **Default Port**: 11434
- **Data Storage**: `/data/cache/ollama`
- **GPU**: CUDA-enabled with full GPU access

### Open WebUI
- **Purpose**: Web-based chat interface for AI interactions
- **Network Mode**: Host (for seamless service communication)
- **Data Storage**: `./open-webui` directory
- **Connected Services**: Ollama and Kokoro TTS

### Speaches
- **Purpose**: Speech-to-text and audio processing
- **Default Port**: 8000
- **Data Storage**: `/data/cache/huggingface`
- **Health Check**: Available at `/health` endpoint

### Kokoro
- **Purpose**: High-quality text-to-speech generation
- **Default Port**: 8880
- **Platform**: ARM64/v8 optimized
- **GPU**: CUDA 12.6+ required

## Managing Services

Start all services:
```bash
docker-compose up -d
```

Stop all services:
```bash
docker-compose down
```

View logs:
```bash
docker-compose logs -f
```

View logs for a specific service:
```bash
docker-compose logs -f ollama
```

Restart a specific service:
```bash
docker-compose restart ollama
```

## GPU Configuration

All services are configured to use NVIDIA GPU acceleration:
- Platform: `linux/arm64/v8`
- Runtime: `nvidia`
- GPU Access: All available GPUs
- CUDA Version: 12.6+

## Storage Requirements

The stack uses persistent volumes for:
- Ollama models: `/data/cache/ollama`
- HuggingFace models: `/data/cache/huggingface`
- Open WebUI data: `./open-webui`

Ensure these directories exist and have appropriate permissions before starting the services.

## Troubleshooting

### Services won't start
- Verify NVIDIA Container Runtime is installed: `docker run --rm --gpus all nvidia/cuda:12.6.0-base nvidia-smi`
- Check available disk space
- Review logs: `docker-compose logs`

### GPU not detected
- Ensure JetPack is properly installed
- Verify NVIDIA runtime in Docker: `docker info | grep -i runtime`
- Check GPU visibility: `nvidia-smi`

### Port conflicts
- Modify port mappings in `.env` file or `docker-compose.yml`
- Check for services already using default ports: `netstat -tulpn | grep -E '8000|8080|8880|11434'`

## License

This project configuration is provided as-is. Individual services have their own licenses:
- Ollama: MIT License
- Open WebUI: MIT License
- Speaches and Kokoro: Check respective project repositories

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests to improve this AI services stack configuration.
