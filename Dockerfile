# App image: Streamlit front end plus retrieval (embedding model and reranker on CPU).
# Qdrant and Ollama run as separate containers, see docker-compose.yml.
FROM python:3.12-slim

WORKDIR /app

# CPU-only PyTorch: the default wheel from PyPI adds about 2 GB of unused CUDA libraries
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

# Dependencies before the code: this layer stays cached until the requirements change
COPY requirements.txt requirements-embed.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-embed.txt

COPY .streamlit ./.streamlit
COPY src ./src
COPY app ./app
# Import the package straight from src (config.ROOT then resolves to /app, so /app/data works)
ENV PYTHONPATH=/app/src

# Run as an unprivileged user; the cache folder must exist so the volume gets this owner
RUN useradd --create-home appuser
USER appuser
RUN mkdir -p /home/appuser/.cache/huggingface

EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
