FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home --uid 1000 notebook && chown -R notebook:notebook /app

USER notebook

EXPOSE 8888

CMD ["jupyter", "notebook", "Pathfinding_HCMC.ipynb", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
