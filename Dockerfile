FROM python:3.11-slim

RUN pip install uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN apt-get update && apt-get install -y ffmpeg && uv sync --frozen

COPY . .

CMD ["uv", "run", "python", "main.py"]