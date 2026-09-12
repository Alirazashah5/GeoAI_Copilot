FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY geoai_copilot ./geoai_copilot
COPY app ./app
COPY data ./data

RUN pip install --no-cache-dir .

EXPOSE 8501 8000

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
