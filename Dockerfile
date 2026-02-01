FROM python:3.9-slim

# Evita arquivos .pyc e melhora logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Instala dependências Python (cache eficiente)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# NLTK data
RUN python -m nltk.downloader punkt wordnet stopwords

# Copia o projeto
COPY . .

# Entry point
RUN chmod +x /app/entrypoint.sh

EXPOSE 8000

CMD ["sh", "/app/entrypoint.sh"]
