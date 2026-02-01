#!/bin/sh

echo "📦 Aplicando migrações..."
python manage.py migrate --noinput

echo "📁 Coletando arquivos estáticos..."
python manage.py collectstatic --noinput

echo "🚀 Iniciando servidor Django..."
python manage.py runserver 0.0.0.0:8000
