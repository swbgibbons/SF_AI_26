#!/bin/bash
set -e

python manage.py collectstatic --noinput

PORT="${PORT:-8000}"

# Timeouts (must align with Cloud Run = 3600)
TIMEOUT="${TIMEOUT:-3600}"
GRACEFUL_TIMEOUT="${GRACEFUL_TIMEOUT:-3600}"

# Concurrency (Cloud Run should be 1 for heavy models)
WORKERS="${WORKERS:-1}"
THREADS="${THREADS:-2}"
WORKER_CLASS="${WORKER_CLASS:-gthread}"

exec gunicorn backend.wsgi:application \
  --bind "0.0.0.0:${PORT}" \
  --workers "${WORKERS}" \
  --worker-class "${WORKER_CLASS}" \
  --threads "${THREADS}" \
  --timeout "${TIMEOUT}" \
  --graceful-timeout "${GRACEFUL_TIMEOUT}"