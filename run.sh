#!/bin/sh
# Arranca la interfaz y la API de Temis en http://127.0.0.1:8765 y en el Wi-Fi para el celular (/m, con clave)
cd "$(dirname "$0")" || exit 1
exec ./.venv/bin/python -m temis.server --port "${PORT:-8765}" --lan "$@"
