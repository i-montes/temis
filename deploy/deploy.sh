#!/bin/sh
# Deploy the Temis API to the VPS (ssh host "vps"), in ~/containers/temis.
#
#   deploy/deploy.sh                 # the active local version (models/active.json)
#   deploy/deploy.sh v1108-2307      # a given run
#
# Copies the code and the int8 model, points TEMIS_MODEL_DIR at it, rebuilds and restarts the
# container and waits for /health. The first run creates the token in the VPS .env (chmod 600)
# and prints it once; later runs keep it.
set -eu
cd "$(dirname "$0")/.."
HOST=${TEMIS_HOST:-vps}
DIR=containers/temis
RUN=${1:-$(./.venv/bin/python -c 'import json,pathlib;print(pathlib.Path(json.load(open("models/active.json"))["dir"]).parent.name)')}
SRC=runs/$RUN/onnx
[ -f "$SRC/decisor.int8.onnx" ] || { echo "no int8 export in $SRC" >&2; exit 1; }

ssh "$HOST" "mkdir -p $DIR/temis $DIR/models/$RUN"
rsync -a deploy/Dockerfile deploy/docker-compose.yml deploy/requirements.txt deploy/README.md "$HOST:$DIR/"
rsync -a temis/__init__.py temis/api.py temis/runtime.py temis/schema.py temis/encoding.py \
      temis/naming.py "$HOST:$DIR/temis/"
rsync -a --partial "$SRC/config.json" "$SRC/tokenizer.json" "$SRC/decisor.int8.onnx" "$HOST:$DIR/models/$RUN/"

ssh "$HOST" sh -s "$DIR" "$RUN" <<'REMOTE'
set -eu
cd "$HOME/$1"
if [ ! -f .env ]; then
  umask 077
  echo "TEMIS_TOKEN=$(python3 -c 'import secrets;print(secrets.token_urlsafe(32))')" > .env
  echo "TEMIS_MODEL_DIR=/models/$2" >> .env
  echo "[deploy] new token (shown once): $(sed -n 's/^TEMIS_TOKEN=//p' .env)"
else
  sed -i "s|^TEMIS_MODEL_DIR=.*|TEMIS_MODEL_DIR=/models/$2|" .env
fi
docker compose up -d --build
for i in $(seq 1 30); do
  if curl -fsS http://127.0.0.1:8093/health; then echo; exit 0; fi
  sleep 2
done
docker compose logs --tail 40
exit 1
REMOTE
