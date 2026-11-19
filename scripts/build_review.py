"""Training cases for Clio's `needs_review` question: does a turn's work deserve an independent check?

  python -m scripts.build_review            # writes data/review_targets.jsonl
  python -m scripts.train ... --extra data/review_targets.jsonl

Clio asks it after a turn that ran commands or wrote files, with the state
{"person_request": ..., "done": ["ran: <command>", "wrote: <path>", ...]} (commands first, then
files, at most 40 entries of 300 characters) and starts a second agent's review at 0.8.

yes: the turn changed code, configuration, a service, a schedule, a database, permissions, or
     deleted/rewrote data.
no:  it only read or measured, or made a trivial change that is right at a glance (a note, a
     folder, a rename the person dictated).

The cases are written here, not by a teacher: hand-written (request, done) pairs in Spanish and
English with slots (project, stack, paths, services...), combined with read-only steps before the
change, long lists, and requests that do not match what was done (a "fix it" turn that only read
is a no; a "why does it fail?" turn that rewrote a config is a yes), so the answer has to come
from `done`. Clio's own test (data/review_test.jsonl) is never trained on: a case that repeats one
of its requests or its exact `done` list is dropped.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

QID = "needs_review"
# verbatim from Clio (crates/clio-brain/src/cli/chat/review.rs, NEEDS_REVIEW_TEXT)
QUESTION = ("The work listed in `done` was carried out on this machine to answer `person_request`. It needs "
            "an independent check by someone else: it changes code, configuration, a service or data, a "
            "mistake in it could go unnoticed, and it is not trivial to see at a glance that it is right.")
OUT = Path("data/review_targets.jsonl")
TEST = Path("data/review_test.jsonl")
YES, NO = [0.05, 0.95], [0.95, 0.05]  # (false, true)
MAX_DONE, MAX_ENTRY = 40, 300          # Clio's cuts

W = str.split
PROJECTS = W("tienda-api blog-astro inventario billing-service newsletter-bot crm-ventas analytics-etl "
             "mobile-backend landing-eventos scraper-prensa auth-server kiosko agenda-medica portal-clientes "
             "facturador notificador reservas-hotel api-gateway data-pipeline wiki-interna chatbot-soporte "
             "catalogo ledger monitor-precios encuestas ml-recomendador cms-noticias pos-restaurante geo-rutas "
             "shop-frontend ticketing docs-site image-resizer url-shortener feature-flags search-indexer "
             "video-encoder mailroom quiz-app taskboard clio temis lasilla-web nomina")
HOMES = W("/home/ubuntu /home/ivan /Users/ivan /home/ana /root /home/deploy /home/marta /Users/sofia")
ROOTS = W("{home}/proyectos {home} {home}/code {home}/dev /srv /opt /var/www {home}/apps {home}/repos {home}/work")
MODS = W("auth billing cart orders users invoice search upload notify export report shipping inventory session "
         "payments cache parser scheduler mailer webhook pricing discounts login profile sync importer validator "
         "queue metrics pedidos facturas usuarios envios carrito clientes reportes pagos descuentos cobros")
TABLES = W("orders users invoices payments products sessions events customers pedidos facturas clientes "
           "productos envios suscripciones logs_audit tickets comments carts cupones empleados")
COLS = W("status email created_at total phone estado correo notas deleted_at tenant_id sku precio country role")
PKGS = W("axios lodash requests django serde express fastapi pydantic react tokio numpy sqlalchemy zod vite "
         "pandas celery redis stripe openssl nginx typescript eslint pytest")
DOMAINS = W("ejemplo.com mitienda.co app.example.org lasilla.example api.acme.dev tienda.example.net intranet.local "
            "blog.example.com panel.miempresa.co docs.example.io")
USERS = W("deploy ana carlos sofia backup ci monitor jorge laura webadmin")
BRANCHES = W("develop staging feature/checkout fix/login release/2.3 hotfix/pagos feat/export main-old")
NOTE_DIRS = W("{home}/notas {home}/Documentos {home}/borradores {home}/Desktop {home}/documentos/personal "
              "{home}/Dropbox/notas {home}/obsidian/diario {home}/Documents {home}/escritorio {home}/apuntes")
NOTES = W("pendientes.md ideas.txt lista-compras.txt reunion-lunes.md todo.txt apuntes-clase.md regalos.txt "
          "libros-por-leer.md receta-arepas.txt plan-viaje.md frases.txt carta.txt resumen.md agenda.txt "
          "meeting-notes.md shopping-list.txt draft.md journal.md packing-list.txt quotes.txt contactos.txt "
          "peliculas.md metas-2027.md recordatorios.txt discurso.md")
FOLDERS = W("viaje-cartagena facturas-2026 fotos-boda proyecto-libro recetas impuestos trabajo personal archive "
            "old-drafts recibos cursos musica wallpapers tax-docs")
WORDS = W("hola listo prueba ok recordar llamar-al-banco test hello done pagar-arriendo comprar-cafe")
PEOPLE = W("marta jorge el-contador mi-jefe laura soporte the-landlord pedro hr mama")
SCRIPTS = W("backup.sh limpiar-logs.sh healthcheck.sh export_csv.py subir_s3.py rotar-claves.sh renew-certs.sh "
            "to_webp.py weekly_report.py sync-folder.sh monitor_disk.sh cleanup.py deploy.sh importar.py notify.sh")
BINS = W("node python3 docker nginx psql cargo go ffmpeg git pm2 redis-cli terraform")
# per stack: source files, test file, check commands, dependency file, dependency command
STACKS = {
    "ts": (W("src/{m}.ts src/lib/{m}.ts src/routes/{m}.ts src/services/{m}.service.ts src/components/{M}.tsx "
             "src/index.ts src/middleware/{m}.ts src/utils/{m}.js app/api/{m}/route.ts"), "test/{m}.test.ts",
           ["npm test", "npm run test -- {m}", "pnpm test", "npx vitest run", "npx tsc --noEmit", "npm run lint",
            "npm run build", "bun test"], "package.json", ["npm install", "npm ci", "pnpm install", "npm install {pkg}@latest", "npm update"]),
    "py": (W("app/{m}.py {m}.py src/core/{m}.py app/services/{m}.py app/models/{m}.py scripts/{m}.py "
             "app/views/{m}.py lib/{m}_utils.py app/main.py"), "tests/test_{m}.py",
           ["pytest -q", "pytest tests/test_{m}.py", "python -m pytest -x", "ruff check .", "mypy .", "python -m compileall -q ."],
           "requirements.txt", ["pip install -r requirements.txt", "uv sync", "poetry lock && poetry install", "pip install -U {pkg}"]),
    "rs": (W("src/{m}.rs crates/core/src/{m}.rs src/main.rs src/lib.rs crates/api/src/{m}.rs src/{m}/mod.rs "
             "crates/store/src/{m}.rs"), "tests/{m}.rs",
           ["cargo test", "cargo build --release", "cargo clippy", "cargo check", "cargo test {m}"],
           "Cargo.toml", ["cargo update", "cargo build", "cargo update -p {pkg}"]),
    "go": (W("internal/{m}/{m}.go cmd/server/main.go pkg/{m}/handler.go internal/{m}/service.go "
             "internal/store/{m}.go pkg/{m}/{m}.go"), "internal/{m}/{m}_test.go",
           ["go test ./...", "go build ./...", "go vet ./...", "go test ./internal/{m}/..."],
           "go.mod", ["go mod tidy", "go get -u ./...", "go get {pkg}@latest"]),
    "php": (W("app/Http/Controllers/{M}Controller.php app/Models/{M}.php routes/web.php routes/api.php "
              "resources/views/{m}.blade.php app/Services/{M}Service.php"), "tests/Feature/{M}Test.php",
            ["php artisan test", "composer test", "vendor/bin/phpunit", "vendor/bin/phpstan analyse"],
            "composer.json", ["composer install", "composer update", "composer require {pkg}"]),
    "rb": (W("app/models/{m}.rb app/controllers/{m}_controller.rb lib/{m}.rb app/services/{m}_service.rb "
             "app/jobs/{m}_job.rb config/routes.rb"), "spec/{m}_spec.rb",
           ["bundle exec rspec", "bin/rails test", "bundle exec rubocop", "bundle exec rspec spec/{m}_spec.rb"],
           "Gemfile", ["bundle install", "bundle update", "bundle update {pkg}"]),
}


def fill(t: str, c: dict) -> str:
    """{slot} -> its value; braces that are not a slot ({} of find, awk's {print $1}) stay."""
    return re.sub(r"\{(\w+)\}", lambda m: str(c[m.group(1)]) if m.group(1) in c else m.group(0), t)


def context(r: random.Random) -> dict:
    c: dict = {"home": r.choice(HOMES)}
    c["p"] = r.choice(PROJECTS)
    c["d"] = fill(r.choice(ROOTS), c) + "/" + c["p"]
    m = r.choice(MODS)
    c.update(m=m, M=m.capitalize(), sym=r.choice(["calcular", "get", "validate", "procesar", "send", "load"]) + m.capitalize())
    files, tf, tests, depf, depcmds = STACKS[r.choice(list(STACKS))]
    c["pkg"] = r.choice(PKGS)
    fs = [fill(f, c) for f in r.sample(files, 4)]
    rel = r.random() < 0.25  # Clio's copy mode records paths as typed, often relative
    for i, f in enumerate(fs, 1):
        c[f"f{i}"] = f if rel else f"{c['d']}/{f}"
    c["src"] = fs[0]
    c["tf"] = fill(tf, c) if rel else f"{c['d']}/{fill(tf, c)}"
    c["test"], c["build"] = (fill(x, c) for x in r.sample(tests, 2))
    c["depf"], c["depcmd"] = depf, fill(r.choice(depcmds), c)
    c["cd"] = r.choice(["", "", f"cd {c['d']} && "])
    c["svc"] = c["p"] + r.choice(["", "", "-worker", "-api", "-web"])
    c["unit"] = c["svc"] + r.choice(["", ".service"])
    c["db"] = c["p"].replace("-", "_") + r.choice(["", "_prod", "_db"])
    c.update(table=r.choice(TABLES), col=r.choice(COLS), port=r.choice([80, 443, 3000, 3306, 4000, 5000, 5432, 6379, 8000, 8080, 8443, 9000, 9090]),
             domain=r.choice(DOMAINS), user=r.choice(USERS), branch=r.choice(BRANCHES),
             ver=f"{r.randint(0, 4)}.{r.randint(0, 20)}.{r.randint(0, 9)}", n=r.choice([5, 10, 15, 20, 30, 50, 100, 200]),
             days=r.choice([7, 14, 30, 60, 90, 180]), ns=r.choice(["prod", "staging", "default", "apps"]),
             bucket=c["p"] + r.choice(["-assets", "-backups", "-static"]), pid=r.randint(900, 64000),
             mig=r.choice(["0007", "0031", "0042", "20261103", "20261118_1420", "V12__"]),
             cron=r.choice(["0 3 * * *", "*/15 * * * *", "0 * * * *", "30 2 * * 0", "*/5 * * * *", "0 6 * * 1-5"]),
             script=r.choice(SCRIPTS), ext=r.choice(["py", "ts", "log", "json", "md", "rs"]),
             mount=r.choice(["/", "/home", "/var", "/data", "/mnt/backups"]), bin=r.choice(BINS),
             nf=r.choice(NOTES), nf2=r.choice(NOTES), folder=r.choice(FOLDERS), word=r.choice(WORDS), person=r.choice(PEOPLE))
    c["sp"] = fill(r.choice(["{home}/bin/{script}", "{home}/scripts/{script}", "{d}/scripts/{script}", "/usr/local/bin/{script}", "/opt/scripts/{script}"]), c)
    c["logdir"] = fill(r.choice(["/var/log/{p}", "{home}/logs", "{d}/logs", "/var/log/nginx", "{d}/storage/logs"]), c)
    c["log"] = c["logdir"] + "/" + r.choice(["app.log", "error.log", "access.log", "worker.log", "cron.log"])
    c["data"] = fill(r.choice(["{d}/data/{table}.csv", "{home}/datos/{table}.csv", "{d}/exports/{table}.json", "/data/{table}.tsv"]), c)
    c["nd"] = fill(r.choice(NOTE_DIRS), c)
    return c


# ---------------------------------------------------------------------------------------------
# (request in Spanish, request in English, done). In `done`: a tuple is one of its alternatives
# ("" = nothing), a leading "?" makes the entry optional.

CHANGES = [
    # code
    ("arregla el bug de {m} en {p}", "fix the bug in {m}", ["wrote: {f1}", "?wrote: {tf}", "?ran: {cd}{test}"]),
    ("el módulo {m} falla cuando el dato viene vacío, arréglalo", "the {m} module crashes on empty input, fix it", ["wrote: {f1}", "?ran: {cd}{test}"]),
    ("hay un bug en {m}: duplica registros. corrígelo", "{m} is creating duplicate records, please fix", ["wrote: {f1}", "wrote: {f2}", "?ran: {cd}{test}"]),
    ("corrige el cálculo de totales en {m}", "{m} returns wrong totals, fix it", ["wrote: {f1}", "?wrote: {tf}"]),
    ("hay una condición de carrera en {m}, resuélvela", "there's a race condition in {m}, sort it out", ["wrote: {f1}", "wrote: {f2}"]),
    ("agrega paginación al listado de {m}", "add pagination to the {m} list", ["wrote: {f1}", "wrote: {f2}", "?wrote: {tf}", "?ran: {cd}{test}"]),
    ("implementa exportar {m} a CSV", "implement CSV export for {m}", ["wrote: {f1}", "?wrote: {f2}", "?ran: {cd}{build}"]),
    ("añade un endpoint para consultar {m}", "add an endpoint to query {m}", ["wrote: {f1}", "wrote: {f2}", "?wrote: {f3}", "?ran: {cd}{test}"]),
    ("agrega validación de entrada en {m}", "add input validation to {m}", ["wrote: {f1}", "?wrote: {tf}", "?ran: {cd}{test}"]),
    ("implementa reintentos con backoff en {m}", "add retry with backoff to {m}", ["wrote: {f1}", "?ran: {cd}{test}"]),
    ("ponle caché a las consultas de {m}", "add caching to the {m} lookups", ["wrote: {f1}", "wrote: {f2}"]),
    ("limita las peticiones por minuto en la API de {m}", "add rate limiting to the {m} API", ["wrote: {f1}", "?wrote: {f2}", "?ran: {cd}{build}"]),
    ("refactoriza {m} para separar la lógica de la base de datos", "refactor {m} to separate the database logic", ["wrote: {f1}", "wrote: {f2}", "wrote: {f3}", "?wrote: {f4}", "?ran: {cd}{test}"]),
    ("renombra la función principal de {m} en todo el proyecto", "rename the main {m} function across the project", ["wrote: {f1}", "wrote: {f2}", "wrote: {f3}", "wrote: {f4}", "?ran: {cd}{build}"]),
    ("divide {m} en módulos más pequeños", "split {m} into smaller modules", ["wrote: {f1}", "wrote: {f2}", "wrote: {f3}"]),
    ("pasa {m} a async", "convert {m} to async", ["wrote: {f1}", "wrote: {f2}", "?ran: {cd}{test}"]),
    ("quita el código muerto de {m}", "remove the dead code in {m}", ["wrote: {f1}", "?wrote: {f2}", "?ran: {cd}{build}"]),
    ("escribe tests para {m} y corrige lo que falle", "write tests for {m} and fix whatever fails", ["wrote: {tf}", "wrote: {f1}", "ran: {cd}{test}"]),
    ("optimiza la consulta lenta de {m}", "speed up the slow query in {m}", ["wrote: {f1}", "?ran: {cd}{test}"]),
    ("haz que {m} registre los errores en vez de tragárselos", "make {m} log errors instead of swallowing them", ["wrote: {f1}", "?wrote: {f2}"]),
    ("cambia {m} para que use {pkg}", "rewrite {m} to use {pkg}", ["wrote: {f1}", "wrote: {d}/{depf}", "ran: {cd}{depcmd}", "?ran: {cd}{test}"]),
    ("termina la función que dejé a medias en {src}", "finish the half-written function in {src}", ["wrote: {f1}"]),
    ("un cambio rápido: sube el timeout por defecto en {src}", "quick one: raise the default timeout in {src}", ["wrote: {f1}"]),
    ("corrige el typo en el mensaje de error de {m}", "fix the typo in {m}'s error message", ["wrote: {f1}", "?ran: {cd}{build}"]),
    # scripts and schedules
    ("escribe un script que comprima los logs viejos, con validación de argumentos", "write a script that compresses old logs, with argument validation", ["wrote: {sp}", "?ran: chmod +x {sp}", ("ran: {sp} --help", "ran: bash {sp} {logdir}", "ran: {sp}", "")]),
    ("hazme un script que revise que los servicios estén arriba", "write me a script that checks the services are up", ["wrote: {sp}", "?ran: chmod +x {sp}", "?ran: {sp}"]),
    ("crea un script en python que exporte la tabla {table} a CSV", "create a python script that exports the {table} table to CSV", ["wrote: {sp}", ("ran: python3 {sp} --dry-run", "ran: python3 {sp}", "")]),
    ("necesito un script que suba los respaldos a S3 y córrelo", "I need a script that uploads the backups to S3, then run it", ["wrote: {sp}", "ran: chmod +x {sp}", "ran: {sp}"]),
    ("arregla el script {script}, falla con rutas con espacios", "fix {script}, it breaks on paths with spaces", ["wrote: {sp}", "?ran: bash -n {sp}"]),
    ("programa {script} para que corra cada noche", "schedule {script} to run every night", [("ran: (crontab -l; echo '{cron} {sp}') | crontab -", "wrote: /etc/cron.d/{p}", "ran: crontab {home}/crontab.txt")]),
    ("crea un script de respaldo de {db} y ponlo en cron", "write a backup script for {db} and add it to cron", ["wrote: {sp}", "ran: chmod +x {sp}", ("ran: (crontab -l 2>/dev/null; echo '{cron} {sp} >> {log} 2>&1') | crontab -", "wrote: /etc/cron.d/{p}-backup")]),
    ("crea un timer de systemd que corra {script} cada hora", "create a systemd timer that runs {script} hourly", ["wrote: /etc/systemd/system/{p}.timer", "wrote: /etc/systemd/system/{p}.service", "ran: sudo systemctl daemon-reload && sudo systemctl enable --now {p}.timer"]),
    ("quita la tarea de cron que manda el reporte", "remove the cron job that sends the report", [("ran: crontab -l | grep -v {script} | crontab -", "ran: sudo rm /etc/cron.d/{p}", "ran: crontab -r")]),
    # web server and app configuration
    ("configura nginx para que {domain} apunte al puerto {port}", "configure nginx to proxy {domain} to port {port}", [("wrote: /etc/nginx/sites-available/{p}", "wrote: /etc/nginx/conf.d/{p}.conf"), "?ran: sudo ln -s /etc/nginx/sites-available/{p} /etc/nginx/sites-enabled/", ("ran: sudo nginx -t && sudo systemctl reload nginx", "ran: sudo systemctl reload nginx", "ran: sudo nginx -s reload", "")]),
    ("agrega la redirección de http a https en {domain}", "add an http to https redirect for {domain}", [("wrote: /etc/nginx/sites-available/{domain}", "wrote: /etc/caddy/Caddyfile", "wrote: /etc/apache2/sites-available/{p}.conf"), ("ran: sudo systemctl reload nginx", "ran: sudo systemctl restart caddy", "ran: sudo apachectl graceful")]),
    ("sube el límite de subida de archivos a 50MB", "raise the upload limit to 50MB", ["wrote: /etc/nginx/nginx.conf", "?ran: sudo nginx -t", "ran: sudo systemctl reload nginx"]),
    ("activa gzip en el servidor web", "enable gzip on the web server", [("wrote: /etc/nginx/conf.d/gzip.conf", "wrote: /etc/nginx/nginx.conf", "wrote: /etc/caddy/Caddyfile"), "?ran: sudo systemctl reload nginx"]),
    ("pon el certificado SSL para {domain}", "set up the SSL certificate for {domain}", [("ran: sudo certbot --nginx -d {domain} --non-interactive --agree-tos", "ran: sudo certbot certonly --standalone -d {domain}"), "?ran: sudo systemctl reload nginx"]),
    ("cambia el puerto de {svc} a {port} y reinicia", "change {svc}'s port to {port} and restart it", [("wrote: {d}/.env", "wrote: {d}/config/production.yaml", "wrote: {d}/docker-compose.yml"), ("ran: pm2 restart {svc}", "ran: sudo systemctl restart {unit}", "ran: {cd}docker compose up -d")]),
    ("sube el timeout de {svc} a 60 segundos", "raise the {svc} timeout to 60 seconds", [("wrote: {d}/config.toml", "wrote: {d}/settings.json", "wrote: /etc/{p}/config.yml", "wrote: {d}/config/settings.py"), "?ran: sudo systemctl restart {unit}"]),
    ("apunta {svc} a la base de datos nueva", "point {svc} at the new database", [("wrote: {d}/.env", "wrote: {d}/.env.production", "wrote: {d}/config/database.yml"), ("ran: pm2 reload {svc} --update-env", "ran: {cd}docker compose restart {svc}", "ran: sudo supervisorctl restart {svc}", "")]),
    ("activa el modo debug en {svc}", "turn on debug mode for {svc}", [("wrote: {d}/.env", "wrote: {d}/config/settings.py", "wrote: {d}/ecosystem.config.js"), "?ran: pm2 restart {svc}"]),
    ("sube los workers de {svc} a 8", "bump {svc} to 8 workers", [("wrote: {d}/gunicorn.conf.py", "wrote: {d}/ecosystem.config.js", "wrote: {d}/docker-compose.yml", "wrote: /etc/supervisor/conf.d/{p}.conf"), ("ran: sudo supervisorctl reread && sudo supervisorctl update", "ran: pm2 restart {svc}", "ran: {cd}docker compose up -d --scale {svc}=8", "")]),
    ("rota la clave de la API de pagos y actualiza la app", "rotate the payments API key and update the app", ["wrote: {d}/.env", ("ran: pm2 restart {svc}", "ran: {cd}docker compose up -d", "ran: sudo systemctl restart {unit}")]),
    ("crea un servicio de systemd para {svc}", "create a systemd service for {svc}", [("wrote: /etc/systemd/system/{svc}.service", "wrote: {home}/.config/systemd/user/{svc}.service"), ("ran: sudo systemctl daemon-reload && sudo systemctl enable --now {svc}", "ran: systemctl --user daemon-reload", "ran: sudo systemctl daemon-reload")]),
    ("ponle límite de memoria al servicio {svc}", "add a memory limit to the {svc} service", ["wrote: /etc/systemd/system/{svc}.service.d/override.conf", "ran: sudo systemctl daemon-reload", "?ran: sudo systemctl restart {svc}"]),
    ("haz que {svc} arranque solo al prender la máquina", "make {svc} start on boot", [("ran: sudo systemctl enable {unit}", "ran: pm2 startup && pm2 save", "wrote: /etc/systemd/system/{svc}.service")]),
    # services
    ("reinicia {svc}", "restart {svc}", [("ran: sudo systemctl restart {unit}", "ran: pm2 restart {svc}", "ran: docker restart {svc}", "ran: kubectl rollout restart deploy/{svc} -n {ns}", "ran: sudo service {svc} restart")]),
    ("detén el servicio {svc}", "stop the {svc} service", [("ran: sudo systemctl stop {unit}", "ran: pm2 stop {svc}", "ran: docker stop {svc}", "ran: sudo supervisorctl stop {svc}", "ran: sudo systemctl disable --now {unit}")]),
    ("levanta de nuevo los contenedores de {p}", "bring the {p} containers back up", [("ran: {cd}docker compose up -d --force-recreate", "ran: docker compose -f {d}/docker-compose.yml up -d", "ran: {cd}docker compose down && docker compose up -d")]),
    ("{svc} está colgado, mátalo y vuelve a levantarlo", "{svc} is hung, kill it and start it again", ["ran: sudo kill -9 {pid}", ("ran: sudo systemctl start {unit}", "ran: pm2 start {svc}", "ran: {cd}nohup ./run.sh &")]),
    ("escala {svc} a 3 réplicas", "scale {svc} to 3 replicas", [("ran: kubectl scale deploy/{svc} --replicas=3 -n {ns}", "ran: {cd}docker compose up -d --scale {svc}=3", "ran: pm2 scale {svc} 3")]),
    ("borra el proceso {svc} de pm2", "remove {svc} from pm2", ["ran: pm2 delete {svc}", "?ran: pm2 save"]),
    ("reinicia el servidor", "reboot the server", [("ran: sudo reboot", "ran: sudo shutdown -r now", "ran: sudo systemctl reboot")]),
    # databases and data
    ("agrega la columna {col} a la tabla {table}", "add a {col} column to the {table} table", [("wrote: {d}/migrations/{mig}_add_{col}.sql", "wrote: {d}/db/migrate/{mig}_add_{col}_to_{table}.rb", "wrote: {d}/alembic/versions/{mig}_{col}.py", "wrote: {d}/prisma/schema.prisma"), ("ran: psql {db} -f {d}/migrations/{mig}_add_{col}.sql", "ran: {cd}bin/rails db:migrate", "ran: {cd}alembic upgrade head", "ran: {cd}npx prisma migrate deploy")]),
    ("crea un índice en {table}.{col}", "create an index on {table}.{col}", [("ran: psql {db} -c \"CREATE INDEX CONCURRENTLY idx_{table}_{col} ON {table} ({col})\"", "ran: mysql {db} -e \"ALTER TABLE {table} ADD INDEX idx_{col} ({col})\"", "ran: sqlite3 {d}/data.db \"create index idx_{col} on {table}({col})\"")]),
    ("corre las migraciones pendientes", "run the pending migrations", [("ran: {cd}php artisan migrate --force", "ran: {cd}python manage.py migrate", "ran: {cd}npx knex migrate:latest", "ran: {cd}alembic upgrade head", "ran: {cd}bin/rails db:migrate RAILS_ENV=production", "ran: {cd}sqlx migrate run")]),
    ("cambia el tipo de {col} en {table} a texto", "change {table}.{col} to text", ["ran: psql {db} -c \"ALTER TABLE {table} ALTER COLUMN {col} TYPE text\""]),
    ("borra los {table} de prueba de la base", "delete the test rows from {table}", [("ran: psql {db} -c \"DELETE FROM {table} WHERE {col} LIKE 'test%'\"", "ran: mysql {db} -e \"DELETE FROM {table} WHERE {col} LIKE 'test%'\"", "ran: mongosh {db} --eval 'db.{table}.deleteMany({test: true})'")]),
    ("marca como pagados los {table} de ayer", "mark yesterday's {table} as paid", ["ran: psql {db} -c \"UPDATE {table} SET status='paid' WHERE created_at::date = current_date - 1\""]),
    ("corrige los correos con mayúsculas y espacios en {table}", "clean up the emails with capitals and spaces in {table}", [("ran: psql {db} -c \"UPDATE {table} SET {col} = LOWER(TRIM({col}))\"", "ran: mysql {db} -e \"UPDATE {table} SET {col} = LOWER(TRIM({col}))\"")]),
    ("vacía la tabla {table}", "truncate the {table} table", [("ran: psql {db} -c 'TRUNCATE {table} CASCADE'", "ran: mysql {db} -e 'TRUNCATE TABLE {table}'", "ran: sqlite3 {d}/data.db 'delete from {table}'")]),
    ("elimina los usuarios inactivos desde hace {days} días", "remove the users inactive for {days} days", ["ran: psql {db} -c \"DELETE FROM users WHERE last_login < now() - interval '{days} days'\""]),
    ("limpia la caché de redis", "flush the redis cache", [("ran: redis-cli FLUSHALL", "ran: redis-cli -n 2 FLUSHDB", "ran: redis-cli --scan --pattern '{p}:*' | xargs redis-cli del")]),
    ("borra la base de staging, ya no se usa", "drop the staging database, nobody uses it", [("ran: dropdb {db}_staging", "ran: psql -c 'DROP DATABASE {db}_staging'", "ran: mysql -e 'DROP DATABASE {db}_staging'")]),
    ("restaura el respaldo de ayer en {db}", "restore yesterday's backup into {db}", [("ran: pg_restore -d {db} --clean {home}/backups/{db}.dump", "ran: mysql {db} < {home}/backups/{db}.sql", "ran: mongorestore --drop --db {db} {home}/backups/{db}/")]),
    ("importa el CSV de {table} a la base", "import the {table} CSV into the database", [("ran: psql {db} -c \"\\copy {table} FROM '{data}' CSV HEADER\"", "ran: {cd}python manage.py loaddata {table}.json", "ran: mongoimport --db {db} --collection {table} --file {data}")]),
    ("copia la base de producción a staging", "copy the production database over staging", ["ran: pg_dump {db} | psql {db}_staging"]),
    # deleting and rewriting files
    ("borra los logs de más de {days} días", "delete the logs older than {days} days", [("ran: find {logdir} -name '*.log' -mtime +{days} -delete", "ran: find {logdir} -type f -mtime +{days} -exec rm {} \\;", "ran: sudo journalctl --vacuum-time={days}d")]),
    ("limpia la carpeta de builds viejos", "clean out the old builds", [("ran: rm -rf {d}/dist {d}/build", "ran: {cd}rm -rf target/", "ran: git -C {d} clean -fdx")]),
    ("libera espacio en disco", "free up some disk space", [("ran: docker system prune -af --volumes", "ran: rm -rf {home}/.cache/*", "ran: sudo apt-get autoremove -y && sudo apt-get clean", "ran: docker builder prune -af", "ran: sudo rm -rf /var/tmp/{p}-*")]),
    ("elimina los respaldos de más de un mes", "remove the backups older than a month", ["ran: find {home}/backups -mtime +30 -delete"]),
    ("borra node_modules de todos los proyectos", "delete node_modules from every project", ["ran: find {home} -name node_modules -type d -prune -exec rm -rf {} +"]),
    ("vacía el log de {svc}, pesa demasiado", "empty {svc}'s log, it's huge", [("ran: truncate -s 0 {log}", "ran: sudo sh -c '> {log}'", "ran: pm2 flush {svc}")]),
    ("borra la carpeta {d}, ya no la necesito", "delete {d}, I don't need it anymore", ["ran: rm -rf {d}"]),
    ("reemplaza la URL vieja por la nueva en todo el proyecto", "replace the old URL with the new one across the project", ["ran: grep -rl 'old.{domain}' {d} | xargs sed -i 's/old.{domain}/{domain}/g'"]),
    ("cambia todos los print por logging en {p}", "swap every print for logging in {p}", ["ran: {cd}sed -i 's/print(/logger.info(/g' $(git ls-files '*.py')", "?ran: {cd}{test}"]),
    ("normaliza los nombres de columnas del CSV de {table}", "normalize the column names in the {table} CSV", ["ran: python3 -c \"import pandas as pd; d=pd.read_csv('{data}'); d.columns=d.columns.str.lower(); d.to_csv('{data}',index=False)\""]),
    ("quita las líneas duplicadas de {data}", "dedupe {data}", ["ran: sort -u {data} -o {data}"]),
    ("pasa el formateador a todo el código", "run the formatter over the whole codebase", [("ran: {cd}npx prettier --write .", "ran: {cd}ruff check --fix . && ruff format .", "ran: {cd}cargo fmt && cargo clippy --fix --allow-dirty", "ran: {cd}gofmt -w .")]),
    ("sincroniza las fotos al servidor de respaldo", "sync the photos to the backup server", ["ran: rsync -az --delete {home}/fotos/ backup@{domain}:/srv/fotos/"]),
    ("sube el build al bucket", "upload the build to the bucket", [("ran: aws s3 sync {d}/dist s3://{bucket}/ --delete", "ran: gsutil -m rsync -r -d {d}/public gs://{bucket}")]),
    # permissions, users, network
    ("abre el puerto {port} en el firewall", "open port {port} in the firewall", [("ran: sudo ufw allow {port}/tcp", "ran: sudo iptables -A INPUT -p tcp --dport {port} -j ACCEPT", "ran: sudo firewall-cmd --permanent --add-port={port}/tcp && sudo firewall-cmd --reload")]),
    ("cierra el puerto {port}", "close port {port}", [("ran: sudo ufw deny {port}", "ran: sudo ufw delete allow {port}")]),
    ("bloquea la IP 203.0.113.{n}", "block the IP 203.0.113.{n}", [("ran: sudo iptables -A INPUT -s 203.0.113.{n} -j DROP", "ran: sudo ufw deny from 203.0.113.{n}", "wrote: /etc/nginx/conf.d/blocklist.conf")]),
    ("dale permisos de escritura a www-data sobre {d}", "give www-data write access to {d}", [("ran: sudo chown -R www-data:www-data {d}", "ran: sudo chmod -R 775 {d}/storage", "ran: sudo setfacl -R -m u:www-data:rwx {d}")]),
    ("crea el usuario {user} con acceso sudo", "create the user {user} with sudo", ["ran: sudo useradd -m -G sudo {user}", "?wrote: /etc/sudoers.d/{user}"]),
    ("agrega mi llave pública al servidor", "add my public key to the server", ["wrote: {home}/.ssh/authorized_keys", "?ran: chmod 600 {home}/.ssh/authorized_keys"]),
    ("desactiva el login por contraseña en ssh", "disable password login over ssh", ["wrote: /etc/ssh/sshd_config", "ran: sudo systemctl restart ssh"]),
    ("mete a {user} en el grupo docker", "add {user} to the docker group", ["ran: sudo usermod -aG docker {user}"]),
    ("agrega {domain} al archivo hosts", "add {domain} to the hosts file", ["wrote: /etc/hosts"]),
    ("cambia la zona horaria del servidor a Bogotá", "set the server timezone to Bogotá", ["ran: sudo timedatectl set-timezone America/Bogota"]),
    ("configura un swap de 4GB", "set up a 4GB swap", ["ran: sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile", "wrote: /etc/fstab"]),
    ("agrega un alias para desplegar {p} en mi shell", "add a shell alias to deploy {p}", [("wrote: {home}/.bashrc", "wrote: {home}/.zshrc", "wrote: {home}/.bash_aliases")]),
    ("sube el límite de archivos abiertos", "raise the open files limit", [("wrote: /etc/security/limits.conf", "wrote: /etc/sysctl.conf"), "?ran: sudo sysctl -p"]),
    # dependencies and versions
    ("actualiza las dependencias de {p}", "update {p}'s dependencies", ["wrote: {d}/{depf}", "ran: {cd}{depcmd}", "?ran: {cd}{test}"]),
    ("sube {pkg} a la última versión", "bump {pkg} to the latest version", ["wrote: {d}/{depf}", "ran: {cd}{depcmd}"]),
    ("actualiza los paquetes del sistema", "upgrade the system packages", [("ran: sudo apt-get update && sudo apt-get -y upgrade", "ran: sudo apt -y full-upgrade", "ran: sudo dnf -y upgrade", "ran: brew upgrade")]),
    ("sube la versión del proyecto a {ver}", "bump the project version to {ver}", ["wrote: {d}/{depf}", "?ran: git -C {d} tag v{ver} && git -C {d} push --tags"]),
    # deploys, containers, infrastructure
    ("despliega la última versión de {p}", "deploy the latest version of {p}", ["ran: git -C {d} pull origin main", ("ran: {cd}docker compose up -d --build", "ran: {cd}./deploy.sh production", "ran: pm2 reload {svc}", "ran: {cd}npm run build && pm2 restart {svc}")]),
    ("reconstruye la imagen de {p} y súbela", "rebuild the {p} image and push it", ["ran: {cd}docker build -t registry.{domain}/{p}:{ver} .", "ran: docker push registry.{domain}/{p}:{ver}"]),
    ("optimiza el Dockerfile de {p}", "slim down {p}'s Dockerfile", ["wrote: {d}/Dockerfile", "?ran: {cd}docker build -t {p} .", "?ran: {cd}docker compose up -d"]),
    ("haz rollback de {svc} a la versión anterior", "roll {svc} back to the previous version", [("ran: kubectl rollout undo deploy/{svc} -n {ns}", "ran: {cd}git checkout v{ver} && ./deploy.sh", "ran: helm rollback {svc} -n {ns}")]),
    ("saca a producción el cambio de {m}", "ship the {m} change to production", [("ran: {cd}fly deploy", "ran: {cd}npx wrangler deploy", "ran: {cd}vercel --prod", "ran: kubectl apply -f {d}/k8s/ -n {ns}", "ran: helm upgrade {svc} ./chart -n {ns}", "ran: {cd}git push heroku main")]),
    ("agrega un contenedor de redis al compose", "add a redis container to the compose file", ["wrote: {d}/docker-compose.yml", "?ran: {cd}docker compose up -d"]),
    ("aplica los cambios de terraform", "apply the terraform changes", ["?wrote: {d}/infra/main.tf", ("ran: {cd}terraform apply -auto-approve", "ran: {cd}pulumi up --yes", "ran: {cd}cdk deploy --require-approval never")]),
    ("corre el playbook de ansible en los servidores web", "run the ansible playbook on the web servers", ["?wrote: {d}/ansible/roles/web/tasks/main.yml", "ran: {cd}ansible-playbook -i inventory/prod site.yml --limit web"]),
    ("cambia el registro DNS de {domain}", "change the DNS record for {domain}", [("ran: aws route53 change-resource-record-sets --hosted-zone-id Z1D633PJN98FT9 --change-batch file://dns.json", "wrote: {d}/infra/dns.tf", "ran: doctl compute domain records update {domain} --record-id {pid} --record-data 198.51.100.{n}")]),
    ("arregla el workflow de CI, está fallando", "fix the CI workflow, it's failing", [("wrote: {d}/.github/workflows/ci.yml", "wrote: {d}/.gitlab-ci.yml"), "?ran: git -C {d} commit -am 'fix ci' && git -C {d} push"]),
    # git
    ("haz commit de los cambios y súbelos", "commit the changes and push", ["ran: git -C {d} add -A && git -C {d} commit -m 'update {m}' && git -C {d} push"]),
    ("fusiona {branch} en main", "merge {branch} into main", ["ran: {cd}git checkout main && git merge {branch}", "?ran: {cd}git push origin main"]),
    ("deshaz el último commit", "undo the last commit", [("ran: {cd}git reset --hard HEAD~1", "ran: {cd}git revert --no-edit HEAD && git push")]),
    ("haz rebase de mi rama sobre main y fuerza el push", "rebase my branch on main and force-push", ["ran: {cd}git rebase main", "ran: {cd}git push --force-with-lease origin {branch}"]),
    ("borra las ramas que ya se fusionaron", "delete the branches that are already merged", ["ran: {cd}git branch --merged | grep -v main | xargs git branch -d"]),
    ("descarta mis cambios locales en {src}", "throw away my local changes to {src}", [("ran: {cd}git checkout -- {src}", "ran: {cd}git restore {src}", "ran: {cd}git stash drop")]),
]

READS = [
    ("¿cuánto disco queda libre?", "how much disk space is left?", [("ran: df -h", "ran: df -h {mount}", "ran: df -h | grep -v tmpfs")]),
    ("¿qué carpetas ocupan más espacio?", "which folders take the most space?", [("ran: du -sh {home}/* | sort -h | tail", "ran: sudo du -xh / --max-depth=2 | sort -h | tail -n {n}", "ran: du -sh {d}")]),
    ("¿cuánta memoria hay libre?", "how much memory is free?", [("ran: free -h", "ran: free -m", "ran: cat /proc/meminfo | head -5")]),
    ("¿hace cuánto está prendido el servidor?", "how long has the server been up?", ["ran: uptime"]),
    ("¿qué procesador tiene esta máquina?", "what CPU does this machine have?", [("ran: lscpu", "ran: cat /proc/cpuinfo | grep 'model name' | head -1", "ran: nproc && lscpu | grep Model")]),
    ("¿con qué usuario estás corriendo?", "which user are you running as?", [("ran: whoami", "ran: id")]),
    ("dime la IP de la máquina", "tell me this machine's IP", [("ran: hostname -I", "ran: ip -4 a", "ran: curl -s https://api.ipify.org")]),
    ("¿qué sistema operativo es?", "what OS is this?", [("ran: uname -a", "ran: cat /etc/os-release", "ran: lsb_release -a")]),
    ("¿qué procesos consumen más memoria?", "which processes use the most memory?", [("ran: ps aux --sort=-%mem | head -n {n}", "ran: top -bn1 | head -20")]),
    ("¿está corriendo {svc}?", "is {svc} running?", [("ran: systemctl is-active {unit}", "ran: ps aux | grep {svc}", "ran: pm2 list", "ran: docker ps --filter name={svc}", "ran: systemctl status {unit}", "ran: pgrep -fa {svc}")]),
    ("¿qué está escuchando en el puerto {port}?", "what is listening on port {port}?", [("ran: sudo ss -ltnp | grep :{port}", "ran: sudo lsof -i :{port}", "ran: sudo netstat -tlnp | grep {port}")]),
    ("¿qué puertos están abiertos?", "which ports are open?", [("ran: sudo ss -ltnp", "ran: sudo ufw status verbose", "ran: sudo iptables -L -n")]),
    ("¿responde {domain}?", "is {domain} responding?", [("ran: curl -sI https://{domain}", "ran: ping -c 3 {domain}", "ran: curl -s -o /dev/null -w '%{http_code}' https://{domain}")]),
    ("¿a qué IP resuelve {domain}?", "what does {domain} resolve to?", [("ran: dig +short {domain}", "ran: nslookup {domain}", "ran: host {domain}")]),
    ("revisa el health de {svc}", "check {svc}'s health endpoint", ["ran: curl -s http://localhost:{port}/health"]),
    ("muéstrame {src}", "show me {src}", [("ran: cat {f1}", "ran: head -n {n} {f1}", "ran: sed -n '1,{n}p' {f1}")]),
    ("muéstrame las últimas {n} líneas del log de {svc}", "show me the last {n} lines of {svc}'s log", [("ran: tail -n {n} {log}", "ran: pm2 logs {svc} --lines {n} --nostream", "ran: docker logs --tail {n} {svc}", "ran: sudo journalctl -u {unit} -n {n} --no-pager", "ran: kubectl logs deploy/{svc} --tail={n} -n {ns}")]),
    ("¿hay errores en el log de nginx?", "any errors in the nginx log?", [("ran: sudo tail -n {n} /var/log/nginx/error.log", "ran: sudo grep -c ' 500 ' /var/log/nginx/access.log", "ran: sudo grep -i error /var/log/nginx/error.log | tail -n {n}")]),
    ("busca dónde se usa {sym}", "find where {sym} is used", [("ran: grep -rn {sym} {d}/src", "ran: rg -n \"{sym}\" {d}", "ran: git -C {d} grep -n {sym}")]),
    ("¿cuántos archivos .{ext} hay en {p}?", "how many .{ext} files are in {p}?", [("ran: find {d} -name '*.{ext}' | wc -l", "ran: git -C {d} ls-files '*.{ext}' | wc -l")]),
    ("lista lo que hay en {d}", "list what's in {d}", [("ran: ls -la {d}", "ran: tree -L 2 {d}", "ran: ls {d}")]),
    ("¿cuántas líneas tiene {src}?", "how many lines does {src} have?", ["ran: wc -l {f1}"]),
    ("¿cuándo se modificó {depf} por última vez?", "when was {depf} last modified?", [("ran: stat {d}/{depf}", "ran: git -C {d} log -1 --format=%cd -- {depf}", "ran: ls -l {d}/{depf}")]),
    ("¿en qué estado está el repo de {p}?", "what's the state of the {p} repo?", [("ran: git -C {d} status", "ran: git -C {d} status -sb", "ran: {cd}git status --short")]),
    ("muéstrame los últimos commits de {p}", "show me the latest commits in {p}", [("ran: git -C {d} log --oneline -n {n}", "ran: {cd}git log --stat -3")]),
    ("¿qué cambió en mi rama respecto a main?", "what changed on my branch versus main?", [("ran: git -C {d} diff --stat main", "ran: {cd}git diff main...HEAD", "ran: {cd}git log main..HEAD --oneline")]),
    ("¿qué ramas hay en {p}?", "which branches does {p} have?", [("ran: git -C {d} branch -a", "ran: {cd}git branch -vv", "ran: git -C {d} remote -v")]),
    ("¿qué contenedores hay?", "which containers are there?", [("ran: docker ps", "ran: docker ps -a", "ran: docker compose -f {d}/docker-compose.yml ps", "ran: docker stats --no-stream")]),
    ("¿qué imágenes de docker tengo y cuánto pesan?", "which docker images do I have and how big are they?", [("ran: docker images", "ran: docker system df")]),
    ("¿qué tareas tengo en cron?", "what's in my crontab?", [("ran: crontab -l", "ran: sudo cat /etc/cron.d/{p}", "ran: systemctl list-timers")]),
    ("¿cuántos registros tiene la tabla {table}?", "how many rows are in {table}?", [("ran: psql {db} -c \"SELECT count(*) FROM {table}\"", "ran: mysql {db} -e 'SELECT COUNT(*) FROM {table}'", "ran: sqlite3 {d}/data.db \"select count(*) from {table}\"")]),
    ("muéstrame los últimos {n} {table}", "show me the last {n} {table}", ["ran: psql {db} -c \"SELECT * FROM {table} ORDER BY created_at DESC LIMIT {n}\""]),
    ("¿qué tablas tiene la base {db}?", "which tables does {db} have?", [("ran: psql {db} -c '\\dt'", "ran: mysql {db} -e 'SHOW TABLES'", "ran: sqlite3 {d}/data.db .tables")]),
    ("¿cuánta memoria está usando redis?", "how much memory is redis using?", [("ran: redis-cli info memory", "ran: redis-cli dbsize")]),
    ("¿qué versión de {bin} hay instalada?", "which version of {bin} is installed?", [("ran: {bin} --version", "ran: which {bin} && {bin} --version")]),
    ("¿qué dependencias de {p} están desactualizadas?", "which of {p}'s dependencies are outdated?", [("ran: {cd}npm outdated", "ran: {cd}pip list --outdated", "ran: {cd}cargo outdated", "ran: {cd}composer outdated")]),
    ("¿está instalado {pkg}?", "is {pkg} installed?", [("ran: pip show {pkg}", "ran: npm ls {pkg}", "ran: dpkg -l | grep {pkg}", "ran: which {pkg}")]),
    ("¿la configuración de nginx está bien?", "is the nginx config valid?", [("ran: sudo nginx -t", "ran: sudo cat /etc/nginx/sites-enabled/{p}")]),
    ("muéstrame la configuración de {svc}", "show me {svc}'s configuration", [("ran: cat {d}/config.toml", "ran: cat {d}/docker-compose.yml", "ran: sudo cat /etc/systemd/system/{svc}.service", "ran: cat {d}/.env.example", "ran: systemctl cat {unit}")]),
    ("¿cuándo vence el certificado de {domain}?", "when does {domain}'s certificate expire?", [("ran: sudo certbot certificates", "ran: echo | openssl s_client -connect {domain}:443 2>/dev/null | openssl x509 -noout -enddate")]),
    ("¿qué hora tiene el servidor?", "what's the server time?", [("ran: timedatectl", "ran: date -u", "ran: date '+%Y-%m-%d %H:%M'")]),
    ("¿quién se conectó últimamente?", "who logged in recently?", [("ran: last -n {n}", "ran: w", "ran: sudo grep 'Accepted' /var/log/auth.log | tail -n {n}")]),
    ("¿cómo están los pods de {ns}?", "how are the pods in {ns}?", [("ran: kubectl get pods -n {ns}", "ran: kubectl describe deploy/{svc} -n {ns}", "ran: kubectl top pods -n {ns}")]),
    ("¿qué cambiaría terraform si lo aplico?", "what would terraform change if I applied it?", [("ran: {cd}terraform plan", "ran: {cd}terraform plan -no-color | tail -n {n}")]),
    ("¿qué hay en el bucket {bucket}?", "what's in the {bucket} bucket?", [("ran: aws s3 ls s3://{bucket}/", "ran: aws s3 ls s3://{bucket}/ --recursive --summarize | tail -2")]),
    ("¿pasó el último CI?", "did the last CI run pass?", [("ran: gh run list --limit 5", "ran: gh pr checks", "ran: gh pr list")]),
    ("corre los tests de {p} y dime si pasan", "run {p}'s tests and tell me if they pass", ["ran: {cd}{test}"]),
    ("¿compila {p}?", "does {p} build?", ["ran: {cd}{build}"]),
    ("lee mi nota {nf}", "read my {nf} note", [("ran: cat {nd}/{nf}", "ran: head -n {n} {nd}/{nf}")]),
    ("¿qué archivos hay en mis descargas?", "what files are in my downloads?", [("ran: ls -lh {home}/Downloads", "ran: ls -lt {home}/Descargas | head")]),
    ("¿cuánto es 2 a la 32?", "what's 2 to the 32?", [("ran: python3 -c 'print(2**32)'", "ran: echo $((2**32))")]),
    ("¿qué variables de entorno usa {svc}?", "which environment variables does {svc} use?", [("ran: grep -o '^[A-Z_]*' {d}/.env.example", "ran: pm2 env 0 | cut -d: -f1", "ran: grep -rn 'process.env' {d}/src | head -n {n}")]),
    ("¿cuántas peticiones hubo hoy?", "how many requests came in today?", [("ran: sudo wc -l /var/log/nginx/access.log", "ran: sudo awk '{print $1}' /var/log/nginx/access.log | sort | uniq -c | sort -rn | head")]),
    ("¿qué haría el script {script} si lo corro?", "what would {script} do if I ran it?", [("ran: cat {sp}", "ran: bash -n {sp} && head -n {n} {sp}", "ran: {sp} --dry-run")]),
    ("¿hay GPU en esta máquina?", "is there a GPU on this machine?", [("ran: nvidia-smi", "ran: lspci | grep -i vga")]),
    ("compara {nf} con {nf2}", "diff {nf} against {nf2}", ["ran: diff {nd}/{nf} {nd}/{nf2}"]),
    ("¿qué tamaño tiene el respaldo de {db}?", "how big is the {db} backup?", [("ran: ls -lh {home}/backups/", "ran: du -sh {home}/backups/{db}.dump")]),
]

# Read-only too, though the command looks like a change: simulations, plans, checks, test runs.
READS += [
    ("simula la copia de {p} al disco de respaldo, sin copiar nada", "do a dry run of copying {p} to the backup disk, copy nothing", [("ran: rsync -avn {d}/ /mnt/backups/{p}/", "ran: rsync -a --dry-run --delete {d}/ /mnt/backups/{p}/", "ran: rsync -avn --delete {d}/ {user}@{domain}:/srv/{p}/ | tail -n {n}")]),
    ("muéstrame qué haría el playbook en {ns}, sin aplicarlo", "show me what the playbook would do on {ns}, without applying it", [("ran: {cd}ansible-playbook -i inventory/{ns} site.yml --check --diff", "ran: ansible-playbook -i {d}/hosts {d}/deploy.yml --check", "ran: {cd}ansible-playbook site.yml --list-tasks")]),
    ("valida el manifiesto de {svc} sin aplicarlo", "validate {svc}'s manifest without applying it", [("ran: kubectl apply -f {d}/k8s/{svc}.yaml --dry-run=client", "ran: kubectl apply -f {d}/k8s/{svc}.yaml --dry-run=server -n {ns}", "ran: kubectl diff -f {d}/k8s/ -n {ns}"), "?ran: kubectl diff -f {d}/k8s/{svc}.yaml -n {ns}"]),
    ("¿qué cambiaría si actualizo el chart de {svc}?", "what would change if I upgraded {svc}'s chart?", [("ran: helm diff upgrade {svc} {d}/chart -n {ns}", "ran: helm upgrade {svc} {d}/chart -n {ns} --dry-run", "ran: helm template {svc} {d}/chart | head -n {n}")]),
    ("¿hay migraciones sin aplicar en {p}?", "are there unapplied migrations in {p}?", [("ran: {cd}python manage.py showmigrations | grep -v '\\[X\\]'", "ran: {cd}python manage.py migrate --plan", "ran: {cd}alembic current && alembic heads", "ran: {cd}npx prisma migrate status", "ran: {cd}php artisan migrate:status", "ran: {cd}bin/rails db:migrate:status")]),
    ("¿qué paquetes del sistema se actualizarían?", "which system packages would get upgraded?", [("ran: apt list --upgradable", "ran: apt-get -s upgrade | grep ^Inst", "ran: sudo apt-get upgrade --dry-run | tail -n {n}")]),
    ("¿qué instalaría {depcmd}? no lo instales", "what would {depcmd} install? don't install it", [("ran: {cd}npm install --dry-run", "ran: {cd}pip install --dry-run -r requirements.txt", "ran: {cd}composer update --dry-run", "ran: {cd}cargo update --dry-run")]),
    ("revisa que el compose de {p} esté bien formado", "check that {p}'s compose file is well formed", [("ran: docker compose -f {d}/docker-compose.yml config", "ran: docker compose -f {d}/docker-compose.yml config -q")]),
    ("corre la prueba de {m} varias veces a ver si falla", "run the {m} test a few times to see if it fails", [("ran: {cd}for i in 1 2 3 4 5; do {test}; done", "ran: {cd}{test} && {test} && {test}"), "?ran: cat {tf}"]),
    ("pasa el linter por {p} sin corregir nada", "lint {p} without fixing anything", [("ran: {cd}npx eslint src", "ran: {cd}ruff check .", "ran: {cd}cargo clippy", "ran: {cd}npx tsc --noEmit", "ran: {cd}npx prettier --check .", "ran: {cd}golangci-lint run", "ran: {cd}mypy .")]),
    ("¿qué ejecutaría make deploy?", "what would make deploy run?", [("ran: {cd}make -n deploy", "ran: make -C {d} --dry-run deploy", "ran: grep -n -A8 '^deploy:' {d}/Makefile")]),
    ("prueba si la renovación del certificado funcionaría", "test whether the certificate renewal would work", ["ran: sudo certbot renew --dry-run"]),
    ("simula la rotación de logs de {svc}", "simulate {svc}'s log rotation", ["ran: sudo logrotate -d /etc/logrotate.d/{svc}"]),
    ("¿qué incluiría el próximo respaldo de {p}?", "what would the next {p} backup include?", [("ran: restic backup {d} --dry-run -v | tail -n {n}", "ran: borg create --dry-run --list /mnt/backups/repo::prueba {d} | head -n {n}")]),
    ("¿qué logs se borrarían si limpio los de más de {days} días?", "which logs would go if I cleaned those older than {days} days?", [("ran: find {logdir} -name '*.log' -mtime +{days} -print", "ran: find {logdir} -type f -mtime +{days} | wc -l", "ran: find {logdir} -mtime +{days} -exec ls -lh {} +")]),
    ("¿qué traería un pull en {p}?", "what would a pull bring into {p}?", [("ran: git -C {d} fetch --dry-run", "ran: git -C {d} log HEAD..origin/{branch} --oneline", "ran: {cd}git diff HEAD...origin/{branch} --stat")]),
    ("¿qué borraría un git clean en {p}?", "what would git clean remove in {p}?", [("ran: git -C {d} clean -nd", "ran: {cd}git clean -n -x")]),
    ("¿por qué es lenta la consulta de {table}?", "why is the {table} query slow?", [("ran: psql {db} -c \"EXPLAIN ANALYZE SELECT * FROM {table} WHERE {col} = 1\"", "ran: mysql {db} -e \"EXPLAIN SELECT * FROM {table} WHERE {col} = 1\""), "?ran: psql {db} -c '\\d {table}'"]),
    ("¿cuántas filas de {table} se borrarían si limpio las de más de {days} días?", "how many {table} rows would go if I purged those older than {days} days?", ["ran: psql {db} -c \"SELECT count(*) FROM {table} WHERE created_at < now() - interval '{days} days'\""]),
    ("revisa si la unidad de {svc} tiene errores", "check {svc}'s unit file for errors", [("ran: systemd-analyze verify /etc/systemd/system/{svc}.service", "ran: systemctl cat {unit}")]),
    ("muéstrame cómo quedaría {src} si cambio {word} por otra palabra, sin guardarlo", "show me how {src} would look with {word} replaced, without saving", [("ran: sed 's/{word}/otra/g' {f1} | diff {f1} -", "ran: sed -n 's/{word}/otra/gp' {f1}")]),
    ("¿hay vulnerabilidades en las dependencias de {p}?", "any vulnerabilities in {p}'s dependencies?", [("ran: {cd}npm audit", "ran: {cd}pip-audit", "ran: {cd}cargo audit", "ran: {cd}composer audit")]),
    ("revisa {script} con shellcheck", "run shellcheck on {script}", [("ran: shellcheck {sp}", "ran: bash -n {sp}")]),
    ("prueba la conexión a la base {db}", "test the connection to {db}", [("ran: pg_isready -d {db}", "ran: psql {db} -c 'select 1'", "ran: mysqladmin ping")]),
    ("¿qué subiría si sincronizo {bucket}?", "what would get uploaded if I synced {bucket}?", [("ran: aws s3 sync {d}/public s3://{bucket}/ --dryrun", "ran: aws s3 sync {d}/dist s3://{bucket}/ --dryrun --delete")]),
    ("¿qué imágenes y volúmenes se liberarían con un prune?", "what would a prune free up?", [("ran: docker system df -v | head -n {n}", "ran: docker images -f dangling=true")]),
]

TRIVIAL = [
    ("guarda esta lista en {nd}/{nf}", "save this list to {nd}/{nf}", ["wrote: {nd}/{nf}"]),
    ("anota en mis notas lo que te acabo de decir", "write down what I just told you in my notes", ["wrote: {nd}/{nf}"]),
    ("crea una nota con lo que te dicté", "make a note with what I dictated", ["wrote: {nd}/{nf}"]),
    ("guarda el resumen de lo que hablamos en un archivo", "save a summary of what we discussed to a file", ["wrote: {nd}/{nf}"]),
    ("escribe un borrador de correo para {person} y guárdalo", "draft an email to {person} and save it", ["wrote: {nd}/correo-{person}.md"]),
    ("escribe un poema corto y guárdalo en {nf}", "write a short poem and save it as {nf}", ["wrote: {nd}/{nf}"]),
    ("toma nota: {word}", "take a note: {word}", [("wrote: {nd}/{nf}", "ran: echo '{word}' >> {nd}/{nf}")]),
    ("crea la carpeta {folder} en {nd}", "make a {folder} folder in {nd}", [("ran: mkdir -p {nd}/{folder}", "wrote: {nd}/{folder}", "ran: mkdir {nd}/{folder}")]),
    ("crea las carpetas para ordenar mis fotos por año", "create folders to sort my photos by year", ["ran: mkdir -p {home}/fotos/2024 {home}/fotos/2025 {home}/fotos/2026"]),
    ("renombra {nf} a {nf2}", "rename {nf} to {nf2}", ["ran: mv {nd}/{nf} {nd}/{nf2}"]),
    ("mueve {nf} a la carpeta {folder}", "move {nf} into the {folder} folder", ["ran: mv {nd}/{nf} {nd}/{folder}/"]),
    ("haz una copia de {nf} que se llame {nf2}", "make a copy of {nf} called {nf2}", ["ran: cp {nd}/{nf} {nd}/{nf2}"]),
    ("copia {nf} a mi escritorio", "copy {nf} to my desktop", ["ran: cp {nd}/{nf} {home}/Desktop/"]),
    ("crea un archivo vacío {nf}", "create an empty file {nf}", ["ran: touch {nd}/{nf}"]),
    ("agrega '{word}' al final de {nf}", "append '{word}' to {nf}", ["ran: echo '{word}' >> {nd}/{nf}"]),
    ("guarda este enlace en mis marcadores", "save this link to my bookmarks", [("ran: echo 'https://{domain}/articulo' >> {nd}/marcadores.txt", "wrote: {nd}/marcadores.md")]),
    ("crea un archivo de prueba que diga {word}", "create a test file that says {word}", [("wrote: /tmp/{folder}/prueba.txt", "wrote: {home}/prueba.txt", "ran: echo {word} > /tmp/prueba.txt")]),
    ("separa mis pendientes en dos archivos: trabajo y casa", "split my to-dos into two files: work and home", ["wrote: {nd}/trabajo.md", "wrote: {nd}/casa.md"]),
    ("descarga el PDF del informe a mis descargas", "download the report PDF to my downloads", ["ran: curl -sL -o {home}/Downloads/informe-{n}.pdf https://{domain}/informe.pdf"]),
    ("pásame a limpio estos apuntes y guárdalos", "tidy up these notes and save them", ["wrote: {nd}/{nf}"]),
    ("hazme una lista de regalos para {person} y guárdala", "make a gift list for {person} and save it", ["wrote: {nd}/regalos-{person}.txt"]),
]

# requests that do not say what was done: with only reads in `done` they are a no, with a change a yes
LOOK = [
    ("revisa por qué {svc} está lento", "look into why {svc} is slow"),
    ("dime cómo está organizado el proyecto {p}", "tell me how the {p} project is organized"),
    ("hazme un resumen del estado del servidor", "give me a summary of the server's state"),
    ("¿qué hace el módulo {m}?", "what does the {m} module do?"),
    ("revisa los logs de {svc} y dime si hay errores", "go through {svc}'s logs and tell me if there are errors"),
    ("audita qué servicios están expuestos a internet", "audit which services are exposed to the internet"),
    ("explícame el flujo de {m} de principio a fin", "walk me through the {m} flow end to end"),
    ("¿por qué falla el login?", "why is login failing?"),
    ("¿por qué no carga {domain}?", "why isn't {domain} loading?"),
    ("investiga el error 500 de {svc}, pero no cambies nada todavía", "investigate {svc}'s 500 error, but don't change anything yet"),
    ("compara la configuración de staging con la de producción", "compare the staging config with production"),
    ("¿{p} está listo para desplegar?", "is {p} ready to deploy?"),
    ("{svc} se cae cada tanto, mira qué pasa", "{svc} keeps crashing, see what's going on"),
    ("el disco se está llenando, averigua por qué", "the disk is filling up, find out why"),
    ("revisa {p}", "take a look at {p}"),
    ("¿cómo va {svc}?", "how is {svc} doing?"),
]

# The same job done for real or only simulated: (asks for it es, en, asks what would happen es, en, real, simulated).
# Either request goes with either command, so only the command tells whether something changed.
PAIRS = [
    ("sincroniza {p} con el respaldo", "sync {p} to the backup", "¿qué pasaría si sincronizo {p} con el respaldo?", "what would syncing {p} to the backup do?", "ran: rsync -av --delete {d}/ /mnt/backups/{p}/", "ran: rsync -avn --delete {d}/ /mnt/backups/{p}/"),
    ("aplica los cambios de terraform en {ns}", "apply the terraform changes in {ns}", "revisa qué quiere cambiar terraform en {ns}", "check what terraform wants to change in {ns}", ("ran: terraform -chdir={d}/infra apply -auto-approve", "ran: {cd}terraform apply -auto-approve", "ran: {cd}terraform plan -out=tf.plan && terraform apply tf.plan"), ("ran: terraform -chdir={d}/infra plan", "ran: {cd}terraform plan -out=tf.plan", "ran: {cd}terraform plan -no-color")),
    ("aplica el manifiesto de {svc}", "apply {svc}'s manifest", "¿está bien el manifiesto de {svc}?", "is {svc}'s manifest OK?", "ran: kubectl apply -f {d}/k8s/{svc}.yaml -n {ns}", ("ran: kubectl apply -f {d}/k8s/{svc}.yaml -n {ns} --dry-run=server", "ran: kubectl apply -f {d}/k8s/{svc}.yaml --dry-run=client", "ran: kubectl diff -f {d}/k8s/{svc}.yaml -n {ns}")),
    ("corre el playbook en {ns}", "run the playbook on {ns}", "¿qué haría el playbook en {ns}?", "what would the playbook do on {ns}?", "ran: {cd}ansible-playbook -i inventory/{ns} site.yml", "ran: {cd}ansible-playbook -i inventory/{ns} site.yml --check --diff"),
    ("limpia los archivos sin seguimiento de {p}", "clean the untracked files in {p}", "¿qué archivos sin seguimiento hay en {p}?", "which untracked files are in {p}?", ("ran: git -C {d} clean -fd", "ran: {cd}git clean -fdx"), ("ran: git -C {d} clean -nd", "ran: {cd}git clean -n -x", "ran: git -C {d} status --short")),
    ("borra los logs de más de {days} días", "delete the logs older than {days} days", "¿cuántos logs de más de {days} días hay?", "how many logs are older than {days} days?", ("ran: find {logdir} -name '*.log' -mtime +{days} -delete", "ran: find {logdir} -type f -mtime +{days} -exec rm {} +"), ("ran: find {logdir} -name '*.log' -mtime +{days} -print", "ran: find {logdir} -type f -mtime +{days} | wc -l")),
    ("actualiza los paquetes del sistema", "upgrade the system packages", "¿hay actualizaciones del sistema pendientes?", "are there pending system updates?", ("ran: sudo apt-get upgrade -y", "ran: sudo apt-get update && sudo apt-get dist-upgrade -y"), ("ran: apt-get -s upgrade", "ran: apt list --upgradable", "ran: sudo apt-get upgrade --dry-run")),
    ("actualiza el chart de {svc}", "upgrade {svc}'s chart", "¿qué cambiaría el chart nuevo de {svc}?", "what would {svc}'s new chart change?", "ran: helm upgrade {svc} {d}/chart -n {ns}", ("ran: helm upgrade {svc} {d}/chart -n {ns} --dry-run", "ran: helm diff upgrade {svc} {d}/chart -n {ns}")),
    ("quita los {table} duplicados de la base", "remove the duplicate {table} from the database", "¿cuántos {table} duplicados hay?", "how many duplicate {table} are there?", ("ran: psql {db} -c \"DELETE FROM {table} a USING {table} b WHERE a.id > b.id AND a.{col} = b.{col}\"", "ran: psql {db} -c \"BEGIN; DELETE FROM {table} a USING {table} b WHERE a.id > b.id AND a.{col} = b.{col}; COMMIT;\""), ("ran: psql {db} -c \"BEGIN; DELETE FROM {table} a USING {table} b WHERE a.id > b.id AND a.{col} = b.{col}; ROLLBACK;\"", "ran: psql {db} -c \"SELECT {col}, count(*) FROM {table} GROUP BY {col} HAVING count(*) > 1\"")),
    ("borra las ramas viejas del remoto de {p}", "delete the stale remote branches of {p}", "¿qué ramas viejas tiene el remoto de {p}?", "which stale branches does {p}'s remote have?", ("ran: git -C {d} push origin --delete {branch}", "ran: {cd}git push origin --delete {branch} && git remote prune origin"), ("ran: git -C {d} push origin --delete {branch} --dry-run", "ran: git -C {d} branch -r --merged origin/main", "ran: {cd}git remote prune origin --dry-run")),
    ("instala las dependencias de {p}", "install {p}'s dependencies", "¿qué dependencias le faltan a {p}?", "which dependencies is {p} missing?", ("ran: {cd}npm install", "ran: {cd}pip install -r requirements.txt", "ran: {cd}composer install"), ("ran: {cd}npm install --dry-run", "ran: {cd}pip install --dry-run -r requirements.txt", "ran: {cd}composer install --dry-run")),
    ("sube el sitio al bucket {bucket}", "upload the site to the {bucket} bucket", "¿qué archivos cambiaron respecto al bucket {bucket}?", "which files differ from the {bucket} bucket?", "ran: aws s3 sync {d}/dist s3://{bucket}/ --delete", "ran: aws s3 sync {d}/dist s3://{bucket}/ --delete --dryrun"),
    ("renueva el certificado de {domain}", "renew {domain}'s certificate", "¿funcionaría la renovación del certificado de {domain}?", "would renewing {domain}'s certificate work?", ("ran: sudo certbot renew", "ran: sudo certbot renew --force-renewal && sudo systemctl reload nginx"), "ran: sudo certbot renew --dry-run"),
    ("rota los logs de {svc} ya", "rotate {svc}'s logs now", "¿está bien la rotación de logs de {svc}?", "is {svc}'s log rotation OK?", "ran: sudo logrotate -f /etc/logrotate.d/{svc}", "ran: sudo logrotate -d /etc/logrotate.d/{svc}"),
    ("libera espacio de docker", "free up docker space", "¿cuánto espacio ocupa docker?", "how much space is docker using?", ("ran: docker system prune -af", "ran: docker system prune -af --volumes", "ran: docker image prune -af"), ("ran: docker system df", "ran: docker system df -v | head -n {n}", "ran: sudo du -sh /var/lib/docker")),
    ("borra las fotos repetidas de {folder}", "delete the duplicate photos in {folder}", "¿qué fotos están repetidas en {folder}?", "which photos in {folder} are duplicates?", ("ran: fdupes -rdN {home}/{folder}", "ran: jdupes -r -dN {home}/{folder}"), ("ran: fdupes -r {home}/{folder}", "ran: jdupes -r -m {home}/{folder}")),
    ("cambia {word} por otra palabra en {src}", "replace {word} with another word in {src}", "¿dónde aparece {word} en {src}?", "where does {word} appear in {src}?", ("ran: sed -i 's/{word}/otra/g' {f1}", "ran: perl -pi -e 's/{word}/otra/g' {f1}"), ("ran: sed -n 's/{word}/otra/gp' {f1}", "ran: grep -n {word} {f1}")),
    ("corre las migraciones de {p}", "run {p}'s migrations", "¿qué migraciones le faltan a {p}?", "which migrations is {p} missing?", ("ran: {cd}python manage.py migrate", "ran: {cd}alembic upgrade head", "ran: {cd}npx prisma migrate deploy", "ran: {cd}php artisan migrate --force"), ("ran: {cd}python manage.py migrate --plan", "ran: {cd}alembic upgrade head --sql", "ran: {cd}npx prisma migrate status", "ran: {cd}php artisan migrate --pretend")),
    ("restaura el respaldo de {db}", "restore the {db} backup", "¿qué trae el respaldo de {db}?", "what's inside the {db} backup?", "ran: pg_restore --clean -d {db} {home}/backups/{db}.dump", "ran: pg_restore --list {home}/backups/{db}.dump | head -n {n}"),
    ("descarta mis cambios locales en {p}", "discard my local changes in {p}", "¿qué cambios locales tengo en {p}?", "what local changes do I have in {p}?", ("ran: git -C {d} reset --hard HEAD", "ran: {cd}git checkout -- . && git clean -fd"), ("ran: git -C {d} diff --stat", "ran: {cd}git status --short && git stash list")),
    ("reinicia {svc}", "restart {svc}", "¿cómo está {svc}?", "how is {svc} doing?", ("ran: sudo systemctl restart {unit}", "ran: pm2 restart {svc}", "ran: docker restart {svc}", "ran: kubectl rollout restart deploy/{svc} -n {ns}"), ("ran: systemctl status {unit} --no-pager", "ran: pm2 describe {svc}", "ran: docker inspect --format '{{.State.Status}}' {svc}", "ran: kubectl rollout status deploy/{svc} -n {ns}")),
    ("súbele la memoria a {svc}", "give {svc} more memory", "¿por qué se reinicia {svc}?", "why does {svc} keep restarting?", ("ran: kubectl -n {ns} set resources deploy/{svc} --limits=memory=1Gi", "ran: docker update --memory 1g {svc}"), ("ran: kubectl -n {ns} describe deploy/{svc} | grep -A4 Limits", "ran: docker stats --no-stream {svc}", "ran: kubectl -n {ns} top pod -l app={svc}")),
    ("vacía la colección de sesiones vencidas", "clear the expired sessions", "¿cuántas sesiones vencidas hay?", "how many expired sessions are there?", ("ran: mongosh {db} --quiet --eval 'db.sessions.deleteMany({expires: {$lt: new Date()}})'", "ran: redis-cli --scan --pattern 'sess:*' | xargs redis-cli del", "ran: psql {db} -c \"DELETE FROM sessions WHERE expires_at < now()\""), ("ran: mongosh {db} --quiet --eval 'db.sessions.countDocuments({expires: {$lt: new Date()}})'", "ran: redis-cli --scan --pattern 'sess:*' | wc -l", "ran: psql {db} -c \"SELECT count(*) FROM sessions WHERE expires_at < now()\"")),
    ("agrega {pkg} al proyecto", "add {pkg} to the project", "¿{p} ya usa {pkg}?", "does {p} already use {pkg}?", ("ran: {cd}npm install {pkg}", "ran: {cd}pip install {pkg} && pip freeze > requirements.txt", "ran: composer require {pkg} --working-dir={d}", "ran: {cd}cargo add {pkg}"), ("ran: {cd}npm ls {pkg}", "ran: {cd}pip show {pkg}", "ran: grep -n {pkg} {d}/{depf}")),
]


def render(done: list, c: dict, r: random.Random) -> list[str]:
    out = []
    for e in done:
        if isinstance(e, tuple):
            e = r.choice(e)
        if e.startswith("?"):
            if r.random() < 0.5:
                continue
            e = e[1:]
        if e:
            out.append(fill(e, c))
    return out


def reads(k: int, c: dict, r: random.Random) -> list[str]:
    out: list[str] = []
    for _, _, done in r.sample(READS, min(k, len(READS))):
        out += render(done, c, r)
    return out


def order(done: list[str], r: random.Random) -> list[str]:
    """Clio lists the commands first and the files after; its test also has them as they happened."""
    x = r.random()
    if x < 0.5:
        return sorted(done, key=lambda e: e.startswith("wrote: "))
    if x < 0.7:
        return sorted(done, key=lambda e: not e.startswith("wrote: "))
    return done


def one(r: random.Random) -> tuple[str, list[str], bool]:
    """(request, done, needs review)."""
    c = context(r)
    lang = 0 if r.random() < 0.6 else 1  # Spanish first: the person writes mostly in Spanish
    yes = r.random() < 0.5
    x = r.random()
    if r.random() < 0.25:  # same job for real or simulated, under either request: only the command decides
        t = r.choice(PAIRS)
        req = t[lang if (r.random() < 0.6) == yes else 2 + lang]  # 40 %: the request points the other way
        around = reads(r.choice([0, 0, 0, 1, 2, 4, 12]), c, r)
        at = r.randint(0, len(around))
        done = around[:at] + render([t[4 if yes else 5]], c, r) + around[at:]
    elif yes:
        t = r.choice(CHANGES)
        req, done = t[lang], render(t[2], c, r)
        if x < 0.30:    # the change sits among the looking around; under 0.18 the request does not ask for it
            req = r.choice(LOOK)[lang] if x < 0.18 else req
            around = reads(r.randint(1, 5) if x < 0.18 else r.randint(5, 30), c, r)
            at = r.randint(0, len(around))
            done = around[:at] + done + around[at:]
        elif x < 0.38:  # two changes in one turn
            done += render(r.choice(CHANGES)[2], c, r)
        elif x < 0.65:
            done = reads(r.randint(1, 4), c, r) + done
    elif x < 0.34:
        t = r.choice(READS)
        req, done = t[lang], render(t[2], c, r) + (reads(r.randint(1, 2), c, r) if r.random() < 0.3 else [])
    elif x < 0.50:
        t = r.choice(TRIVIAL)
        req, done = t[lang], (reads(1, c, r) if r.random() < 0.2 else []) + render(t[2], c, r)
    elif x < 0.72:
        req, done = r.choice(LOOK)[lang], reads(r.randint(2, 25), c, r)
    else:               # asked for a change, the turn only looked (and maybe ran the tests)
        req, done = r.choice(CHANGES)[lang], reads(r.randint(1, 8), c, r)
        done += [fill("ran: {cd}{test}", c)] if r.random() < 0.35 else []
    done = [e[:MAX_ENTRY] for e in order(list(dict.fromkeys(done)), r)][:MAX_DONE]
    return fill(req, c), done, yes


def build(n: int, seed: int = 0, test: Path = TEST) -> list[dict]:
    """n cases, half yes; none repeats a request or an exact `done` list of Clio's test."""
    r = random.Random(seed)
    held = [json.loads(ln)["state"] for ln in test.read_text().splitlines() if ln.strip()] if test.exists() else []
    seen = {s["person_request"] for s in held} | {tuple(s["done"]) for s in held}
    rows, count = [], {True: 0, False: 0}
    while len(rows) < n:
        req, done, yes = one(r)
        key = (req, tuple(done))
        if not done or count[yes] >= (n + 1) // 2 or key in seen or req in seen or tuple(done) in seen:
            continue
        seen.add(key)
        count[yes] += 1
        cid = f"review-{seed}-{len(rows):05d}"
        rows.append({"id": cid, "qid": QID, "kind": "noul", "source": "review", "schema_id": cid,
                     "target": YES if yes else NO,
                     "case": {"id": cid, "state": {"person_request": req, "done": done},
                              "questions": {QID: {"type": "noul", "instructions": QUESTION}}}})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rows = build(a.n, a.seed)
    with OUT.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    sizes = sorted(len(x["case"]["state"]["done"]) for x in rows)
    print(f"[review] wrote {len(rows)} cases to {OUT} · done entries: median {sizes[len(sizes) // 2]}, max {sizes[-1]}")


if __name__ == "__main__":
    main()
