#!/usr/bin/env bash
# Deploy the Research Assistant stack to Fly.io.
#
# Prerequisites:
#   - flyctl installed and authenticated (`flyctl auth login`)
#   - Secrets already set via `flyctl secrets set ...` (see fly.toml)
#   - Postgres app attached (`flyctl postgres attach` writes DATABASE_URL)
#   - Redis app created (`flyctl redis create` writes REDIS_URL)

set -euo pipefail

cd "$(dirname "$0")/.."

echo "==> Deploying API service"
flyctl deploy --config fly.toml --remote-only --wait-timeout 300

echo "==> Deploying bot worker"
flyctl deploy --config fly.bot.toml --remote-only --wait-timeout 300

echo
echo "Done. URLs:"
flyctl status --config fly.toml | grep -E "Hostname|Status"
echo
echo "Check logs:   flyctl logs -c fly.toml"
echo "Check secrets: flyctl secrets list -c fly.toml"
