#!/bin/bash

set -eo pipefail
shopt -s nullglob

# this is used by other processes, so needs to be exported
export OVERPASS_MAX_TIMEOUT=${OVERPASS_MAX_TIMEOUT:-1000s}

echo "custom overpass eval"

# shellcheck disable=SC2016 # ignore SC2016 (variables within single quotes) as this is exactly what we want to do here
envsubst '${OVERPASS_MAX_TIMEOUT}' </etc/nginx/nginx.conf.template >/etc/nginx/nginx.conf

echo "Starting supervisord process"
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/supervisord.conf