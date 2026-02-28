#!/bin/bash
set -e

if [ "$SERVE_MODE" = "ray" ]; then
    echo "Starting in Ray Serve mode..."
    exec python3 -c "
import ray
from ray import serve

ray.init()
serve.start(http_options={'host': '0.0.0.0', 'port': 9000})

from app.serve_app import app
serve.run(app, blocking=True)
"
else
    echo "Starting in simple mode (uvicorn)..."
    exec python3 -m uvicorn app.main:app --host 0.0.0.0 --port 9000
fi
