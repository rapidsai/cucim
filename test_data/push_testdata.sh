#!/bin/bash

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

docker build -t gigony/svs-testdata:little-big ${SCRIPT_DIR}
docker push gigony/svs-testdata:little-big
