#!/usr/bin/env bash

CMD=${1}

case "$CMD" in
  build)
    docker compose build
    ;;
  shell)
    docker compose run --rm ml bash
    ;;
  run)
    STACKING=${2:-v1}
    GA=${3:-}
    GA_ARG=""
    if [ -n "$GA" ]; then
      GA_ARG="--ga=$GA"
    fi
    docker compose run --rm ml \
      conda run -n dili_ml_pipeline_env \
      python src/train.py --stacking="$STACKING" $GA_ARG
    ;;
  env-test)
    docker compose run --rm ml \
      conda run -n dili_ml_pipeline_env \
      python src/env_test.py
    ;;
  *)
    echo "Usage: ./run.sh {build|shell|run [stacking_version] [ga_version]|env-test}"
    echo "  예시: ./run.sh run          # GA 없이 stacking v1"
    echo "        ./run.sh run v1       # GA 없이 stacking v1"
    echo "        ./run.sh run v1 v1    # GA v1 + stacking v1"
    exit 1
    ;;
esac
