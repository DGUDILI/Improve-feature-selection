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
    CLEAN=${4:-}

    # 3번째 인자가 "clean"이면 GA 없이 clean으로 처리
    if [ "$GA" = "clean" ]; then
      CLEAN="clean"
      GA=""
    fi

    GA_ARG=""
    CLEAN_ARG=""
    if [ -n "$GA" ]; then
      GA_ARG="--ga=$GA"
    fi
    if [ "$CLEAN" = "clean" ]; then
      CLEAN_ARG="--clean"
    fi
    docker compose run --rm ml \
      conda run -n dili_ml_pipeline_env \
      python src/train.py --stacking="$STACKING" $GA_ARG $CLEAN_ARG
    ;;
  env-test)
    docker compose run --rm ml \
      conda run -n dili_ml_pipeline_env \
      python src/env_test.py
    ;;
  *)
    echo "Usage: ./run.sh {build|shell|run [stacking_version] [ga_version] [clean]|env-test}"
    echo "  stacking: v0 | v0.5 | v1       (기본값: v1)"
    echo "  ga:       v0 | v1 | v4 | v5   (생략 가능)"
    echo "  clean:    'clean' 입력 시 Train-Test 중복 제거 실행"
    echo ""
    echo "  예시: ./run.sh run                    # GA 없이 stacking v1"
    echo "        ./run.sh run v1                 # GA 없이 stacking v1"
    echo "        ./run.sh run v1 v4              # GA v4 + stacking v1"
    echo "        ./run.sh run v1 v4 clean        # GA v4 + stacking v1 + 데이터 정제"
    echo "        ./run.sh run v1 \"\" clean        # GA 없이 stacking v1 + 데이터 정제"
    exit 1
    ;;
esac
