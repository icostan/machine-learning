#!/usr/bin/env sh

PROJECT=home-credit-default-risk
SRC=icostan/datasets/$PROJECT/2
DST=$PROJECT
CMD="ln -sf /$PROJECT ./input && python $PROJECT.py"

echo "Running: floyd run --data $SRC:$DST \"$CMD\" ..."
floyd run --data $SRC:$DST "$CMD"
