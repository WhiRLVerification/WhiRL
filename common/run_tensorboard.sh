#!/bin/bash


if [[ ! -z "$1" ]]; then
    python3 -m tensorboard.main --logdir=$1 --host localhost --port 62212

else
    echo "Usages:"
    echo "    $0 <logdir>"
fi
