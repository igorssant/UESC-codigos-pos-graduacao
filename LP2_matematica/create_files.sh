#!/bin/bash

set -e

cd src/

for i in {1..30}; do
    touch questao_"$i".c
done
