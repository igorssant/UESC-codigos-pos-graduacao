#!/bin/bash

set -e


if [ -z "$1" ]; then
    exit 1
fi

src="src/$1"
target="bin/$2"
which_lib="lib/$3/funcoes.c"

gcc -Wall -Wextra "$src" "$which_lib" -o "$target" -lm -Ilib
