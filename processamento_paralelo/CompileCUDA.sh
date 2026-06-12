#!/bin/bash

set -e

compile() {
    echo "Compiling CUDA implementation...\n"
    nvcc -O3 -use_fast_math -arch=sm_90 -Xcompiler -fPIC -shared -o bin/cuda_c.so -O3 src/CUDA_implementation.cu
}

#run() {
#    echo "Running CUDA implementation...\n"
#    ./bin/CUDA "$1" "$2"
#}

main() {
    if [ "$#" -lt 1 ]; then
        echo "Bad arguments: $@"
        exit
    fi

    if [ "$1" = "compile" ]; then
        compile "$2"
    #elif [ "$1" = "run" ]; then
    #    input_file="$2"
    #    threshold="$3"
    #
    #    run "$input_file" "$threshold"
    else
        echo "Unknown arguments: $@"
    fi
}

main $@

