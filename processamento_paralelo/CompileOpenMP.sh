#!/bin/bash

compile() {
    echo "Compiling OpenMP implementation...\n"
    gcc -o bin/openMP -Wall -O3 -fopenmp src/openMP_implementation.c src/config.c src/tensor.c -Ilib
}

run() {
    echo "Running OpenMP implementation...\n"
    ./bin/openMP "$1" "$2" "$3"
}

main() {
    if [ "$#" -lt 1 ]; then
        echo "Bad arguments: $@"
        exit
    fi

    if [ "$1" = "compile" ]; then
        compile "$2"
    elif [ "$1" = "run" ]; then
        input_file="$2"
        threshold="$3"
        number_of_threads="$4"

        run "$input_file" "$threshold" "$number_of_threads"
    else
        echo "Unknown arguments: $@"
    fi
}

main $@
# MELHOR VALOR DE THREADS: 4
# COMO USAR: sh CompileOpenMP.sh run input3.txt 0.5 4

