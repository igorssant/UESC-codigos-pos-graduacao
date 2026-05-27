#!/bin/bash

set -e

compile() {
    echo "Compiling MPI implementation...\n"
    mpicc -o bin/MPI -Wall -O3 src/mpi_implementation.c src/config.c src/tensor.c -Ilib
}

run() {
    echo "Running MPI implementation...\n"
    mpirun -np "$4" ./bin/MPI "$1" "$2" "$3"
}

main() {
    if [ "$#" -lt 1 ]; then
        echo "Bad arguments: $@"
        exit 1
    fi

    if [ "$1" = "compile" ]; then
        compile
    elif [ "$1" = "run" ]; then
        input_file="$2"
        threshold="$3"
        number_of_processes="$4"

        run "$input_file" "$threshold" "$number_of_processes"
    else
        echo "Unknown arguments: $@"
    fi
}

main "$@"
# COMO USAR: sh CompileOpenMP.sh run input3.txt 0.5 4
