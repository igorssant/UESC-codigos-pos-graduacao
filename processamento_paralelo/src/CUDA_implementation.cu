#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

EXPORT __device__ void binarize_data(double *data_ptr, const int x_axis, const int y_axis, const int z_axis, const double THRESHOLD) {
    int i = blockIdx.x * blockDim.x + threadIdx.x,
        j = blockIdx.y * blockDim.y + threadIdx.y,
        k = blockIdx.z * blockDim.z + threadIdx.z;

    // garante os limites da matriz
    if(i < x_axis && j < y_axis && k < z_axis) {
        int index = (i * y_axis * z_axis) + (j * z_axis) + k;

        data_ptr[index] = (data_ptr[index] > THRESHOLD) ? 1.0 : 0.0;
    }

    return;
}

EXPORT __device__ void get_indices_with_padding(
    const double *data_ptr,
    const int x_axis,
    const int y_axis,
    const int z_axis,
    int coordinates[][3]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x,
        j = blockIdx.y * blockDim.y + threadIdx.y,
        k = blockIdx.z * blockDim.z + threadIdx.z,
        index = (i * y_axis * z_axis) + (j * z_axis) + k;

    if(data_ptr[index]) {
        // minimo
        atomicMin(coordinates[0][0], i);
        atomicMin(coordinates[0][1], j);
        atomicMin(coordinates[0][2], k);
        // maximo
        atomicMax(coordinates[1][0], i);
        atomicMax(coordinates[1][1], j);
        atomicMax(coordinates[1][2], k);
    }

    return;
}

EXPORT __global__ void crop_data(
    double *data_ptr,
    const int x_axis,
    const int y_axis,
    const int z_axis,
    const double THRESHOLD,
    double *output_data_ptr
) {
    int coordinates[2][3],
        i,
        j,
        k,
        x_output_axis,
        y_output_axis,
        z_output_axis;

    binarize_data(data_ptr, x_axis, y_axis, z_axis, THRESHOLD);
    get_indices_with_padding(data_ptr, x_axis, y_axis, z_axis, coordinates);

    if(coordinates[1][0] < coordinates[0][0]) {
        printf("Error: No data was found.\n");
        return;
    }

    x_output_axis = (coordinates[1][0] - coordinates[0][0] + 1);
    y_output_axis = (coordinates[1][1] - coordinates[0][1] + 1);
    z_output_axis = (coordinates[1][2] - coordinates[0][2] + 1);
    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;
    k = blockIdx.z * blockDim.z + threadIdx.z;

    // garante os limites da nova matriz recortada
    if(i < x_output_axis && j < y_output_axis && k < y_output_axis) {
        // Mapeia onde esse elemento estava na matriz original usando os offsets
        int input_i = i + coordinates[0][0],
            input_j = j + coordinates[0][1],
            input_k = k + coordinates[0][2],
            // indice da entrada
            input_index = (input_i * y_axis * z_axis) + (input_j * z_axis) + input_k,
            // indice da saida
            output_index = (i * y_output_axis * z_output_axis) + (j * z_output_axis) + k;

        // copia o dado diretamente via registradores da GPU
        output_data_ptr[output_index] = data_ptr[input_index];
    }

    return;
}

