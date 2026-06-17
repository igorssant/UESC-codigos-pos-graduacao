#include <ctime>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

EXPORT __device__ void get_indices_with_padding(int coordinates[6], const int i, const int j, const int k) {
    // minimo
    atomicMin(&coordinates[0], i);
    atomicMin(&coordinates[1], j);
    atomicMin(&coordinates[2], k);
    // maximo
    atomicMax(&coordinates[3], i);
    atomicMax(&coordinates[4], j);
    atomicMax(&coordinates[5], k);
    return;
}

EXPORT __global__ void binarize_data(
    double *data_ptr,
    const int x_axis,
    const int y_axis,
    const int z_axis,
    const double THRESHOLD,
    int coordinates[6]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x,
        j = blockIdx.y * blockDim.y + threadIdx.y,
        k = blockIdx.z * blockDim.z + threadIdx.z;

    // garante os limites da matriz
    if(i < x_axis && j < y_axis && k < z_axis) {
        int index = (i * y_axis * z_axis) + (j * z_axis) + k;

        if(data_ptr[index] > THRESHOLD) {
            data_ptr[index] = 1.0;
            // ja pega os limites diretamente aqui
            get_indices_with_padding(coordinates, i, j, k);
        } else {
            data_ptr[index] = 0.0;
        }
    }

    return;
}

EXPORT __global__ void apply_padding_and_dimensions(
    int coordinates[6],
    const int x_axis,
    const int y_axis,
    const int z_axis,
    int *output_dimensions
) {
    if(blockIdx.x == 0 && threadIdx.x == 0) {
        if(coordinates[3] < coordinates[0]) {
            // retornando 0 como erro
            output_dimensions[0] = 0;
            return;
        }

        // aplicando o padding
        coordinates[0] = (coordinates[0] > 5) ? coordinates[0] - 5 : 0;
        coordinates[1] = (coordinates[1] > 5) ? coordinates[1] - 5 : 0;
        coordinates[2] = (coordinates[2] > 5) ? coordinates[2] - 5 : 0;
        coordinates[3] = (coordinates[3] + 5 < x_axis) ? coordinates[3] + 5 : x_axis - 1;
        coordinates[4] = (coordinates[4] + 5 < y_axis) ? coordinates[4] + 5 : y_axis - 1;
        coordinates[5] = (coordinates[5] + 5 < z_axis) ? coordinates[5] + 5 : z_axis - 1;

        // calculando as dimensoes finais
        output_dimensions[0] = (coordinates[3] - coordinates[0] + 1);
        output_dimensions[1] = (coordinates[4] - coordinates[1] + 1);
        output_dimensions[2] = (coordinates[5] - coordinates[2] + 1);
    }

    return;
}

EXPORT __global__ void crop_data(
    double *data_ptr,
    const int x_axis,
    const int y_axis,
    const int z_axis,
    const int coordinates[6],
    double *output_data_ptr,
    int x_output_axis,
    int y_output_axis,
    int z_output_axis
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x,
        j = blockIdx.y * blockDim.y + threadIdx.y,
        k = blockIdx.z * blockDim.z + threadIdx.z;

    // garante os limites da nova matriz recortada
    if(i < x_output_axis && j < y_output_axis && k < z_output_axis) {
        // Mapeia onde esse elemento estava na matriz original usando os offsets
        int input_i = i + coordinates[0],
            input_j = j + coordinates[1],
            input_k = k + coordinates[2],
            // indice da entrada
            input_index = (input_i * y_axis * z_axis) + (input_j * z_axis) + input_k,
            // indice da saida
            output_index = (i * y_output_axis * z_output_axis) + (j * z_output_axis) + k;

        // copia o dado diretamente via registradores da GPU
        output_data_ptr[output_index] = data_ptr[input_index];
    }

    return;
}

EXPORT __host__ void bounding_box_pipeline(
    double *host_input_data_ptr,
    const int x_axis,
    const int y_axis,
    const int z_axis,
    const double THRESHOLD,
    double *host_output_data_ptr
) {
    double *device_input_data = NULL,
        *device_output_data = NULL;
    int host_coordinates[6] = {x_axis, y_axis, z_axis, -1, -1, -1},
        *device_coordinates = NULL,
        host_output_dimensions[3] = {0, 0, 0},
        *device_output_dimensions = NULL,
        x_output_axis,
        y_output_axis,
        z_output_axis;
    // dimensoes da GPU | loucuras de CUDA
    dim3 threads_in_block(8, 8, 8),
        number_of_blocks((x_axis + threads_in_block.x - 1) / threads_in_block.x,
                         (y_axis + threads_in_block.y - 1) / threads_in_block.y,
                         (z_axis + threads_in_block.z - 1) / threads_in_block.z);
    // calculo dos tempos
    time_t start,
        total_time = clock();

    // alocando espaco CUDA
    cudaMalloc((void**) &device_input_data, x_axis * y_axis * z_axis * sizeof(double));
    cudaMalloc((void**) &device_coordinates, 6 * sizeof(int));
    cudaMalloc((void**) &device_output_dimensions, 3 * sizeof(int));
    // copia RAM -> VRAM
    cudaMemcpy(device_input_data, host_input_data_ptr, x_axis * y_axis * z_axis * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(device_coordinates, host_coordinates, 6 * sizeof(int), cudaMemcpyHostToDevice);

    start = clock();
    // chamando a funcao na GPU
    binarize_data<<<number_of_blocks, threads_in_block>>>(device_input_data, x_axis, y_axis, z_axis, THRESHOLD, device_coordinates);
    cudaDeviceSynchronize();
    printf("Tempo (binarize_data): %.3lf\n", (double) (clock() - start) / CLOCKS_PER_SEC);
    fflush(stdout);
    start = clock();
    apply_padding_and_dimensions<<<1, 1>>>(device_coordinates, x_axis, y_axis, z_axis, device_output_dimensions);
    printf("Tempo (apply_padding_and_dimensions): %.3lf\n", (double) (clock() - start) / CLOCKS_PER_SEC);
    fflush(stdout);
    // copia VRAM -> RAM
    cudaMemcpy(host_output_dimensions, device_output_dimensions, 3 * sizeof(int), cudaMemcpyDeviceToHost);

    // verificacao de erro no bounding box
    if(!host_output_dimensions[0]) {
        printf("Error!\nNo data was found.\n");
        fflush(stdout);
        cudaFree(device_input_data);
        cudaFree(device_coordinates);
        cudaFree(device_output_dimensions);
        return;
    }

    x_output_axis = host_output_dimensions[0];
    y_output_axis = host_output_dimensions[1];
    z_output_axis = host_output_dimensions[2];
    cudaMalloc((void**) &device_output_data, x_output_axis * y_output_axis * z_output_axis * sizeof(double));

    dim3 number_of_blocks_crop((x_output_axis + threads_in_block.x - 1) / threads_in_block.x,
                               (y_output_axis + threads_in_block.y - 1) / threads_in_block.y,
                               (z_output_axis + threads_in_block.z - 1) / threads_in_block.z);

    start = clock();
    crop_data<<<number_of_blocks_crop, threads_in_block>>>(
        device_input_data,
        x_axis,
        y_axis,
        z_axis,
        device_coordinates,
        device_output_data,
        x_output_axis,
        y_output_axis,
        z_output_axis
    );
    cudaDeviceSynchronize();
    printf("Tempo (crop_data): %.3lf\n", (double) (clock() - start) / CLOCKS_PER_SEC);
    fflush(stdout);

    // resultado final | copia VRAM -> RAM
    cudaMemcpy(
        host_output_data_ptr,
        device_output_data,
        x_output_axis * y_output_axis * z_output_axis * sizeof(double),
        cudaMemcpyDeviceToHost
    );
    printf("Tempo total (C): %.3lf\n", (double) (clock() - total_time) / CLOCKS_PER_SEC);
    fflush(stdout);

    // desalocando tudo
    cudaFree(device_input_data);
    cudaFree(device_output_data);
    cudaFree(device_coordinates);
    cudaFree(device_output_dimensions);
    return;
}

#ifdef __cplusplus
}
#endif
