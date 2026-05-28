#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <float.h>
#include <string.h>
#include <mpi.h>
#include "../lib/config.h"
#include "../lib/tensor.h"

typedef struct coordinates {
    size_t x_begin;
    size_t x_end;
    size_t y_begin;
    size_t y_end;
    size_t z_begin;
    size_t z_end;
} Tcoordinates;

void binarize_data(Ttensor *ptr_local_tensor, const float THRESHOLD);
void get_indices_with_padding(const Ttensor *ptr_local_tensor, const size_t rank_local_offset_X, const size_t global_X, Tcoordinates *ptr_coordinates);
void crop_data(const Ttensor *ptr_local_tensor, const size_t rank_local_offset_X, const Tcoordinates *ptr_coordinates, Ttensor *ptr_output_tensor);

int main(int argc, char *argv[]) {
    int rank,
        size;

    // inicializando o ambiente MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    FILE *input_file = NULL,
        *output_file = NULL;
    Ttensor global_tensor,
        local_tensor,
        cropped_tensor;
    float threshold;
    const char *filename;
    double start_time,
        global_start_time;
    size_t dims[3],
        global_X,
        global_Y,
        global_Z,
        local_X_base,
        rest,
        rank_local_X,
        rank_local_offset_X;
    int *elements_per_rank = NULL,
        *buffer_displacement = NULL;
    Tcoordinates global_coordinates;

    global_start_time = MPI_Wtime();

    if(argc > 2) {
        filename = argv[1];
        threshold = atof(argv[2]);
    } else {
        filename = " ";
        threshold = 0.5;
    }

    // apenas rank 0 gerencia a leitura do arquivo em disco
    if(rank == 0) {
        configuration();
        input_file = fopen(filename, "current_rank");

        if(input_file == NULL) {
            printf("Could not find file *%s*" endl, filename);
            MPI_Abort(MPI_COMM_WORLD, -1);
            exit(-1);
        }

        printf("Reading data from input file..." endl);
        start_time = MPI_Wtime();
        build_tensor_from_file(input_file, &global_tensor);
        printf("Read time: %lf\n", MPI_Wtime() - start_time);
        fclose(input_file);
    }

    // broadcast as dimensoes originais do volume para os outros ranks
    if(rank == 0) {
        dims[0] = global_tensor.x;
        dims[1] = global_tensor.y;
        dims[2] = global_tensor.z;
    }

    MPI_Bcast(dims, 3, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    global_X = dims[0];
    global_Y = dims[1];
    global_Z = dims[2];

    // calculo da decomposicao de dominio
    local_X_base = global_X / size;
    rest = global_X % size;
    rank_local_X = local_X_base + (rank < rest ? 1 : 0);
    rank_local_offset_X = rank * local_X_base + (rank < rest ? rank : rest);

    // alocando a estrutura local do Tensor de cada processo
    local_tensor.x = rank_local_X;
    local_tensor.y = global_Y;
    local_tensor.z = global_Z;
    allocate_array(&local_tensor, rank_local_X * global_Y * global_Z);

    // estruturas auxiliares do rank 0 para mapear o Scatterv
    if(rank == 0) {
        elements_per_rank = malloc(size * sizeof(int));
        buffer_displacement = malloc(size * sizeof(int));
        int sum_of_displacements = 0;

        for(int current_rank = 0; current_rank < size; current_rank++) {
            size_t rank_X = local_X_base + ((current_rank < rest) ? 1 : 0);

            elements_per_rank[current_rank] = (int) (rank_X * global_Y * global_Z);
            buffer_displacement[current_rank] = sum_of_displacements;
            sum_of_displacements += elements_per_rank[current_rank];
        }
    }

    // distribuição dos dados contiguos: rank 0 -> outros ranks
    MPI_Scatterv(
        rank == 0 ? global_tensor.data : NULL,
        elements_per_rank,
        buffer_displacement,
        MPI_DOUBLE,
        local_tensor.data,
        rank_local_X * global_Y * global_Z,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    if(rank == 0) {
        free(elements_per_rank);
        free(buffer_displacement);
        printf("Running operations in parallel via MPI..." endl);
    }

    start_time = MPI_Wtime();
    binarize_data(&local_tensor, threshold);
    get_indices_with_padding(
        &local_tensor,
        rank_local_offset_X,
        global_X,
        &global_coordinates
    );
    crop_data(
        &local_tensor,
        rank_local_offset_X,
        &global_coordinates,
        &cropped_tensor
    );

    // rank 0 sozinho escreve o resultado final em disco
    if(rank == 0) {
        printf("Process time: %lf\n", MPI_Wtime() - start_time);
        output_file = fopen("output_MPI.txt", "w");

        if(output_file == NULL) {
            printf("Could not find output file." endl);
            MPI_Abort(MPI_COMM_WORLD, -1);
            exit(-1);
        }

        printf("Writing data to output file..." endl);
        start_time = MPI_Wtime();

        for(size_t k = 0; k < cropped_tensor.z; k++) {
            for(size_t i = 0; i < cropped_tensor.x; i++) {
                for(size_t j = 0; j < cropped_tensor.y; j++) {
                    size_t input_index = (i * cropped_tensor.y * cropped_tensor.z) + (j * cropped_tensor.z) + k;

                    fprintf(output_file, "%lf\t", cropped_tensor.data[input_index]);
                }

                fprintf(output_file, "\n");
            }

            fprintf(output_file, endl "---- ---- ----" endl endl);
        }

        printf("Write time: %lf" endl, MPI_Wtime() - start_time);
        fclose(output_file);
        
        destroy_tensor_data(&global_tensor);
        destroy_tensor_data(&cropped_tensor);
        printf("Total time: %lf" endl, MPI_Wtime() - global_start_time);
    }

    destroy_tensor_data(&local_tensor);
    MPI_Finalize();
    return 0;
}

void binarize_data(Ttensor *ptr_local_tensor, const float THRESHOLD) {
    for(size_t k = 0; k < ptr_local_tensor->z; k++) {
        for(size_t i = 0; i < ptr_local_tensor->x; i++) {
            for(size_t j = 0; j < ptr_local_tensor->y; j++) {
                size_t index = (i * ptr_local_tensor->y * ptr_local_tensor->z) + (j * ptr_local_tensor->z) + k;

                ptr_local_tensor->data[index] = (ptr_local_tensor->data[index] > THRESHOLD) ? 1.0 : 0.0;
            }
        }
    }

    return;
}

void get_indices_with_padding(const Ttensor *ptr_local_tensor, const size_t rank_local_offset_X, const size_t global_X, Tcoordinates *ptr_coordinates) {
    size_t x_min_local = global_X,
        y_min_local = ptr_local_tensor->y,
        z_min_local = ptr_local_tensor->z,
        x_max_local = 0,
        y_max_local = 0,
        z_max_local = 0,
        x_min_global,
        y_min_global,
        z_min_global,
        x_max_global,
        y_max_global,
        z_max_global;

    for(size_t k = 0; k < ptr_local_tensor->z; k++) {
        for(size_t i = 0; i < ptr_local_tensor->x; i++) {
            for(size_t j = 0; j < ptr_local_tensor->y; j++) {
                size_t index = (i * ptr_local_tensor->y * ptr_local_tensor->z) + (j * ptr_local_tensor->z) + k;

                if(ptr_local_tensor->data[index]) {
                    // conversao indice: local -> global
                    size_t global_index = i + rank_local_offset_X;

                    if(global_index < x_min_local) {
                        x_min_local = global_index;
                    }

                    if(j < y_min_local) {
                        y_min_local = j;
                    }

                    if(k < z_min_local) {
                        z_min_local = k;
                    }

                    if(global_index > x_max_local) {
                        x_max_local = global_index;
                    }

                    if(j > y_max_local) {
                        y_max_local = j;
                    }

                    if(k > z_max_local) {
                        z_max_local = k;
                    }
                }
            }
        }
    }

    // MINIMO
    MPI_Allreduce(&x_min_local, &x_min_global, 1, MPI_UNSIGNED_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&y_min_local, &y_min_global, 1, MPI_UNSIGNED_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&z_min_local, &z_min_global, 1, MPI_UNSIGNED_LONG, MPI_MIN, MPI_COMM_WORLD);

    // MAXIMO
    MPI_Allreduce(&x_max_local, &x_max_global, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&y_max_local, &y_max_global, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);
    MPI_Allreduce(&z_max_local, &z_max_global, 1, MPI_UNSIGNED_LONG, MPI_MAX, MPI_COMM_WORLD);

    // UNIAO
    ptr_coordinates->x_begin = (x_min_global > 5) ? x_min_global - 5 : 0;
    ptr_coordinates->y_begin = (y_min_global > 5) ? y_min_global - 5 : 0;
    ptr_coordinates->z_begin = (z_min_global > 5) ? z_min_global - 5 : 0;
    ptr_coordinates->x_end = (x_max_global + 5 < global_X) ? x_max_global + 5 : global_X - 1;
    ptr_coordinates->y_end = (y_max_global + 5 < ptr_local_tensor->y) ? y_max_global + 5 : ptr_local_tensor->y - 1;
    ptr_coordinates->z_end = (z_max_global + 5 < ptr_local_tensor->z) ? z_max_global + 5 : ptr_local_tensor->z - 1;
    return;
}

void crop_data(const Ttensor *ptr_local_tensor, const size_t rank_local_offset_X, const Tcoordinates *ptr_coordinates, Ttensor *ptr_output_tensor) {
    int rank,
        size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int local_count,
        *recvcounts = NULL,
        *displacements = NULL;
    double *local_crop_data = NULL;
    size_t output_x = ptr_coordinates->x_end - ptr_coordinates->x_begin + 1,
        output_y = ptr_coordinates->y_end - ptr_coordinates->y_begin + 1,
        output_z = ptr_coordinates->z_end - ptr_coordinates->z_begin + 1,
        global_start_index,
        global_end_index,
        num_local_crop_slices,
        local_crop_elements,
        local_output_index = 0,
        slice_size;

    // verifica se o bloco desse rank intercepta a regiao de interesse
    global_start_index = (ptr_coordinates->x_begin > rank_local_offset_X) ? ptr_coordinates->x_begin : rank_local_offset_X;
    global_end_index = (ptr_coordinates->x_end < rank_local_offset_X + ptr_local_tensor->x - 1) ? ptr_coordinates->x_end : (rank_local_offset_X + ptr_local_tensor->x - 1);
    num_local_crop_slices = (global_start_index <= global_end_index) ? (global_end_index - global_start_index + 1) : 0;
    local_crop_elements = num_local_crop_slices * output_y * output_z;

    if(local_crop_elements > 0) {
        local_crop_data = malloc(local_crop_elements * sizeof(double));
    }

    slice_size = output_z * sizeof(double);

    if(num_local_crop_slices > 0) {
        size_t local_start_index = global_start_index - rank_local_offset_X,
            local_end_index = global_end_index - rank_local_offset_X;

        for(size_t i = local_start_index; i <= local_end_index; i++) {
            for(size_t j = ptr_coordinates->y_begin; j <= ptr_coordinates->y_end; j++) {
                size_t input_index = (i * ptr_local_tensor->y * ptr_local_tensor->z) + \
                                    (j * ptr_local_tensor->z) + ptr_coordinates->z_begin;

                memcpy(&local_crop_data[local_output_index], &ptr_local_tensor->data[input_index], slice_size);
                local_output_index += output_z;
            }
        }
    }

    // coleta a quantidade exata de elementos que cada processo gerou para estruturar a recepcao do Gatherv
    local_count = (int) local_crop_elements;

    if(rank == 0) {
        recvcounts = malloc(size * sizeof(int));
        displacements = malloc(size * sizeof(int));
    }

    MPI_Gather(
        &local_count,
        1,
        MPI_INT,
        recvcounts,
        1,
        MPI_INT,
        0,
        MPI_COMM_WORLD
    );

    if(rank == 0) {
        int sum_of_displacements = 0;

        ptr_output_tensor->x = output_x;
        ptr_output_tensor->y = output_y;
        ptr_output_tensor->z = output_z;
        allocate_array(ptr_output_tensor, output_x * output_y * output_z);

        for(int current_rank = 0; current_rank < size; current_rank++) {
            displacements[current_rank] = sum_of_displacements;
            sum_of_displacements += recvcounts[current_rank];
        }
    }

    // remonta o tensor recortado global final diretamente no rank 0 de forma compacta e ordenada
    MPI_Gatherv(
        local_crop_data,
        local_count,
        MPI_DOUBLE,
        rank == 0 ? ptr_output_tensor->data : NULL,
        recvcounts,
        displacements,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    if(local_crop_data != NULL) {
        free(local_crop_data);
    }

    if(rank == 0) {
        free(recvcounts);
        free(displacements);
    }

    return;
}

