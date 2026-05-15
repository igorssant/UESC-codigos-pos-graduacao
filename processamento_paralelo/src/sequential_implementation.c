#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <float.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
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

void binarize_data(Ttensor *ptr_tensor, const float THRESHOLD);
void get_indices_with_padding(const Ttensor *ptr_tensor, Tcoordinates *ptr_coordinates);
void crop_data(const Ttensor *ptr_input_tensor, Ttensor *ptr_output_tensor);

int main(const int argc, const char *argv[]) {
    FILE *input_file,
        *output_file;
    Ttensor tensor,
        croped_tensor;
    float threshold;
    const char *filename;
    clock_t start,
        global_start = clock();

    if(argc > 2) {
        filename = argv[1];
        threshold = atof(argv[2]);
    } else {
        filename = " ";
        threshold = 0.5;
    }

    configuration();
    input_file = fopen(filename, "r");

    // lendo o tensor do arquivo
    if(input_file == NULL) {
        printf("Could not find file *%s*" endl, filename);
        exit(-1);
    }

    start = clock();
    printf("Reading data from input file..." endl);
    build_tensor_from_file(input_file, &tensor);
    printf("Read time: %lf" endl, (double) (clock() - start) / CLOCKS_PER_SEC);
    fclose(input_file);
    start = clock();

    // binarizando o tensor
    printf("Running bounding box..." endl);
    binarize_data(&tensor, threshold);
    // fazendo o bounding box
    crop_data(&tensor, &croped_tensor);
    printf("Process time: %lf" endl, (double) (clock() - start) / CLOCKS_PER_SEC);
    // escrevendo dados no arquivo de saida
    output_file = fopen("output_sequential.txt", "w");

    if(output_file == NULL) {
        printf("Could not find output file." endl);
        exit(-1);
    }

    printf("Writing data to output file..." endl);
    start = clock();

    for(size_t k = 0; k < croped_tensor.z; k++) {
        for(size_t i = 0; i < croped_tensor.x; i++) {
            for(size_t j = 0; j < croped_tensor.y; j++) {
                size_t input_index = (i * croped_tensor.y * croped_tensor.z) + (j * croped_tensor.z) + k;

                fprintf(output_file, "%lf\t", croped_tensor.data[input_index]);
            }

            fprintf(output_file, endl);
        }

        fprintf(output_file, endl "---- ---- ----" endl endl);
    }

    // finalizando o programa
    printf("Write time: %lf" endl, (double) (clock() - start) / CLOCKS_PER_SEC);
    fclose(output_file);
    destroy_tensor_data(&tensor);
    destroy_tensor_data(&croped_tensor);
    printf("Total time: %lf" endl, (double) (clock() - global_start) / CLOCKS_PER_SEC);
    return 0;
}

void binarize_data(Ttensor *ptr_tensor, const float THRESHOLD) {
    for(size_t k = 0; k < ptr_tensor->z; k++) {
        for(size_t i = 0; i < ptr_tensor->x; i++) {
            for(size_t j = 0; j < ptr_tensor->y; j++) {
                size_t index = (i * ptr_tensor->y * ptr_tensor->z) + (j * ptr_tensor->z) + k;

                ptr_tensor->data[index] = (ptr_tensor->data[index] > THRESHOLD)? 1.0 : 0.0;
            }
        }
    }

    return;
}

void get_indices_with_padding(const Ttensor *ptr_tensor, Tcoordinates *ptr_coordinates) {
    size_t x_min = ptr_tensor->x,
        y_min = ptr_tensor->y,
        z_min = ptr_tensor->z,
        x_max = 0,
        y_max = 0,
        z_max = 0;

    for(size_t k = 0; k < ptr_tensor->z; k++) {
        for(size_t i = 0; i < ptr_tensor->x; i++) {
            for(size_t j = 0; j < ptr_tensor->y; j++) {
                size_t index = (i * ptr_tensor->y * ptr_tensor->z) + (j * ptr_tensor->z) + k;

                if(ptr_tensor->data[index]) {
                    if(i < x_min) {
                        x_min = i;
                    }

                    if(j < y_min) {
                        y_min = j;
                    }

                    if(k < z_min) {
                        z_min = k;
                    }

                    if(i > x_max) {
                        x_max = i;
                    }

                    if(j > y_max) {
                        y_max = j;
                    }

                    if(k > z_max) {
                        z_max = k;
                    }
                }
            }
        }
    }

    ptr_coordinates->x_begin = (x_min > 5) ? x_min - 5 : 0;
    ptr_coordinates->y_begin = (y_min > 5) ? y_min - 5 : 0;
    ptr_coordinates->z_begin = (z_min > 5) ? z_min - 5 : 0;
    ptr_coordinates->x_end = (x_max + 5 < ptr_tensor->x) ? x_max + 5 : ptr_tensor->x - 1;
    ptr_coordinates->y_end = (y_max + 5 < ptr_tensor->y) ? y_max + 5 : ptr_tensor->y - 1;
    ptr_coordinates->z_end = (z_max + 5 < ptr_tensor->z) ? z_max + 5 : ptr_tensor->z - 1;
    return;
}

void crop_data(const Ttensor *ptr_input_tensor, Ttensor *ptr_output_tensor) {
    Tcoordinates coordinates;
    size_t slice_size = 0,
        output_index = 0,
        total_elements;

    get_indices_with_padding(ptr_input_tensor, &coordinates);

    if(coordinates.x_end < coordinates.x_begin) {
        printf("Error: No data was found." endl);
        exit(-1);
    }

    ptr_output_tensor->x = coordinates.x_end - coordinates.x_begin + 1;
    ptr_output_tensor->y = coordinates.y_end - coordinates.y_begin + 1;
    ptr_output_tensor->z = coordinates.z_end - coordinates.z_begin + 1;
    total_elements = ptr_output_tensor->x * ptr_output_tensor->y * ptr_output_tensor->z;

    allocate_array(ptr_output_tensor, total_elements); 

    slice_size = ptr_output_tensor->z * sizeof(double);

    for(size_t i = coordinates.x_begin; i <= coordinates.x_end; i++) {
        for(size_t j = coordinates.y_begin; j <= coordinates.y_end; j++) {
            size_t input_index = (i * ptr_input_tensor->y * ptr_input_tensor->z) + \
                                (j * ptr_input_tensor->z) + coordinates.z_begin;

            memcpy(&ptr_output_tensor->data[output_index], &ptr_input_tensor->data[input_index], slice_size);
            output_index += ptr_output_tensor->z;
        }
    }

    return;
}

// 2D index = (i * colunas) + j
// 3D index = (i * colunas * profundidade) + (j * profundidade) + k
//            (i * Y * Z) + (j * Z) + k
