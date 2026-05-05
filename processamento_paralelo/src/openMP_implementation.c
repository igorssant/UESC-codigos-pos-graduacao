#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <strings.h>

#ifdef _OPENMP
    #include <omp.h>
#else
    #define omp_get_wtime() 0.0
#endif

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

void omp_configuration(const unsigned int number_of_threads);
void build_tensor_from_file(FILE *file, Ttensor *tensor);
void binarize_data(Ttensor *ptr_tensor, const float THRESHOLD);
void get_indices(const Ttensor *ptr_tensor, Tcoordinates *ptr_coordinates);
void crop_data(const Ttensor *ptr_input_tensor, Ttensor *ptr_output_tensor);
void print_load_balance(const long *thread_work, const int number_of_threads, const long total_work);

int main(const int argc, const char *argv[]) {
    FILE *input_file,
        *output_file;
    Ttensor tensor,
        croped_tensor;
    int number_of_threads;
    const char *filename;
    float threshold,
        start,
        global_start = omp_get_wtime();

    if(argc > 3) {
        filename = argv[1];
        threshold = atof(argv[2]);
        number_of_threads = atoi(argv[3]);
    } else {
        filename = " ";
        threshold = 0.5;
        number_of_threads = 4;
    }

    omp_set_num_threads(number_of_threads);
    configuration();
    input_file = fopen(filename, "r");

    // lendo o tensor do arquivo
    if(input_file == NULL) {
        printf("Could not find file *%s*" endl, filename);
        exit(-1);
    }

    start = omp_get_wtime();
    printf("Reading data from input file..." endl);
    build_tensor_from_file(input_file, &tensor);
    printf("Read time: %f" endl, omp_get_wtime() - start);
    fclose(input_file);
    start = omp_get_wtime();

    // binarizando o tensor
    printf("Running bounding box..." endl);
    binarize_data(&tensor, threshold);
    // fazendo o bounding box
    crop_data(&tensor, &croped_tensor);
    printf("Process time: %f" endl, omp_get_wtime() - start);
    // escrevendo dados no arquivo de saida
    output_file = fopen("output_openMP.txt", "w");

    if(output_file == NULL) {
        printf("Could not find output file." endl);
        exit(-1);
    }

    printf("Writing data to output file..." endl);
    start = omp_get_wtime();

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
    printf("Write time: %f" endl, omp_get_wtime() - start);
    fclose(output_file);
    destroy_tensor_data(&tensor);
    destroy_tensor_data(&croped_tensor);
    printf("Total time: %f" endl, omp_get_wtime() - global_start);
    return 0;
}

void binarize_data(Ttensor *ptr_tensor, const float THRESHOLD) {
    size_t X = ptr_tensor->x,
        Y = ptr_tensor->y,
        Z = ptr_tensor->z;
    int number_of_threads = omp_get_max_threads();
    long *thread_work = (long*) calloc(number_of_threads, sizeof(long));

    // collapse(2)
    #pragma omp parallel for schedule(guided, 4)
    for(size_t i = 0; i < X; i++) {
        for(size_t j = 0; j < Y; j++) {
            for(size_t k = 0; k < Z; k++) {
                size_t index = (i * Y * Z) + (j * Z) + k;
                
                ptr_tensor->data[index] = (ptr_tensor->data[index] > THRESHOLD)? 1.0 : 0.0;
                thread_work[omp_get_thread_num()]++;
            }
        }
    }

    printf("BINARIZE DATA" endl);
    print_load_balance(thread_work, number_of_threads, X * Y * Z);
    free(thread_work);
    return;
}

void get_indices(const Ttensor *ptr_tensor, Tcoordinates *ptr_coordinates) {
    size_t x_min = ptr_tensor->x,
        y_min = ptr_tensor->y,
        z_min = ptr_tensor->z,
        x_max = 0,
        y_max = 0,
        z_max = 0;
    int number_of_threads = omp_get_max_threads();
    long *thread_work = (long*) calloc(number_of_threads, sizeof(long));

    // collapse(2)
    #pragma omp parallel for schedule(guided, 4) reduction(min:x_min, y_min, z_min) reduction(max:x_max, y_max, z_max)
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

                thread_work[omp_get_thread_num()]++;

            }
        }
    }

    ptr_coordinates->x_begin = x_min;
    ptr_coordinates->y_begin = y_min;
    ptr_coordinates->z_begin = z_min;
    ptr_coordinates->x_end = x_max;
    ptr_coordinates->y_end = y_max;
    ptr_coordinates->z_end = z_max;
    printf("GET INDICES" endl);
    print_load_balance(thread_work, number_of_threads, ptr_tensor->x * ptr_tensor->y * ptr_tensor->z);
    free(thread_work);
    return;
}

void crop_data(const Ttensor *ptr_input_tensor, Ttensor *ptr_output_tensor) {
    Tcoordinates coordinates;
    size_t total_elements,
        slice_size;
    int number_of_threads = omp_get_max_threads();
    long *thread_work = (long*) calloc(number_of_threads, sizeof(long));

    get_indices(ptr_input_tensor, &coordinates);

    if(coordinates.x_end < coordinates.x_begin) {
        printf("Error: No data was found.\n");
        exit(-1);
    }

    ptr_output_tensor->x = coordinates.x_end - coordinates.x_begin + 1;
    ptr_output_tensor->y = coordinates.y_end - coordinates.y_begin + 1;
    ptr_output_tensor->z = coordinates.z_end - coordinates.z_begin + 1;
    total_elements = ptr_output_tensor->x * ptr_output_tensor->y * ptr_output_tensor->z;
    allocate_array(ptr_output_tensor, total_elements);
    slice_size = ptr_output_tensor->z * sizeof(double);
    
    // collapse(1)
    #pragma omp parallel for schedule(guided, 4)
    for(size_t i = coordinates.x_begin; i <= coordinates.x_end; i++) {
        for(size_t j = coordinates.y_begin; j <= coordinates.y_end; j++) {
            size_t output_index = ((i - coordinates.x_begin) * ptr_output_tensor->y * ptr_output_tensor->z) + ((j - coordinates.y_begin) * ptr_output_tensor->z),
                input_index = (i * ptr_input_tensor->y * ptr_input_tensor->z) + (j * ptr_input_tensor->z) + coordinates.z_begin;

            memcpy(&ptr_output_tensor->data[output_index], &ptr_input_tensor->data[input_index], slice_size);
            thread_work[omp_get_thread_num()]++;
        }
    }

    printf("CROP DATA" endl);
    print_load_balance(thread_work, number_of_threads, ptr_input_tensor->x * ptr_input_tensor->y * ptr_input_tensor->z);
    free(thread_work);
    return;
}

void print_load_balance(const long *thread_work, const int number_of_threads, const long total_work) {
    long max_work = 0,
        min_work = total_work;
    double imbalance;

    for(int i = 0; i < number_of_threads; i++) {
        double percentage = (100.0 * thread_work[i]) / total_work;

        printf("Thread %d -> %ld iterations (%f)" endl, i, thread_work[i], percentage);

        if(thread_work[i] > max_work) {
            max_work = thread_work[i];
        }

        if(thread_work[i] < min_work) {
            min_work = thread_work[i];
        }
    }

    imbalance = ((double) (max_work - min_work) / max_work) * 100.0;

    printf("===== LOAD BALANCING ======" endl);
    printf("Max work: %ld" endl, max_work);
    printf("Min work: %ld" endl, min_work);
    printf("Imbalance: %f%%" endl, imbalance);
    printf("===========================" endl);
    return;
}

