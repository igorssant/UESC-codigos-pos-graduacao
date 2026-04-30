#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <float.h>
#include "../lib/tensor.h"

void build_tensor_from_file(FILE *file, Ttensor *ptr_tensor) {
    size_t array_size;

    fscanf(file, "%lu", &ptr_tensor->x);
    fscanf(file, "%lu", &ptr_tensor->y);
    fscanf(file, "%lu", &ptr_tensor->z);
    array_size = ptr_tensor->x * ptr_tensor->y * ptr_tensor->z;
    allocate_array(ptr_tensor, array_size);

    for(size_t i = 0; i < array_size; i++) {
        fscanf(file, "%lf", (ptr_tensor->data + i));
    }

    return;
}

void destroy_tensor_data(Ttensor *ptr_tensor) {
    free(ptr_tensor->data);

    return;
}

void allocate_array(Ttensor *ptr_tensor, size_t data_size) {
    ptr_tensor->data = (double*) malloc(data_size * sizeof(double));
    
    if(ptr_tensor->data == NULL) {
        printf("Could not allocate enough space to save tensor data.\n");
        exit(-1);
    }

    return;
}

