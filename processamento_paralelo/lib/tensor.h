#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <float.h>

typedef struct tensor {
    double *data;
    size_t x;
    size_t y;
    size_t z;
} Ttensor;

void build_tensor_from_file(FILE *file, Ttensor *ptr_tensor);
void destroy_tensor_data(Ttensor *ptr_tensor);
void allocate_array(Ttensor *ptr_tensor, size_t data_size);

#endif

