#ifdef GRADIENT_H
#define GRADIENT_H

#include <stdio.h>
#include <stdlib.h>
#include <locale.h>
#include <math.h>

void calculate_gradient(double *x, double *y, double *theta, int data_size, double learning_rate, int max_iter, double tolerance);

#endif
