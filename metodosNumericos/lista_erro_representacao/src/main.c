#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BASE 10
#define LIMIT 33
#define EULER 2.718281828459

long double calculate_euler(int);
double calculate_euler_double(int);
float calculate_euler_float(int);

int main(int argc, char **argv) {
    double euler_hat,
      error = 0;
    FILE *file = fopen("../output/euler_results_double.txt", "w");

    if(file == NULL) {
        printf("Error opening file!\n");
        return -1;
    }

    fprintf(file, "Power\tEuler Value\t\tRelative Error\n");
    fprintf(file, "----------------------------------------\n");

    for(int i = 0; i < LIMIT; i++) {
        euler_hat = calculate_euler_double(i);
        error = fabs(EULER - euler_hat) / EULER;
        fprintf(file, "%d\t%.12lf\t%.12lf\n", i, euler_hat, error);
        printf("Euler: %.12lf, Error: %.12lf\n", euler_hat, error);
    }

    fclose(file);
    return 0;
}

long double calculate_euler(int power) {
    long double power_of_ten = (long double)powl(BASE, power);
    return (long double)powl((1.0L + (1.0L / power_of_ten)), power_of_ten);
}

double calculate_euler_double(int power) {
    double power_of_ten = (double) pow(BASE, power);
    return (double) pow((1.0L + (1.0L / power_of_ten)), power_of_ten);
}

float calculate_euler_float(int power) {
    float power_of_ten = (float) pow(BASE, power);
    return (float) pow((1.0f + (1.0f / power_of_ten)), power_of_ten);
}
