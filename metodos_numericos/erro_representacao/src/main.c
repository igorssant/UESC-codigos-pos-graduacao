#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BASE 10
#define LIMIT 33
#define EULER 2.718281828459

long double calculate_euler(int);

int main(int argc, char **argv)
{
    long double euler_hat;
    long double error = 0;

    FILE *file = fopen("output/euler_results.txt", "w");

    if (file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    fprintf(file, "Power\tEuler Value\t\tAbsolute Error\n");
    fprintf(file, "----------------------------------------\n");

    for (int i = 0; i < LIMIT; i++)
    {
        euler_hat = calculate_euler(i);
        error = fabs(EULER - euler_hat) / EULER;

        fprintf(file, "%d\t%.12Lf\t%.12Lf\n", i, euler_hat, error);
        printf("Euler: %.12Lf, Error: %.12Lf\n", euler_hat, error);
    }

    fclose(file);

    return 0;
}

long double calculate_euler(int power)
{
    long double power_of_ten = (long double)powl(BASE, power);
    return (long double)powl((1.0L + (1.0L / power_of_ten)), power_of_ten);
}
