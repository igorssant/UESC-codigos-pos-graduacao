#include "../lib/medio/funcoes.h"
#include <stdio.h>

int main(int argc, char **argv) {
    double valores[4][3] = {
        {1.0, -3.0, 2.0},
        {1.0, 2.0, 1.0},
        {1.0, 0.0, 1.0},
        {0.0, 2.0, 1.0}
    };

    for(int i = 0; i < 4; i++) {
        double x_1,
            x_2;

        x_1 = x_2 = 0.0;
        printf("Caso %d.\n\tRaizes: %d.\n", i, raizes(valores[i][0], valores[i][1], valores[i][2], &x_1, &x_2));
        printf("As raizes sao: \'%.2lf\' e \'%.2lf\'.\n", x_1, x_2);
    }

    return 0;
}
