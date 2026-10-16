#include <stdio.h>
#include "../lib/medio/funcoes.h"

int main(int argc, char **argv) {
    int valor_1 = 0,
        valor_2 = 0,
        valor_3 = 0;

    printf("Digite tres numeros inteiros:\n>");
    scanf("%d %d %d", &valor_1, &valor_2, &valor_3);
    printf("Antes de ordena3():\n");
    printf("a = %d, b = %d e c = %d\n", valor_1, valor_2, valor_3);
    ordena3(&valor_1, &valor_2, &valor_3);
    printf("Depois de ordena3():\n");
    printf("a = %d, b = %d e c = %d\n", valor_1, valor_2, valor_3);
    return 0;
}
