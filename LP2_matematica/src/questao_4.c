#include <stdio.h>
#include "../lib/facil/funcoes.h"

int main(int argc, char **argv) {
    int a = 0,
        b = 0,
        c = 0;

    printf("Digite tres numeros inteiros:\n>");
    scanf("%d %d %d", &a, &b, &c);
    printf("O maior numero entre %d, %d e %d eh: %d.\n", a, b, c, maior3(a, b, c));
    printf("O menor numero entre %d, %d e %d eh: %d.\n", a, b, c, menor3(a, b, c));
    return 0;
}
