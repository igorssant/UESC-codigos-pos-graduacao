#include <stdio.h>
#include "../lib/facil/funcoes.h"

int main(int argc, char **argv) {
    int a = 7;

    printf("(Inicial) a = %d.\n", a);
    dobraV(a);
    printf("(Apos dobraV()) a = %d.\n", a);
    a = dobraR(a);
    printf("(Apos dobraR()) a = %d.\n", a);

    int b = 0;

    printf("Digite um numero inteiro para a base e um numero inteiro para o expoente:\n>");
    scanf("%d %d", &a, &b);

    printf("(Antes de chamar potencia()) base = %d e expoente = %d.\n", a, b);
    potencia(a, b);
    printf("(Depois de chamar potencia()) base = %d e expoente = %d.\n", a, b);
    return 0;
}
