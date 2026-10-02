#include <stdio.h>
#include "../lib/facil/funcoes.h"

int main(int argc, char **argv) {
    int limite_fatorial = 0;

    printf("Digite um numero natural menor que 20 para calcular seu fatorial:\n>");
    scanf("%d", &limite_fatorial);

    long long int fatorial_resultado = fatorial(limite_fatorial);

    printf("O fatorial de %d eh: %lld", limite_fatorial, fatorial_resultado);
    return 0;
}
