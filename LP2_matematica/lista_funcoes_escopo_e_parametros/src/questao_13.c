#include <stdio.h>
#include "../lib/medio/funcoes.h"

int main(int argc, char **argv) {
    int primeiro = 0,
        segundo = 0;

    printf("Digite dois numeros inteiros:\n>");
    scanf("%d %d", &primeiro, &segundo);
    printf("(Antes de trocaErrada()) primeiro = %d e segundo = %d.\n", primeiro, segundo);
    trocaErrada(primeiro, segundo);
    printf("(Depois de trocaErrada()) primeiro = %d e segundo = %d.\n", primeiro, segundo);
    troca(&primeiro, &segundo);
    printf("(Depois de troca()) primeiro = %d e segundo = %d.\n", primeiro, segundo);
    return 0;
}
