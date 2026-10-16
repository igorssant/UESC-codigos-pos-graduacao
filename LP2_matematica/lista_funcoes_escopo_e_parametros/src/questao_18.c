#include <stdio.h>
#include "../lib/medio/funcoes.h"

int main(int argc, char **argv) {
    int valor_1 = 0,
        valor_2 = 0;

    printf("Digite dois numeros inteiros:\n>");
    scanf("%d %d", &valor_1, &valor_2);

    printf("O MDC entre %d e %d eh: %d.\n", valor_1, valor_2, mdc(valor_1, valor_2));
    printf("O MMC entre %d e %d eh: %d.\n", valor_1, valor_2, mmc(valor_1, valor_2));
    printf("O valor %d eh primo?\n(1: sim; 0: nao)\n\t=>%d.\n", valor_1, ehPrimo(valor_1));
    printf("O valor %d eh primo?\n(1: sim; 0: nao)\n\t=>%d.\n", valor_2, ehPrimo(valor_2));
    printf("Os valores %d e %d sao primos entre si?\n(1: sim; 0: nao)\n\t=>%d.\n", valor_1, valor_2, primosEntreSi(valor_1, valor_2));
    return 0;
}
