#include <stdio.h>
#include "../lib/medio/funcoes.h"

int main(int argc, char **argv) {
    int dividendo = 0,
        divisor = 0;

    printf("Digite dois numero inteiros; O divisor e o dividendo:\n>");
    scanf("%d %d", &dividendo, &divisor);

    int quociente = 0,
        resto = 0;

    int erro = divide(dividendo, divisor, &quociente, &resto);

    if(!erro) {
        printf("Erro.\n Nao foi possivel fazer a divisao");
        return -1;
    }

    printf("O quociente eh: %d\nE o resto eh: %d\n", quociente, resto);
    return 0;
}
