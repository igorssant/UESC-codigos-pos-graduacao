#include <stdio.h>
#include "../lib/facil/funcoes.h"


int main(int argc, char **argv) {
    int n = 0;

    printf("Digite um numero natural:\n>");
    scanf("%d", &n);

    if(ehPar(n)) {
        printf("%d eh um numero par.\n", n);
    } else {
        printf("%d eh um numero impar.\n", n);
    }

    int contador_impar = 0;

    for(int i = 1; i < n; i++) {
        if(!ehPar(i)) {
            contador_impar++;
        }
    }

    printf("Entre 1 e %d existem %d numeros impares.\n", n, contador_impar);
    return 0;
}

