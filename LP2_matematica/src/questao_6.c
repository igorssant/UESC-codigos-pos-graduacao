#include <stdio.h>
#include "../lib/facil/funcoes.h"

int main(int argc, char **argv) {
    int largura = 0,
        altura = 0;

    printf("Digite a largura a altura de um retangulo:\n>");
    scanf("%d %d", &largura, &altura);
    retangulo(largura, altura);
    return 0;
}
