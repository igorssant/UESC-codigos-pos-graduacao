#include <stdio.h>
#include "../lib/dificil/funcoes.h"

int main(int argc, char **argv) {
    int maximo_iteracoes = 0,
        iteracoes_real = -1;
    double intervalo[2] = {1.0, 2.0},
        tolerancia[3] = {1e-3, 1e-8, 1e-10},
        raiz = 0.0;

    printf("O intervalo eh: [%lf, %lf].\nDigite a quantidade maxima de iteracoes:\n>", intervalo[0], intervalo[1]);
    scanf("%d", &maximo_iteracoes);

    for(int i = 0; i < 3; i++) {
        printf(
            "Em teoria, eh para demorar %d iteracoes para convegir.\n",
            iteracoesTeoricas(intervalo[0], intervalo[1], tolerancia[i])
        );
        bissecao(intervalo[0], intervalo[1], tolerancia[i],  maximo_iteracoes, &raiz, &iteracoes_real);
        printf(
            "A raiz para tolerancia = %lf eh: %lf.\nE demorou %d iteracoes para convergir.\n",
            tolerancia[i],
            raiz,
            iteracoes_real
        );
    }

    return 0;
}
