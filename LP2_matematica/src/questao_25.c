#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../lib/dificil/funcoes.h"

void preencher_aleatorio(double matriz[][MAX]);

int main(int argc, char **argv) {
    srand(time(NULL));

    double matriz_A[MAX][MAX],
        matriz_B[MAX][MAX],
        matriz_C[MAX][MAX];

    //leMatriz(MAX, matriz_A);
    //leMatriz(MAX, matriz_B);
    preencher_aleatorio(matriz_A);
    preencher_aleatorio(matriz_B);
    printf("A matriz \'A\' eh:\n");
    imprimeMatriz(MAX, matriz_A);
    printf("A matriz \'B\' eh:\n");
    imprimeMatriz(MAX, matriz_B);
    multiplica(MAX, matriz_A, matriz_B, matriz_C);
    printf("A matriz \'C\' eh:\n");
    imprimeMatriz(MAX, matriz_C);

    double matriz_At[MAX][MAX];

    transposta(MAX, matriz_A, matriz_At);
    printf("A matriz \'A^t\' eh:\n");
    imprimeMatriz(MAX, matriz_At);
    return 0;
}

void preencher_aleatorio(double matriz[][MAX]) {
    for(int i = 0; i < MAX; i++) {
        for(int j = 0; j < MAX; j++) {
            matriz[i][j] = ((float) rand() / (float) RAND_MAX) * (20.0 - 1.0);
        }
    }
}
