#include <stdio.h>
#include "../lib/medio/funcoes.h"

int main(int argc, char **argv) {
    int tamanho_vetor = 1;

    printf("Digite o tamanho do vetor:\n>");
    scanf("%d", &tamanho_vetor);

    double vetor[tamanho_vetor];

    printf("Digite o conteudo do vetor:\n");

    for(int i = 0; i < tamanho_vetor; i++) {
        printf(">");
        scanf("%lf", &vetor[i]);
    }

    double max = 0.0,
        min = 0.0,
        media = 0.0;

    estatisticas(vetor, tamanho_vetor, &max, &min, &media);
    printf("As estatisticas sao:\nMaximo: %.2lf\nMinimo: %.2lf\nMedia: %.2lf\n", max, min, media);

    double desvio_padrao = desvioPadrao(vetor, tamanho_vetor, media);

    printf("O desvio padrao eh: %.2lf\n", desvio_padrao);
    return 0;
}
