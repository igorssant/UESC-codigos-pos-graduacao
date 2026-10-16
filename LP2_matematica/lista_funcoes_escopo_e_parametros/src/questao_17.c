#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../lib/medio/funcoes.h"

int main(int argc, char **argv) {
    int tamanho_vetor = 0;

    printf("Digite o tamanho do vetor:\n>");
    scanf("%d", &tamanho_vetor);

    if(tamanho_vetor < 1) {
        return -1;
    }

    int vetor[tamanho_vetor],
        limite_menor = 0,
        limite_maior = 0;

    printf("Digite os limites do vetor:\n>");
    scanf("%d %d", &limite_menor, &limite_maior);
    srand(time(NULL));
    preencheAleatorio(vetor, tamanho_vetor, limite_menor, limite_maior);
    imprimeVetor(vetor, tamanho_vetor);
    inverteVetor(vetor, tamanho_vetor);
    printf("Vetor invertido:\n");
    imprimeVetor(vetor, tamanho_vetor);

    int valor_ocorrido = 0;

    printf("Digite um valor inteiro:\n>");
    scanf("%d", &valor_ocorrido);
    printf("A ocorrencia de %d no vetor foi de %d vez(es).\n", valor_ocorrido, contaOcorrencias(vetor, tamanho_vetor, valor_ocorrido));
    return 0;
}
