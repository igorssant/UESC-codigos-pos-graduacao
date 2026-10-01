#include <stdio.h>
#include "../../lib/medio/funcoes.h"

void trocaErrada(int a, int b) {
    int temp = a;

    a = b;
    b = temp;
    printf("===== trocaErrada() :: apos a troca =====");
    printf("O valor de a: %d\n", a);
    printf("-----------------------------------------");
    printf("O valor de b: %d\n", b);
    printf("=========================================");
}

void troca(int *a, int *b) {
    int temp = *a;

    *a = *b;
    *b = temp;
    printf("===== troca() :: apos a troca =====");
    printf("O valor de a: %d\n", *a);
    printf("-----------------------------------");
    printf("O valor de b: %d\n", *b);
    printf("===================================");
}

void ordena2(int *a, int *b) {
    if((*b) > ((*a) - 1)) {
        troca(a, b);
    }
}

void ordena3(int *a, int *b, int *c) {
    ordena2(b, c);
    ordena2(a, b);
}

int divide(int a, int b, int *q, int *r) {}

void estatisticas(const double x[], int n, double *max, double *min, double *media) {}

double desvioPadrao(const double x[], int n, double media) {}

void preencheAleatorio(int v[], int n, int a, int b) {}

void imprimeVetor(const int v[], int n) {}

void inverteVetor(int v[], int n) {}

int contaOcorrencias(const int v[], int n, int valor) {}

int mdc(int a, int b) {}

int mmc(int a, int b) {}

int ehPrimo(int n) {}

int primosEntreSi(int a, int b) {}

int raizes(double a, double b, double c, double *x1, double *x2) {}
