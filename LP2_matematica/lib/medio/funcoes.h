#ifndef FUNCOES_H
#define FUNCOES_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// questao 13
void trocaErrada(int a, int b);
void troca(int *a, int *b);
// questao 14
void ordena2(int *a, int *b);
void ordena3(int *a, int *b, int *c);
// questao 15
int divide(int a, int b, int *q, int *r);
// questao 16
void estatisticas(const double x[], int n, double *max, double *min, double *media);
double desvioPadrao(const double x[], int n, double media);
// questao 17
void preencheAleatorio(int v[], int n, int a, int b);
void imprimeVetor(const int v[], int n);
void inverteVetor(int v[], int n);
int contaOcorrencias(const int v[], int n, int valor);
// exercicio 18
int mdc(int a, int b);
int mmc(int a, int b);
int ehPrimo(int n);
int primosEntreSi(int a, int b);
// questao 19
int raizes(double a, double b, double c, double *x1, double *x2);

#endif
