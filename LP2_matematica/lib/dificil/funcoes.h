#ifndef FUNCOES_H
#define FUNCOES_H

#include <stdio.h>

// questao 22
int simplifica(int *num, int *den);
int somaFracoes(int n1, int d1, int n2, int d2, int *nr, int *dr);
int multiplicaFracoes(int n1, int d1, int n2, int d2, int *nr, int *dr);
int divideFracoes(int n1, int d1, int n2, int d2, int *nr, int *dr);
void imprimeFracao(int num, int den);
// questao 23
double f(double x);
int bissecao(double a, double b, double tol, int maxIter, double *raiz, int *iter);
int iteracoesTeoricas(double a, double b, double tol);

#endif
