#ifndef FUNCOES_H
#define FUNCOES_H

#include "../medio/funcoes.h"
#include <stdio.h>
#define MAX 10

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
// questao 24
double g(double x);
double trapezio(double a, double b, double n);
int simpson(double a, double b, double n, double *resultado);
// questao 25
void leMatriz(int n, double A[][MAX]);
void imprimeMatriz(int n, double A[][MAX]);
void multiplica(int n, double A[][MAX], double B[][MAX], double C[][MAX]);
void transposta(int n, double A[][MAX], double T[][MAX]);
double traco(int n, double A[][MAX]);
int ehSimetrica(int n, double A[][MAX]);
int iguais(int n, double A[][MAX], double B[][MAX], double tol);
void potenciaMatriz(int n, double A[][MAX], int k, double P[][MAX]);
// questao 26
int crivo(int n, int ehPrimo[]);
int goldbach(int n, const int ehPrimo[], int *p, int *q);
int contaDecomposicoes(int n, const int ehPrimo[]);
// questao 27
int valorDigito(char c);
char caractereDigito(int d);
int decimalParaBase(int n, int base, char s[]);
int baseParaDecimal(const char s[], int base, int *valor);
// questao 28
double horner(const double c[], int grau, double x);
int derivada(const double c[], int grau, double d[]);
int newton(const double c[], int grau, double x0, double tol, int maxIter, double *raiz, int *iter);
// questao 30
void somaC(double a, double b, double c, double d, double *re, double *im);
void multiplicaC(double a, double b, double c, double d, double *re, double *im);
int divideC(double a, double b, double c, double d, double *re, double *im);
double moduloC(double a, double b);
double argumentoC(double a, double b);
void polarParaRetangular(double r, double theta, double *a, double *b);
void potenciaC(double a, double b, int n, double *re, double *im);
int raizesC(double a, double b, int n, double re[], double im[]);
void imprimeC(double a, double b);

#endif
