#ifndef FUNCOES_H
#define FUNCOES_H

#include <stdio.h>
#include <math.h>

// questao 1
int quadrado(int x);
int cubo(int x);
double media(double a, double b, double c);
//questao 3
int ehPar(int n);
// questao 4
int maior2(int a, int b);
int maior3(int a, int b, int c);
int menor2(int a, int b);
int menor3(int a, int b, int c);
// questao 5
double celciusParaFahrenheit(double c);
double fahrenheitParaCelcius(double f);
// questao 6
void repete(char c, int n);
void retangulo(int largura, int altura);
// questao 7
long long int fatorial(int n);
// questao 8
double areaCirculo(double r);
double perimetroCirculo(double r);
double volumeEsfera(double r);
// questao 11
void dobraV(int x);
int dobraR(int x);
int potencia(int base, int exp);
// questao 12
int contaDigitos(int n);
int somaDigitos(int n);
int inverteNumero(int n);
int ehPalindromo(int n);

#endif
