#include "../../lib/dificil/funcoes.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX 10

int mdc(int a, int b) {
    a = abs(a);
    b = abs(b);

    while(b != 0) {
        int resto = a % b;

        a = b;
        b = resto;
    }

    return a;
}

int simplifica(int *num, int *den) {
    if(num == NULL || den == NULL || !(*den)) {
        return 0;
    }

    int divisor_comum = mdc(*num, *den);

    if(divisor_comum) {
        *num /= divisor_comum;
        *den /= divisor_comum;
    }

    if(*den < 0) {
        *num = -(*num);
        *den = -(*den);
    }

    return 1;
}

int somaFracoes(int n1, int d1, int n2, int d2, int *nr, int *dr) {
    if(!d1 || !d2 || nr == NULL || dr == NULL) {
        return 0;
    }

    int numerador = (n1 * d2) + (n2 * d1),
        denominador = d1 * d2,
        sucesso = simplifica(&numerador, &denominador);

    if(!sucesso) {
        return 0;
    }

    *nr = numerador;
    *dr = denominador;
    return 1;
}

int multiplicaFracoes(int n1, int d1, int n2, int d2, int *nr, int *dr) {
    if(!d1 || !d2 || nr == NULL || dr == NULL) {
        return 0;
    }

    int numerador = n1 * n2,
        denominador = d1 * d2,
        sucesso = simplifica(&numerador, &denominador);

    if(!sucesso) {
        return 0;
    }

    *nr = numerador;
    *dr = denominador;
    return 1;
}

int divideFracoes(int n1, int d1, int n2, int d2, int *nr, int *dr) {
    if(!d1 || !d2 || !n2 || nr == NULL || dr == NULL) {
        return 0;
    }

    int numerador = n1 * d2,
        denominador = d1 * n2,
        sucesso = simplifica(&numerador, &denominador);

    if(!sucesso) {
        return 0;
    }

    *nr = numerador;
    *dr = denominador;
    return 1;
}

void imprimeFracao(int num, int den) {
    printf("%d/%d\n", num, den);
}

double quadrado(double x) {
    return x * x;
}

double cubo(double x) {
    return quadrado(x) * x;
}

double f(double x) {
    return cubo(x) - x - 2.0;
}

int bissecao(double a, double b, double tol, int maxIter, double *raiz, int *iter) {
    if(raiz == NULL || iter == NULL || a >= b || tol <= 0.0 || maxIter < 1) {
        return 0;
    }

    double fa = f(a),
        fb = f(b);

    // verifica se pode existir raizes no intervalo
    if(fa * fb > 0.0) {
        return 0;
    }

    double c = a;
    int contador = 0;

    while(contador < maxIter) {
        contador++;
        c = (a + b) / 2.0;

        double fc = f(c);

        // verifica convergencia
        if(fabs(fc) < tol || ((b - a) / 2.0) < tol) {
            *raiz = c;
            *iter = contador;
            return 1; 
        }

        // atualiza limites
        if(fa * fc < 0.0) {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }
    }

    // deu ruim
    *raiz = c;
    *iter = contador;
    return 0;
}

int iteracoesTeoricas(double a, double b, double tol) {
    if(a > (b - 1) || tol <= 0.0) {
        return 0;
    }

    int iteracoes = 0;
    double amplitude = b - a;

    while(amplitude >= tol) {
        amplitude /= 2.0;
        iteracoes++;
    }

    return iteracoes;
}

double g(double x) {
    return sin(x);
}

double trapezio(double a, double b, int n) {
    if(n < 1) {
        return 0.0;
    }

    double h = (b - a) / (double) n,
        soma = (g(a) + g(b)) / 2.0;

    for(int i = 1; i < n; i++) {
        double xi = a + (double) i * h;

        soma += g(xi);
    }

    return soma * h;
}

int simpson(double a, double b, int n, double *resultado) {
    if(resultado == NULL || n < 2 || n % 2) {
        return 0;
    }

    double h = (b - a) / (double) n,
        soma = g(a) + g(b);

    for(int i = 1; i < n; i++) {
        double xi = a + i * h;

        if(i % 2) {
            soma += 4.0 * g(xi);
        } else {
            soma += 2.0 * g(xi);
        }
    }

    *resultado = (h / 3.0) * soma;
    return 1;
}

void leMatriz(int n, double A[][MAX]) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < MAX; j++) {
            scanf("%lf", &A[i][j]);
        }
    }
}

void imprimeMatriz(int n, double A[][MAX]) {
    for(int i = 0; i < n; i++) {
        printf("[\t");

        for(int j = 0; j < MAX; j++) {
            printf("%.2lf\t", A[i][j]);
        }

        printf("]\n");
    }
}

void multiplica(int n, double A[][MAX], double B[][MAX], double C[][MAX]) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < MAX; j++) {
            C[i][j] = 0.0;

            for(int k = 0; k < MAX; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

void transposta(int n, double A[][MAX], double T[][MAX]) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < MAX; j++) {
            T[j][i] = A[i][j];
        }
    }
}

double traco(int n, double A[][MAX]) {
    double soma = 0.0;
    int limite = (n < MAX) ? n : MAX; // diagonal principal

    for(int i = 0; i < limite; i++) {
        soma += A[i][i];
    }

    return soma;
}

int ehSimetrica(int n, double A[][MAX]) {
    if(n != MAX) {
        return 0;
    }

    for(int i = 0; i < n; i++) {
        for(int j = i + 1; j < n; j++) {
            if(A[i][j] != A[j][i]) {
                return 0;
            }
        }
    }

    return 1;
}

int iguais(int n, double A[][MAX], double B[][MAX], double tol) {
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < MAX; j++) {
            if(fabs(A[i][j] - B[i][j]) >= tol) {
                return 0;
            }
        }
    }

    return 1;
}

void potenciaMatriz(int n, double A[][MAX], int k, double P[][MAX]) {
    // se k == 0, a matriz eh identidade
    if(k == 0) {
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < MAX; j++) {
                P[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }

        return;
    }

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < MAX; j++) {
            P[i][j] = A[i][j];
        }
    }

    // se k == 1, o resultado eh A
    if(k == 1) {
        return;
    }

    double temp[MAX][MAX];

    for(int p = 2; p < (k + 1); p++) {
        multiplica(n, P, A, temp);

        for(int i = 0; i < n; i++) {
            for(int j = 0; j < MAX; j++) {
                P[i][j] = temp[i][j];
            }
        }
    }
}

int crivo(int n, int ehPrimo[]) {
    if(n < 0 || ehPrimo == NULL) {
        return 0;
    }

    // assumindo valores primos
    for(int i = 0; i <= n; i++) {
        ehPrimo[i] = 1;
    }

    if(n > -1) {
        ehPrimo[0] = 0;
    }

    if(n > 0) {
        ehPrimo[1] = 0;
    }

    // crivo de eratostenes
    for (int i = 2; i * i <= n; i++) {
        if(ehPrimo[i]) {
            for(int j = i * i; j < (n + 1); j += i) {
                // marcando multiplos de i como nn-primos
                ehPrimo[j] = 0;
            }
        }
    }

    // conta a quantidade total de primos
    int qtd_primos = 0;

    for(int i = 0; i < (n + 1); i++) {
        if(ehPrimo[i]) {
            qtd_primos++;
        }
    }

    return qtd_primos;
}

int goldbach(int n, const int ehPrimo[], int *p, int *q) {
    if(n < 3 || !(n % 2) || ehPrimo == NULL || p == NULL || q == NULL) {
        return 0;
    }

    for(int i = 2; i < ((n / 2) + 1); i++) {
        if(ehPrimo[i] && ehPrimo[n - i]) {
            *p = i;
            *q = n - i;
            return 1;
        }
    }

    return 0;
}

int contaDecomposicoes(int n, const int ehPrimo[]) {
    if(n < 3 || !(n % 2) || ehPrimo == NULL) {
        return 0;
    }

    int contador = 0;

    for(int i = 2; i < ((n / 2) + 1); i++) {
        if(ehPrimo[i] && ehPrimo[n - i]) {
            contador++;
        }
    }

    return contador;
}

int valorDigito(char c) {
    if(c >= '0' && c <= '9') {
        return c - '0';
    }

    if(c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    }

    if(c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }

    return -1;
}

char caractereDigito(int d) {
    if(d >= 0 && d <= 9) {
        return '0' + d;
    }

    if(d >= 10 && d <= 15) {
        return 'A' + (d - 10);
    }

    return '\0';
}

int decimalParaBase(int n, int base, char s[]) {
    if(n < 0 || base < 2 || base > 16 || s == NULL) {
        return 0;
    }

    // caso n = 0
    if(n == 0) {
        s[0] = caractereDigito(0);
        s[1] = '\0';
        return 1;
    }

    int qtd_digitos = 0,
        temp = n;

    // extraindo os digitos do menor para o maior peso
    while(temp > 0) {
        int resto = temp % base;

        s[qtd_digitos++] = caractereDigito(resto);
        temp /= base;
    }

    s[qtd_digitos] = '\0';

    // invertento a string para obter a ordem correta dos digitos
    for(int i = 0; i < (qtd_digitos / 2); i++) {
        char aux = s[i];

        s[i] = s[qtd_digitos - 1 - i];
        s[qtd_digitos - 1 - i] = aux;
    }

    return qtd_digitos;
}

int baseParaDecimal(const char s[], int base, int *valor) {
    if(s == NULL || valor == NULL || base < 2 || base > 16 || s[0] == '\0') {
        return 0;
    }

    int acumulador = 0;

    for(int i = 0; s[i] != '\0'; i++) {
        int d = valorDigito(s[i]);

        if (d == -1 || d >= base) {
            return 0;
        }

        // metodo de Horner
        acumulador = acumulador * base + d;
    }

    *valor = acumulador;
    return 1;
}

double horner(const double c[], int grau, double x) {
    if(c == NULL || grau < 0) {
        return 0.0;
    }

    double resultado = c[grau];
    
    for(int i = grau - 1; i > -1; i--) {
        resultado = resultado * x + c[i];
    }

    return resultado;
}

int derivada(const double c[], int grau, double d[]) {
    if(c == NULL || d == NULL || grau < 0) {
        return 0;
    }

    // polinomio constante: 0
    if(grau == 0) {
        d[0] = 0.0;
        return 0;
    }

    for(int i = 0; i < grau; i++) {
        d[i] = (i + 1) * c[i + 1];
    }

    return grau - 1;
}

int newton(const double c[], int grau, double x0, double tol, int maxIter, double *raiz, int *iter) {
    if(c == NULL || raiz == NULL || iter == NULL || grau < 1 || tol <= 0.0 || maxIter < 1) {
        return 0;
    }

    double *d = (double *) malloc(grau * sizeof(double));

    if(d == NULL) {
        return 0;
    }

    int grau_d = derivada(c, grau, d),
        k = 0;
    double x = x0;

    while(k < maxIter) {
        double fx = horner(c, grau, x);

        k++;

        // sucesso: valor do polinomio aprox 0
        if(fabs(fx) < tol) {
            *raiz = x;
            *iter = k;
            free(d);
            return 1;
        }

        double dfx = horner(d, grau_d, x);

        // falha: derivada nula (ou aprox zero)
        if(fabs(dfx) < 1e-12) {
            *raiz = x;
            *iter = k;
            free(d);
            return 0;
        }

        double x_proximo = x - (fx / dfx);

        // sucesso: convergiu
        if(fabs(x_proximo - x) < tol) {
            *raiz = x_proximo;
            *iter = k;
            free(d);
            return 1;
        }

        x = x_proximo;
    }

    // deu ruim
    *raiz = x;
    *iter = k;
    free(d);
    return 2;
}

void somaC(double a, double b, double c, double d, double *re, double *im) {
    if(re != NULL) {
        *re = a + c;
    }

    if(im != NULL) {
        *im = b + d;
    }
}

void multiplicaC(double a, double b, double c, double d, double *re, double *im) {
    if(re != NULL) {
        *re = (a * c) - (b * d);
    }

    if(im != NULL) {
        *im = (a * d) + (b * c);
    }
}

int divideC(double a, double b, double c, double d, double *re, double *im) {
    double den = (c * c) + (d * d);

    if(den == 0.0 || re == NULL || im == NULL) {
        return 0;
    }

    *re = (a * c + b * d) / den;
    *im = (b * c - a * d) / den;
    return 1;
}

double moduloC(double a, double b) {
    return hypot(a, b);
}

double argumentoC(double a, double b) {
    return atan2(b, a);
}

void polarParaRetangular(double r, double theta, double *a, double *b) {
    if(a != NULL) {
        *a = r * cos(theta);
    }

    if(b != NULL) {
        *b = r * sin(theta);
    }
}

void potenciaC(double a, double b, int n, double *re, double *im) {
    if(re == NULL || im == NULL) {
        return;
    }

    double real = moduloC(a, b),
        theta = argumentoC(a, b),
        real_n = pow(real, n),
        theta_n = n * theta;

    polarParaRetangular(real_n, theta_n, re, im);
}

int raizesC(double a, double b, int n, double re[], double im[]) {
    if(n < 1 || re == NULL || im == NULL) {
        return 0;
    }

    double real = moduloC(a, b),
        theta = argumentoC(a, b),
        real_raiz = pow(real, 1.0 / n);

    for(int k = 0; k < n; k++) {
        double theta_k = (theta + 2.0 * M_PI * k) / n;

        polarParaRetangular(real_raiz, theta_k, &re[k], &im[k]);
    }

    return n;
}

void imprimeC(double a, double b) {
    if(b >= 0.0) {
        printf("%.2lf + %.2lfi\n", a, b);
        return;
    }

    printf("%.2lf - %.2lfi\n", a, fabs(b));
}
