#include "../../lib/dificil/funcoes.h"
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
