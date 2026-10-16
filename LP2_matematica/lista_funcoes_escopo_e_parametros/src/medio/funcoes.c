#include "../../lib/medio/funcoes.h"
#include <iso646.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void trocaErrada(int a, int b) {
    int temp = a;

    a = b;
    b = temp;
    printf("===== trocaErrada() :: apos a troca =====\n");
    printf("O valor de a: %d\n", a);
    printf("-----------------------------------------\n");
    printf("O valor de b: %d\n", b);
    printf("=========================================\n");
}

void troca(int *a, int *b) {
    int temp = *a;

    *a = *b;
    *b = temp;
    printf("===== troca() :: apos a troca =====\n");
    printf("O valor de a: %d\n", *a);
    printf("-----------------------------------\n");
    printf("O valor de b: %d\n", *b);
    printf("===================================\n");
}

void ordena2(int *a, int *b) {
    if(*b > (*a - 1)) {
        troca(a, b);
    }
}

void ordena3(int *a, int *b, int *c) {
    for(int i = 0; i < 2; i++) {
        ordena2(a, b);
        ordena2(b, c);
    }
}

int divide(int a, int b, int *q, int *r) {
    if(a < 0 || ((b - 1) < 0) || q == NULL || r == NULL) {
        return 0;
    }

    int quociente = 0,
        resto = a;

    while(resto >= b) {
        resto -= b;
        quociente++;
    }

    *q = quociente;
    *r = resto;
    return 1;
}

void estatisticas(const double x[], int n, double *max, double *min, double *media) {
    if((n - 1) < 0 || max == NULL || min == NULL || media == NULL) {
        return;
    }

    *max = *min = *media = x[0];

    for(int i = 1; i < n; i++) {
        *media += x[i];

        if(*max < x[i]) {
            *max = x[i];
        } else if(*min > x[i]) {
            *min = x[i];
        }
    }

    *media /= n;
}

double desvioPadrao(const double x[], int n, double media) {
    if(n < 1) {
        return -1.0;
    }

    double desvio_padrao = 0.0;

    for(int i = 0; i < n; i++) {
        desvio_padrao += pow((x[i] - media), 2);
    }

    desvio_padrao = sqrt(desvio_padrao / n);
    return desvio_padrao;
}

void preencheAleatorio(int v[], int n, int a, int b) {
    if(n < 1 || a > (b - 1)) {
        return;
    }

    for(int i = 0; i < n; i++) {
        v[i] = rand() % ((b - a) + 1);
    }
}

void imprimeVetor(const int v[], int n) {
    if(n < 1) {
        return;
    }

    printf("[ %d, ", v[0]);

    for(int i = 1; i < n - 2; i++) {
        printf("%d, ", v[i]);
    }

    printf("%d ]\n", v[n - 1]);
}

void inverteVetor(int v[], int n) {
    if(n < 1) {
        return;
    }

    for(int i = 0; i < 1 + (n / 2); i++) {
        troca(&v[i], &v[n - 1 - i]);
    }
}

int contaOcorrencias(const int v[], int n, int valor) {
    if(n < 1) {
        return -1;
    }

    int contador = 0;

    for(int i = 0; i < n; i++) {
        if(v[i] == valor) {
            contador++;
        }
    }

    return contador;
}

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

int mmc(int a, int b) {
    if(a == 0 || b == 0) {
        return 0;
    }

    a = abs(a);
    b = abs(b);
    return (a / mdc(a, b)) * b;
}

int ehPrimo(int n) {
    if(n < 0) {
        return -1;
    } else if(n > 0 && n < 2) {
        return 0;
    }

    for(int i = 2; i * i <= n; i++) {
        if(mdc(n, i) != 1) {
            return 0;
        }
    }

    return 1;
}

int primosEntreSi(int a, int b) {
    return mdc(a, b) == 1;
}

int raizes(double a, double b, double c, double *x1, double *x2) {
    if(a == 0.0 || x1 == NULL || x2 == NULL) {
        return -1;
    }

    double delta = (b * b) - (4.0 * a * c);

    if(delta < 0.0) {// delta < 0
        return 0;
    } else if(delta == 0.0) {// delta == 0
        double r = -b / (2.0 * a);
        
        *x1 = r;
        *x2 = r;
        return 1;
    }

    // delta > 0
    double sqrt_delta = sqrt(delta),
        r1 = (-b - sqrt_delta) / (2.0 * a),
        r2 = (-b + sqrt_delta) / (2.0 * a);

    // garantindo que *x1 < *x2
    if(r1 < r2) {
        *x1 = r1;
        *x2 = r2;
    } else {
        *x1 = r2;
        *x2 = r1;
    }

    return 2;
}
