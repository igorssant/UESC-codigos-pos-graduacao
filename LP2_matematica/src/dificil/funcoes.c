#include "../../lib/medio/funcoes.h"
#include "../../lib/dificil/funcoes.h"

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
