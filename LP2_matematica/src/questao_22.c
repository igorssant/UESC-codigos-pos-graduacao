#include <stdio.h>
#include "../lib/dificil/funcoes.h"

int main(int argc, char **argv) {
    int numerador = 0,
        denominador = 0;

    printf("Digite dois numeros de ponto flutuante para formar uma fracao:\n>");
    scanf("%d %d", &numerador, &denominador);
    printf("A fracao eh: ");
    imprimeFracao(numerador, denominador);

    int sucesso = simplifica(&numerador, &denominador);

    if(!sucesso) {
        return -1;
    }

    printf("Apos uma simplificacao, a fracao fica:");
    imprimeFracao(numerador, denominador);

    int novo_numerador = 0,
        novo_denominador = 0;

    printf("Digite outros dois numeros de ponto flutuante para formar uma nova fracao:\n>");
    scanf("%d %d", &novo_numerador, &novo_denominador);
    sucesso = simplifica(&novo_numerador, &novo_denominador);

    if(!sucesso) {
        return -1;
    }

    printf("Apos uma simplificacao, a fracao fica:");
    imprimeFracao(novo_numerador, novo_denominador);

    int mul_numerador = 0,
        mul_denominador = 0,
        div_numerador = 0,
        div_denominador = 0;

    multiplicaFracoes(numerador, denominador, novo_numerador, novo_denominador, &mul_numerador, &mul_denominador);
    divideFracoes(numerador, denominador, novo_numerador, novo_denominador, &mul_numerador, &mul_denominador);
    printf("O resultado da multiplicacao entre ");
    imprimeFracao(numerador, denominador);
    printf("e ");
    imprimeFracao(novo_numerador, novo_denominador);
    printf("eh:\n");
    imprimeFracao(mul_numerador, mul_denominador);
    printf("\n\nO resultado da divisao entre ");
    imprimeFracao(numerador, denominador);
    printf("e ");
    imprimeFracao(novo_numerador, novo_denominador);
    printf("eh:\n");
    imprimeFracao(div_numerador, div_denominador);
    printf("\n");
    return 0;
}
