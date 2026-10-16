#include <stdio.h>
#include "../lib/facil/funcoes.h"


int main(int argc, char **argv) {
    int nro = 0;

    printf("Digite um numero que queira saber o quadrado e o cubo:\n>");
    scanf("%d", &nro);
    printf("O quadrado de %d eh: %d.\n", nro, quadrado(nro));
    printf("O cubo de %d eh %d.\n", nro, cubo(nro));

    double nro_1 = 0.0,
        nro_2 = 0.0,
        nro_3 = 0.0;

    printf("Digite tres numeros para saber a media entre eles:\n >");
    scanf("%lf %lf %lf", &nro_1, &nro_2, &nro_3);
    printf("A media entre %lf, %lf e %lf eh: %lf.\n", nro_1, nro_2, nro_3, media(nro, nro_2, nro_3));
    return 0;
}
