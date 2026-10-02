#include <stdio.h>
#include "../lib/facil/funcoes.h"

int main(int argc, char **argv) {
    double raio_circulo = 0.0;

    printf("Digite o raio de um circulo:\n>");
    scanf("%lf", &raio_circulo);
    printf("Seu diametro eh: %.2lf m\n", 2.0 * raio_circulo);
    printf("Sua area eh: %.2lf m^2\n", areaCirculo(raio_circulo));
    printf("Seu perimetro eh: %.2lf m\n", perimetroCirculo(raio_circulo));
    printf("Seu volume eh: %.2lf m^3\n", volumeEsfera(raio_circulo));
    return 0;
}
