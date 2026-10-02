#include <stdio.h>
#include "../lib/facil/funcoes.h"

#define VALOR 507

int main(int argc, char ** argv) {
    printf("O valor escolhido eh: %d.\n", VALOR);
    printf("Resultado de contaDigitos() => %d.\n", contaDigitos(VALOR));
    printf("Resultado de somaDigitos() => %d.\n", somaDigitos(VALOR));
    printf("Resultado de inverteNumero() => %d.\n", inverteNumero(VALOR));
    printf("Resultado de ehPalindromo() => %d.\n", ehPalindromo(VALOR));
    return 0;
}
