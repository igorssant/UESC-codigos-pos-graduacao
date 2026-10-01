#include <stdio.h>
#include <math.h>
#include "../../lib/facil/funcoes.h"

int quadrado(int x) {
    return x * x;
}

int cubo(int x) {
    return quadrado(x) * x;
}

int ehPar(int n) {
    return !(n % 2);
}

int maior2(int a, int b) {
	if(a > (b - 1)) {
		return a;
	}

	return b;
}

int maior3(int a, int b, int c) {
	return maior2(maior2(a, b), c);
}

double celciusParaFahrenheit(double c) {
	return ((9.0 * c) / 5.0) + 32.0;
}

double fahrenheitParaCelcius(double f) {
	return (f - 32.0) * (5.0 / 9.0);
}


void repete(char c, int n) {
	for(int i = 0; i < n; i++) {
		printf("%c", c);
	}
}

void retangulo(int largura, int altura) {
	char simbolo = '*',
		vazio = ' ';

	repete(simbolo, largura);
	printf("\n");
	altura -= 2;

	for(int i = 0; i < altura; i++) {
		repete(simbolo, 1);
		repete(vazio, largura - 2);
		repete(simbolo, 1);
		printf("\n");
	}

	repete(simbolo, largura);
}


long long int fatorial(int n) {
	if(n < 0 || n > 20) {
		return -1;
	}

	if(n < 3) {
		return n;
	}

	long long int resultado = 2;

	for(int i = 3; i < n; i++) {
		resultado *= (long long int) i;
	}

	return resultado;
}

double areaCirculo(double r) {
	if(r < 0) {
		return -1;
	}

	return M_PI * quadrado(r);
}

double perimetroCirculo(double r) {
	if(r < 0) {
		return -1;
	}

	return 2.0 * M_PI * r;
}

double volumeEsfera(double r) {
	if(r < 0) {
		return -1;
	}

	return areaCirculo(r) * quadrado(r);
}

void dobraV(int x) {
	x = 2 * x;
}

int dobraR(int x) {
	return 2 * x;
}

// quantidade de dígitos de n (por exemplo, 3 para 507)
int contaDigitos(int n) {
    int contador = 0;

    while(n > 9) {
        n /= 10;
        contador++;
    }

    contador++;
    return contador;
}

// soma dos dígitos de n (por exemplo, 12 para 507)
int somaDigitos(int n) {
    int soma = 0;

    while(n > 9) {
        int digito = n % 10;

        n /= 10;
        soma += digito;
    }

    soma += n;
    return soma;
}

// número com os dígitos na ordem inversa (por exemplo, 4321 para 1234)
int inverteNumero(int n) {
    int invertido = 0;

    while(n != 0) {
        int ultimoDigito = n % 10;

        invertido = (invertido * 10) + ultimoDigito;
        n /= 10;
    }

    return invertido;
}

// retorna 1 se n for palíndromo, usando obrigatoriamente inverteNumero
int ehPalindromo(int n) {
    if(n < 0) {
        return 0;
    }

    return n == inverteNumero(n);
}
