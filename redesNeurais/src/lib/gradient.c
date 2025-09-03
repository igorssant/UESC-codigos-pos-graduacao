#include "gradient.h"
#include <stdio.h>
#include <stdlib.h>
#include <locale.h>
#include <math.h>

void include_portuguese() {
    if(setlocale(LC_ALL, "pt_BR.UTF-8") == NULL) {
        setlocale(LC_ALL, "");
    }

    return;
}

void calculate_gradient(double *x, double *y, double *theta, int data_size, double learning_rate, int max_iter, double tolerance) {
    double errors[data_size],
        predict = 0.0,
        gradient_w0 = 0.0,
        gradient_w1 = 0.0,
        magnitude = 0.0;

    include_portuguese();
    
    if(learning_rate < 0.0 || learning_rate > 1.2) {
        printf("Erro.\nO gradiente deve ser um número entre '0,0' e '1,2'.\nAbortando execução do programa...\n");
        exit(-1);
    }

    if(tolerance <= 0.0) {
        printf("Erro.\nA toelrância deve ser um número maior que '0,0'.\nAbortando execução do programa...\n");
        exit(-1);
    }

    printf("Inicializando o cálculo do gradiente descendente...\n");

    for(int i = 0; i < max_iter; i++) {
        // calculando os erros
        for(int j = 0; j < data_size; j++) {
            predict = theta[0] + theta[1] * x[j];
            errors[j] = predict - y[j];
        }

        // calculando os gradientes
        for(int j = 0; j < data_size; j++) {
            gradient_w0 += errors[j];
            gradient_w1 += errors[j] * x[j];
        }

        // normalizando o gradiente
        gradient_w0 *= (1.0 / data_size);
        gradient_w1 *= (1.0 / data_size);

        // atualizando os pesos
        theta[0] -= learning_rate * gradient_w0;
        theta[1] -= learning_rate * gradient_w1;

        // verificando a convergencia
        magnitude = sqrt(pow(gradient_w0, 2.0) + pow(gradient_w1, 2.0));

        if(magnitude < tolerance) {
            break;
        }
        
    }

    printf("Cálculo do gradiente descendente finalizado.\n");
    return;
}
