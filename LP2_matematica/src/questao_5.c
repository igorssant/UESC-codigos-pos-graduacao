#include <stdio.h>
#include "../lib/facil/funcoes.h"

int main(int argc, char **argv) {
    for(int i = 0; i < 3; i++) { 
        printf("+");

        for(int j = 0; j < 22; j++) {
            printf("-");
        }
    }

    printf("+\n| %20s | %20s | %20s |\n", "Original (C)", "Fahrenheit (F)", "Celcius (C)");

    for(int temp = -10.0; temp < 41; temp += 5) {
        for(int i = 0; i < 3; i++) { 
            printf("+");

            for(int j = 0; j < 22; j++) {
                printf("-");
            }
        }

        double fahrenheit = celciusParaFahrenheit((double) temp),
            celcius = fahrenheitParaCelcius(fahrenheit);

        printf("+\n| %20.2lf | %20.2lf | %20.2lf |\n", (double) temp, fahrenheit, celcius);
    }

    for(int i = 0; i < 3; i++) { 
        printf("+");

        for(int j = 0; j < 22; j++) {
            printf("-");
        }
    }

    printf("+\n");
    return 0;
}
