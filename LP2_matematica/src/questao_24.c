#include <stdio.h>
#include <math.h>
#include "../lib/dificil/funcoes.h"

int main(int argc, char **argv) {
    double a = 0.0,
        b = M_PI,
        exato = 2.0,
        erro_trap_prev = 0.0,
        erro_simp_prev = 0.0;

    printf("=========================================================================================================\n");
    printf(
        "%-6s | %-14s %-12s %-12s | %-14s %-12s %-12s\n",
        "n",
        "Trap (Aprox)",
        "Trap (Erro)",
        "Trap (Razao)",
        "Simp (Aprox)",
        "Simp (Erro)",
        "Simp (Razao)"
    );
    printf("=========================================================================================================\n");

    for(int n = 2; n < 1025; n *= 2) {
        double val_trap = trapezio(a, b, n),
            erro_trap = fabs(val_trap - exato),
            val_simp = 0.0;

        simpson(a, b, n, &val_simp);

        double erro_simp = fabs(val_simp - exato);

        // dados de 'n' e trapezio
        printf("%-6d | %-14.10f %-12.4e ", n, val_trap, erro_trap);

        if (n == 2) {
            printf("%-12s | ", "-");
        } else {
            double razao_trap = erro_trap / erro_trap_prev;

            printf("%-12.4f | ", razao_trap);
        }

        // dados de simpson
        printf("%-14.10f %-12.4e ", val_simp, erro_simp);

        if(n == 2) {
            printf("%-12s\n", "-");
        } else {
            double razao_simp = erro_simp / erro_simp_prev;

            printf("%-12.4f\n", razao_simp);
        }

        erro_trap_prev = erro_trap;
        erro_simp_prev = erro_simp;
    }

    printf("=========================================================================================================\n");
    return 0;
}
