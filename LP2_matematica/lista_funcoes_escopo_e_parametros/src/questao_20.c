#include <stdio.h>

int contaA();
int contaB();

int main(int argc, char **argv) {
    for(int i = 1; i < 4; i++) {
        int a = contaA(),
            b = contaB();

        printf("%d: A = %d, B = %d\n", i, a, b);
    }

    return 0;
}

int contaA() {
    int c = 0;

    c++;
    return c;
}

int contaB() {
    static int c = 0;

    c++;
    return c;
}
