#include<stdio.h>
#include<stdlib.h>
#include<math.h>

#define TEN 10

void ask_power();
void ask_precision();
void praise_result();
float calculate_euler_float(int);
double calculate_euler_double(int);
long double calculate_euler_long_double(int);

int main(int argc, char **argv) {
  int power = 0,
    precision = 0;
  
  ask_power();
  scanf("%d", &power);
  ask_precision();
  scanf("%d", &precision);
  praise_result();
  
  switch(precision) {
    case 1:
      printf("%f\n", calculate_euler_float(power));
      break;
    
    case 2:
    printf("%lf\n", calculate_euler_double(power));
      break;
    
    case 3:
      printf("%Lf\n", calculate_euler_long_double(power));
      break;
    
    default:
      printf("Opcao incorreta.\nAbortando o codigo...\n");
  }
  
  return 0;
}

void ask_power() {
  printf("Qual potencia de 10 vode deseja calcular ?\n");
  printf("Exemplos de uso:\n");
  printf("- digite 0 para 10^0\n");
  printf("- digite 1 para 10^1\n");
  printf("- digite 2 para 10^2\n");
  printf("- digite 3 para 10^3\n");
  printf("- etc.\n> ");
  return;
}

void ask_precision() {
  printf("Qual a precisao que voce deseja ?\n");
  printf("Digite 1 para *** float ***\n");
  printf("Digite 2 para *** double ***\n");
  printf("Digite 3 para *** long double ***\n");
  printf("> ");
  return;
}

void praise_result() {
  printf("O resultado: ");
  return;
}

float calculate_euler_float(int power) {
  float power_of_ten = (float) pow(TEN, power);
  return (float) pow((1.0 + (1.0 / power_of_ten)), power_of_ten);
}

double calculate_euler_double(int power) {
  double power_of_ten = (double) pow(TEN, power);
  return (double) pow((1.0L + (1.0L / power_of_ten)), power_of_ten);
}

long double calculate_euler_long_double(int power) {
  long double power_of_ten = (long double) pow(TEN, power);
  return (long double) pow((1.0L + (1.0L / power_of_ten)), power_of_ten);
}

