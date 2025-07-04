# Metodologia de Resolução de Sistemas Lineares

## 1. Descrição do Problema
Um sistema de equações lineares é um conjunto de equações que envolvem variáveis lineares. O objetivo é encontrar os valores das variáveis que satisfazem todas as equações simultaneamente. Esses sistemas podem ser representados na forma matricial Ax = b, onde A é a matriz dos coeficientes, x é o vetor das variáveis e b é o vetor dos termos independentes.

## 2. Verificação da condição do sistema

Sistemas mal condicionados são extremamente sensíveis a pequenas variações nos dados de entrada. Isso significa que uma pequena mudança nos dados pode levar a grandes mudanças na solução do sistema. Isso pode ocorrer em sistemas lineares e não lineares, e é uma característica importante a ser considerada ao resolver problemas numéricos.

Uma interpretação alternativa do mal condicionamento é que uma gama ampla de respostas pode satisfazer aproximadamente as
equações.

O condicionamento de um sistema linear pode ser verificado através de três métodos principais:

- a) Pequenas mudanças nos coeficientes da matriz A e no vetor b, e observar a mudança na solução x.

    Para verificar a condição de um sistema linear, podemos fazer pequenas perturbações nos coeficientes da matriz A e no vetor b, e observar como isso afeta a solução x. Se pequenas mudanças em A ou b resultarem em grandes mudanças em x, o sistema é considerado mal condicionado.

- b) Determinante da matriz A

    O determinante de uma matriz é uma medida de quão bem condicionada ela é. Se o determinante for próximo de zero, a matriz é considerada mal condicionada. Isso significa que pequenas mudanças nos dados de entrada podem levar a grandes mudanças na solução do sistema.

- c) Número de condição da matriz A

    O número de condição de uma matriz é uma medida de quão sensível a solução do sistema é a pequenas mudanças nos dados de entrada. Um número de condição alto indica que o sistema é mal condicionado, enquanto um número de condição baixo indica que o sistema é bem condicionado. O número de condição pode ser calculado usando a norma da matriz e sua inversa.

## 3. Verifica critérios de convergência

Matriz com diagonal dominante:
- Uma matriz é dita diagonalmente dominante se, para cada linha, o valor absoluto do elemento da diagonal é maior ou igual à soma dos valores absolutos dos outros elementos da linha. Isso garante que o método de Gauss-Seidel converja.

## 4. Métricas de Avaliação

a) Calculo do erro verdadeiro para verificar a precisão do sistema de equações lineares.

b) Convergência (para métodos iterativos):
 - Número de iterações até convergir
 - Evolução do erro por iteração (pode ser plotado)

c) Desempenho computacional:
 - Tempo de execução de cada método
 - Número de operações aproximadas (em teoria ou medindo com time)
