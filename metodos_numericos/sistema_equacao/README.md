# Sistemas de Equações Lineares — Métodos Numéricos

Este projeto explora a resolução de sistemas de equações lineares utilizando métodos numéricos diretos e iterativos, com foco em análise de condicionamento, precisão, convergência e desempenho computacional.

## Estrutura do Projeto

- `src/`: Notebooks e scripts principais para experimentos e análise.
- `output/`: Resultados gerados (csv, gráficos).
- `dev_files/`: Materiais de apoio e referências.
- `environment.yml`: Ambiente Conda para reprodução dos experimentos.

## Descrição dos Arquivos do Projeto

A seguir está uma explicação detalhada dos principais arquivos e pastas do projeto:

### src/

- **main.ipynb**
  Notebook principal do projeto. Contém toda a lógica de experimentação, análise, execução dos métodos numéricos, geração de gráficos e exportação dos resultados. É o ponto de partida para reproduzir todos os experimentos do trabalho.

- **methods/**
  Pasta que contém os módulos Python com as implementações dos métodos numéricos:
  - **direct_methods.py**: Implementa os métodos diretos para resolução de sistemas lineares, como eliminação de Gauss (simples, com pivotamento parcial, escalonado e completo) e fatoração LU.
  - **iterative_methods.py**: Implementa os métodos iterativos, incluindo Jacobi, Gauss-Seidel e métodos de relaxamento (SOR).

- **utils/**
  Pasta com funções auxiliares:
  - **matrix.py**: Funções para geração de matrizes especiais (como a matriz de Hilbert) e outras utilidades para manipulação de matrizes.

### output/

- **gauss_testes.csv**
  Tabela com os resultados dos métodos diretos: tempo de execução, erro absoluto e relativo para cada método e tamanho de matriz.

- **gauss_solucao.csv**
  Soluções dos sistemas lineares obtidas pelos métodos diretos, para cada valor de n.

- **iterativos_testes.csv**
  Tabela com os resultados dos métodos iterativos: tempo de execução, erro absoluto e relativo para cada método e tamanho de matriz.

- **iterativos_solucao.csv**
  Soluções dos sistemas lineares obtidas pelos métodos iterativos, para cada valor de n.

- **convergencia_hilbert_n{n}_tol{tol}.png**
  Gráficos de convergência dos métodos iterativos para cada valor de n e tolerância utilizada.

- **media_iterativos.csv**
  Tabela com a média dos tempos de execução e erros relativos dos métodos iterativos, removendo o maior e menor valor de cada conjunto de execuções.

### dev_files/

- Materiais de apoio, anotações, PDFs de referência e outros arquivos utilizados durante o desenvolvimento do projeto.

### environment.yml

Arquivo de configuração do ambiente Conda, listando todas as dependências necessárias para executar o projeto.

## Principais Funcionalidades

- **Geração de matrizes de Hilbert** para análise de condicionamento.
- **Resolução de sistemas** por métodos diretos:
  - Eliminação de Gauss (simples, pivotamento parcial, escalonado, completo)
  - Fatoração LU
- **Resolução por métodos iterativos**:
  - Jacobi
  - Gauss-Seidel
  - Relaxamento (SOR)
- **Cálculo de determinante, número de condição e inversa** das matrizes.
- **Avaliação de precisão** (erro absoluto e relativo), convergência e tempo de execução.
- **Exportação de resultados** para arquivos CSV e geração de gráficos.

## Como Executar

1. **Crie o ambiente Conda:**

   ```sh
   conda env create -f environment.yml
   conda activate mn
   ```

2. **Abra os notebooks em `src/` no VS Code ou Jupyter.**
   - Execute as células para reproduzir os experimentos e gerar os resultados.

## Dependências

Veja [environment.yml](environment.yml) para detalhes. Principais pacotes:

- Python 3.13
- numpy, pandas, sympy
- matplotlib, seaborn
- ipykernel, nbformat

## Resultados

Os resultados dos experimentos (soluções, erros, gráficos de convergência) são salvos em [output/](output/).

## Exemplos de Saída

- Tabelas de erros absolutos e relativos para cada método e tamanho de matriz.
- Gráficos de convergência dos métodos iterativos.
- Arquivos CSV com os resultados das execuções.

## Referências

- Métodos Numéricos para Engenharia — CAP 257
- Algebra Linear e suas Aplicações — Petronio Pulino

---

> Projeto acadêmico para estudo de métodos numéricos aplicados à resolução de sistemas lineares.
