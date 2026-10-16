# Repositório para as listas de exercícios do curso de matemática - bacharelado - da UESC

## Atividades desenvolvidas como parte da ementa do PPG em Modelagem Computacional

## Como criar os arquivos ".c" das questões
No terminal (Linux) ou no PowerShell (Windows) execute o arquivo `create_files.sh`.

Por exemplo:
- No Linux:
```sh
sh create_files.sh 
# ou ./create_files.sh
```
- No Windows:

## Como compilar e executar os códigos
Para compilar os códigos fonte utilize o arquivo `compile_file.sh` passando nome do arquivo, nome do executável e o caminho para a biblioteca como parâmetros.

Para executar os arquivos compilados, utilize o comando padrão de seu sistema operacional para lidar com isso.

Por exemplo:

- No Linux:
```sh
# para compilar: 
# shellscript       |     fonte .c   | executavel | caminho para biblioteca
sh compile_file         questao_1.c    questao_1        facil
# ou ./compile_file     questao_1.c    questao_1        facil

# para executar:
./arquivo_binario
```
- No Windows:
