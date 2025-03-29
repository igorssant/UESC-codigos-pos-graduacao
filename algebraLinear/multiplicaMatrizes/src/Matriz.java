import java.lang.Math;
import java.util.Arrays;

public class Matriz {
    private int linhas;
    private int colunas;
    private int[][] matriz;

    public Matriz(int linhas) {
        this.linhas = linhas;
        this.colunas = linhas;
        createIdentityMatrix();
    }

    public Matriz(int[][] matriz) {
        this.matriz = matriz;
        this.linhas = this.matriz.length;
        this.colunas = this.matriz[0].length;
    }

    public Matriz(int[][] valores, int linhas) {
        if(!this.isItSquare(valores.length, linhas) || !this.isItSquare(valores[0].length, linhas)) {
            throw new RuntimeException(
                "A matriz formada não é uma matriz quadrada.\n" +
                "Os valores passados como parâmetros não formam uma matriz quadrada."
            );
        }

        this.linhas = linhas;
        this.colunas = linhas;
        this.populateMatrix(valores);
    }

    public Matriz(int fill, int linhas) {
        this.linhas = linhas;
        this.colunas = linhas;
        this.populateMatrix(fill);
    }

    public Matriz(int[][] valores, int linhas, int colunas) {
        this.linhas = linhas;
        this.colunas = colunas;
        this.populateMatrix(valores);
    }

    public Matriz(int fill, int linhas, int colunas) {
        this.linhas = linhas;
        this.colunas = colunas;
        this.populateMatrix(fill);
    }

    public void matrixSum(Matriz matrizB) {
        if(this.linhas != matrizB.getQuantityOfLines() || this.colunas != matrizB.getQuantityOfColumns()) {
            throw new RuntimeException("Erro.\nAs matrizes possuem dimensões diferentes.");
        }

        for(int i = 0; i < this.linhas; i++) {
            for(int j = 0; j < this.colunas; j++) {
                this.matriz[i][j] = matrizB.getValue(i, j);
            }
        }
    }

    public void matrixMultiplyByScalar(int scalar) {
        for(int i = 0; i < this.linhas; i++) {
            for(int j = 0; j < this.linhas; j++) {
                this.matriz[i][j] *= scalar;
            }
        }
    }

    public void matrixMultiply(Matriz matrizB) {
        int[][] newMatrix;

        if(this.colunas != matrizB.getQuantityOfLines()) {
            throw new RuntimeException(
                "Erro.\n" +
                "A quantidade de linhas da matriz A é diferente da quantidade de colunas da matriz B."
            );
        }

        newMatrix = new int[this.linhas][matrizB.getQuantityOfColumns()];

        for(int k = 0; k < this.linhas; k++) {
            for(int i = 0; i < this.linhas; i++) {
                for(int j = 0; j < matrizB.getQuantityOfColumns(); j++) {
                    newMatrix[k][i] += this.matriz[k][j] * matrizB.getValue(j, i);
                }
            }
        }

        this.matriz = newMatrix;
    }

    public int getValue(int linha, int coluna) {
        return this.matriz[linha][coluna];
    }

    public int getQuantityOfLines() {
        return this.linhas;
    }

    public int getQuantityOfColumns() {
        return this.colunas;
    }

    public int[][] getMatrix() {
        return this.matriz;
    }

    private Boolean isItSquare(int quantidadeDeValores, int quantidadeDeLinhas) {
        return quantidadeDeValores == (int) Math.pow(quantidadeDeLinhas, 2);
    }

    private void populateMatrix(int[][] valores) {
        this.matriz = new int[this.linhas][this.colunas];

        for(int i = 0; i < this.linhas; i++) {
            System.arraycopy(valores[i], 0, this.matriz[i], 0, this.colunas);
        }
    }

    private void populateMatrix(int value) {
        this.matriz = new int[this.linhas][this.colunas];

        for(int[] linha : matriz) {
            Arrays.fill(linha, value);
        }
    }

    private void createIdentityMatrix() {
        for(int i = 0; i < this.linhas; i++) {
            for(int j = 0; j < this.colunas; j++) {
                this.matriz[i][j] = (i == j)? 1 : 0;
            }
        }
    }

    private int[] getLine(int index) {
        return this.matriz[index];
    }

    private int[] getColumn(int index) {
        int[] column = new int[this.colunas];

        for(int i = 0; i < this.linhas; i++) {
            column[i] = matriz[i][index];
        }

        return column;
    }
}
