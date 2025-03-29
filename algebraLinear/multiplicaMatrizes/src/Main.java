import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        Matriz matrizA = new Matriz(1, 3),
            matrizB = new Matriz(new int[][] {
                {3, 2, 1},
                {4, 5, 6},
                {9, 8, 7}
            }
        );

        System.out.println("Matriz A original:\n" + Arrays.deepToString(matrizA.getMatrix()) + "\n\n");
        System.out.println("Matriz B original:\n" + Arrays.deepToString(matrizB.getMatrix()) + "\n\n");
        matrizA.matrixMultiply(matrizB);
        System.out.println("Matriz A após multiplicação:\n" + Arrays.deepToString(matrizA.getMatrix()) + "\n\n");
    }
}