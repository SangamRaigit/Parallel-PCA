#include "t_io.h"
#include <stdlib.h>
#include <time.h>

/*
    Reads the matrix from the input file and stores it in 1D array D.
    The matrix dimensions M (rows) and N (columns) are also retrieved.
*/
void read_matrix(const char* input_filename, int* M, int* N, float** D) {
    FILE *fin = fopen(input_filename, "r");
    if (fin == NULL) {
        printf("Error: Unable to open file %s\n", input_filename);
        exit(EXIT_FAILURE);
    }

    // Reading the matrix dimensions M (rows) and N (columns)
    fscanf(fin, "%d %d", M, N);
    
    // Number of elements in the matrix
    int num_elements = (*M) * (*N);
    
    // Allocate memory for the matrix D as a 1D array
    *D = (float*) malloc(sizeof(float) * num_elements);
    if (*D == NULL) {
        printf("Error: Memory allocation failed\n");
        fclose(fin);
        exit(EXIT_FAILURE);
    }

    // Reading the matrix elements into the 1D array
    for (int i = 0; i < num_elements; i++) {
        fscanf(fin, "%f", (*D + i));
    }

    fclose(fin);
}

/*
    Writes the results after performing SVD and PCA. It outputs the original matrix, 
    the matrices U, SIGMA, V_T, the reduced matrix D_HAT, and the computation time.
*/
void write_result(int M, 
                  int N, 
                  float* D, 
                  float* U, 
                  float* SIGMA, 
                  float* V_T,
                  int K, 
                  float* D_HAT,
                  double computation_time) {
    // Output the original matrix D
    printf("Original Matrix D (%d x %d):\n", M, N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", D[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");

    // Output the matrix U
    printf("Matrix U (%d x %d):\n", N, N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", U[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");

    // Output the singular values SIGMA
    printf("Singular Values SIGMA (%d x 1):\n", N);
    for (int i = 0; i < N; i++) {
        printf("%f ", SIGMA[i]);
    }
    printf("\n\n");

    // Output the matrix V_T
    printf("Matrix V_T (%d x %d):\n", M, M);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            printf("%f ", V_T[i * M + j]);
        }
        printf("\n");
    }
    printf("\n");

    // Output the reduced matrix D_HAT
    printf("Reduced Matrix D_HAT (%d x %d):\n", M, K);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            printf("%f ", D_HAT[i * K + j]);
        }
        printf("\n");
    }
    printf("\n");

    // Output the computation time
    printf("Computation Time: %f seconds\n", computation_time);
}
