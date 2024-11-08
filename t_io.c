#include "t_io.h"
#include <stdio.h>
#include <stdlib.h>

void read_matrix (const char* input_filename, int* M, int* N, float** D) {
    FILE *fin = fopen(input_filename, "r");
    if (fin == NULL) {
        printf("Error: Unable to open file %s\n", input_filename);
        exit(1);
    }

    // Read dimensions of matrix M and N
    fscanf(fin, "%d %d", M, N);
    
    // Allocate memory for matrix D (M x N)
    int num_elements = (*M) * (*N);
    *D = (float*) malloc(sizeof(float) * num_elements);
    
    // Read elements of matrix D
    for (int i = 0; i < num_elements; i++) {
        fscanf(fin, "%f", (*D + i));
    }

    fclose(fin);
}

void write_result(int M, 
                  int N, 
                  float* D, 
                  float* U, 
                  float* SIGMA, 
                  float* V_T, 
                  int K, 
                  float* D_HAT, 
                  double computation_time) {
    printf("Matrix Dimensions: M = %d, N = %d\n", M, N);
    
    // Print the original matrix D (MxN)
    printf("\nOriginal Matrix (D):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", D[i * N + j]);
        }
        printf("\n");
    }

    // Print the left singular matrix U (NxN)
    printf("\nLeft Singular Matrix (U):\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", U[i * N + j]);
        }
        printf("\n");
    }

    // Print the singular values (SIGMA)
    printf("\nSingular Values (SIGMA):\n");
    for (int i = 0; i < N; i++) {
        printf("%f ", SIGMA[i]);
    }
    printf("\n");

    // Print the right singular matrix V_T (MxM)
    printf("\nRight Singular Matrix (V_T):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            printf("%f ", V_T[i * M + j]);
        }
        printf("\n");
    }

    // Print the reduced matrix D_HAT (MxK)
    printf("\nReduced Matrix (D_HAT):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            printf("%f ", D_HAT[i * K + j]);
        }
        printf("\n");
    }

    // Print the computation time
    printf("\nComputation Time: %f seconds\n", computation_time);
}
