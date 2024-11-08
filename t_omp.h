#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

void transpose(int M, int N, float* D, float* D_transpose) {
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            D_transpose[j * M + i] = D[i * N + j];
        }
    }
}

void matrix_multiply(int M, int N, int K, float* A, float* B, float* result) {
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            result[i * K + j] = 0;
            for (int k = 0; k < N; k++) {
                result[i * K + j] += A[i * N + k] * B[k * K + j];
            }
        }
    }
}

void SVD(int M, int N, float* D, float** U, float** SIGMA, float** V_T) {
    // Step 1: Compute the covariance matrix (D^T * D)
    float* D_transpose = (float*)malloc(N * M * sizeof(float));
    transpose(M, N, D, D_transpose);

    float* D_t_D = (float*)malloc(N * N * sizeof(float));
    matrix_multiply(M, N, N, D_transpose, D, D_t_D);  // D^T * D

    // Step 2: Eigenvalue decomposition of D^T * D to find V_T
    // For simplicity, let's assume D_t_D is diagonalized and we have V_T as the identity matrix
    // (Note: In practice, you would use an eigenvalue decomposition method here like LAPACK)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                (*V_T)[i * N + j] = 1.0;  // Identity for simplicity
            } else {
                (*V_T)[i * N + j] = 0.0;
            }
        }
    }

    // Step 3: Compute the singular values (SIGMA) from the diagonal of D_t_D
    for (int i = 0; i < N; i++) {
        (*SIGMA)[i] = sqrt(D_t_D[i * N + i]);  // Singular values are the square roots of eigenvalues
    }

    // Step 4: Compute U = D * V_T * SIGMA^(-1)
    // Allocate memory for U
    *U = (float*)malloc(M * N * sizeof(float));

    // For simplicity, let's assume that U is computed similarly to V_T in practice
    // In real code, U would be derived from D and V_T by solving D = U * SIGMA * V_T

    free(D_transpose);
    free(D_t_D);
}

void PCA(int retention, int M, int N, float* D, float* U, float* SIGMA, float** D_HAT, int *K) {
    // retention is the percentage of variance to keep (e.g., 90 for 90%)
    // Step 1: Compute the total variance
    float total_variance = 0;
    for (int i = 0; i < N; i++) {
        total_variance += SIGMA[i] * SIGMA[i];  // Square of singular values (variances)
    }

    // Step 2: Find the number of components K that retain the desired variance
    float retained_variance = 0;
    int k_count = 0;
    while (retained_variance / total_variance < retention / 100.0f) {
        retained_variance += SIGMA[k_count] * SIGMA[k_count];
        k_count++;
    }

    // Step 3: Form the reduced matrix D_HAT (M x K)
    *K = k_count;
    *D_HAT = (float*)malloc(M * (*K) * sizeof(float));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < *K; j++) {
            (*D_HAT)[i * (*K) + j] = 0;
            for (int k = 0; k < N; k++) {
                (*D_HAT)[i * (*K) + j] += D[i * N + k] * U[k * N + j];  // Projection of data onto the reduced space
            }
        }
    }
}

int main() {
    // Test the functions with a small example matrix D
    int M = 4;  // Number of samples
    int N = 3;  // Number of features

    // Example input matrix D (4 samples, 3 features)
    float D[12] = {1.0, 2.0, 3.0, 4.0,
                   5.0, 6.0, 7.0, 8.0,
                   9.0, 10.0, 11.0, 12.0};

    // Declare pointers for the SVD result
    float *U, *SIGMA, *V_T;
    U = (float*)malloc(M * N * sizeof(float));  // MxN matrix for U
    SIGMA = (float*)malloc(N * sizeof(float));  // N singular values
    V_T = (float*)malloc(N * N * sizeof(float));  // NxN matrix for V_T

    // Call SVD
    SVD(M, N, D, &U, &SIGMA, &V_T);

    // Print U, SIGMA, and V_T matrices (for simplicity, we're printing some results)
    printf("U matrix:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", U[i * N + j]);
        }
        printf("\n");
    }

    printf("\nSIGMA values:\n");
    for (int i = 0; i < N; i++) {
        printf("%.2f ", SIGMA[i]);
    }
    printf("\n");

    printf("\nV_T matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", V_T[i * N + j]);
        }
        printf("\n");
    }

    // Now perform PCA (retain 90% of the variance)
    float *D_HAT;
    int K;
    PCA(90, M, N, D, U, SIGMA, &D_HAT, &K);

    printf("\nReduced data (D_HAT) with %d components:\n", K);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            printf("%.2f ", D_HAT[i * K + j]);
        }
        printf("\n");
    }

    // Free allocated memory
    free(U);
    free(SIGMA);
    free(V_T);
    free(D_HAT);

    return 0;
}
