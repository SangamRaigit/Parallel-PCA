#include <malloc.h>
#include <omp.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

#define TOLERANCE 1e-3

// Function to calculate the L2 norm of a matrix difference (used for convergence checking)
float matrix_similarity(float *M_1, int m, int n, float *M_2)
{
    float l2_diff = 0.0;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            l2_diff += (M_1[i * n + j] - M_2[i * n + j]) * (M_1[i * n + j] - M_2[i * n + j]);
        }
    }
    return sqrtf(l2_diff);
}

// Function to transpose a matrix
void transpose(float *M, int m, int n, float *M_T)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            M_T[j * m + i] = M[i * n + j];
        }
    }
}

// Matrix multiplication
void multiply(float *M_1, int m1, int n1, float *M_2, int m2, int n2, float *result)
{
    assert(n1 == m2);
    float sum = 0.0;
    float *M_2_T = (float *)malloc(sizeof(float) * n2 * m2);
    transpose(M_2, m2, n2, M_2_T);

    for (int i = 0; i < m1; i++)
    {
        for (int j = 0; j < n2; j++)
        {
            sum = 0.0;
            for (int k = 0; k < n1; k++)
            {
                sum += M_1[i * n1 + k] * M_2_T[j * m2 + k];
            }
            result[i * n2 + j] = sum;
        }
    }
    free(M_2_T);
}

// Function to initialize an identity matrix
float *initialize_identity(int size)
{
    float *I = (float *)calloc(size * size, sizeof(float));
    for (int i = 0; i < size; i++)
    {
        I[i * size + i] = 1.0;
    }
    return I;
}

// Function to calculate the L2 norm of a vector
float l2_norm(float *v_col, int length)
{
    float sum_sq = 0.0;
    for (int i = 0; i < length; i++)
    {
        sum_sq += v_col[i] * v_col[i];
    }
    return sqrtf(sum_sq);
}

// Function to check convergence based on tolerance
inline bool convergence(float diff_norm)
{
    return diff_norm < TOLERANCE;
}

// Classical Gram-Schmidt (CGS) process for QR factorization
void classicalGS(float *A_current, float *A_T, int P, float *Q_current, float *R_current)
{
    float *v_col = (float *)malloc(sizeof(float) * P);
    for (int col = 0; col < P; col++)
    {
        memcpy(v_col, A_T + col * P, sizeof(float) * P);
        for (int row = 0; row < col; row++)
        {
            float result = 0.0;
            for (int row_ = 0; row_ < P; row_++)
            {
                result += Q_current[row_ * P + row] * A_T[col * P + row_];
            }
            R_current[row * P + col] = result;
            for (int row_ = 0; row_ < P; row_++)
            {
                v_col[row_] -= R_current[row * P + col] * Q_current[row_ * P + row];
            }
        }
        R_current[col * P + col] = l2_norm(v_col, P);
        for (int row = 0; row < P; row++)
        {
            Q_current[row * P + col] = v_col[row] / R_current[col * P + col];
        }
    }
    free(v_col);
}

// Function to compute V_T for SVD
void compute_V(float **SIGMA, float *D_T, float **U, float **V_T, int N, int P)
{
    float *INV_SIGMA = (float *)calloc(N * P, sizeof(float));
    for (int i = 0; i < P; i++)
    {
        INV_SIGMA[i * P + i] = 1.0 / (*SIGMA)[i];
    }

    float *U_T = (float *)malloc(sizeof(float) * P * P);
    transpose(*U, P, P, U_T);

    float *product = (float *)malloc(sizeof(float) * N * P);
    multiply(INV_SIGMA, N, P, U_T, P, P, product);
    multiply(product, N, P, D_T, P, N, *V_T);

    free(INV_SIGMA);
    free(U_T);
    free(product);
}

// Singular Value Decomposition (SVD) function
void SVD(int N, int P, float *D, float **U, float **SIGMA, float **V_T)
{
    float *D_T = (float *)malloc(sizeof(float) * P * N);
    transpose(D, N, P, D_T);
    float *A = (float *)calloc(P * P, sizeof(float));
    float *A_T = (float *)calloc(P * P, sizeof(float));
    multiply(D_T, P, N, D, N, P, A);

    float *A_current = (float *)malloc(sizeof(float) * P * P);
    memcpy(A_current, A, sizeof(float) * P * P);
    float *E_current = initialize_identity(P);
    float *Q_ = (float *)malloc(sizeof(float) * P * P);
    float *R_ = (float *)malloc(sizeof(float) * P * P);
    float diff_norm;

    int iter = 1;
    float *v_col = (float *)malloc(sizeof(float) * P);
    float sum_sq = 0.0;
    float *A_next = (float *)malloc(sizeof(float) * P * P);
    float *E_next = (float *)malloc(sizeof(float) * P * P);

    do
    {
        transpose(A_current, P, P, A_T);
        classicalGS(A_current, A_T, P, Q_, R_);

        A_next = (float *)malloc(sizeof(float) * P * P);
        E_next = (float *)malloc(sizeof(float) * P * P);
        multiply(R_, P, P, Q_, P, P, A_next);
        multiply(E_current, P, P, Q_, P, P, E_next);

        diff_norm = l2_norm_diagonal_diff(A_next, A_current, P);

        free(A_current);
        free(E_current);
        A_current = A_next;
        E_current = E_next;

    } while (diff_norm > TOLERANCE);

    float temp = FLT_MAX;
    for (int i = 0; i < P; i++)
    {
        (*SIGMA)[i] = sqrtf(A_current[i * P + i]);
        if ((*SIGMA)[i] > temp)
        {
            exit(0);
        }
        temp = (*SIGMA)[i];
    }

    for (int i = 0; i < P; i++)
    {
        for (int j = 0; j < P; j++)
        {
            (*U)[i * P + j] = E_current[i * P + j];
        }
    }

    float *temp_sigma = (float *)calloc(P * N, sizeof(float));
    for (int i = 0; i < P; i++)
    {
        temp_sigma[i * N + i] = (*SIGMA)[i];
    }

    compute_V(SIGMA, D_T, U, V_T, N, P);

    free(temp_sigma);
}

// Principal Component Analysis (PCA)
void PCA(int retention, int N, int P, float *D, float *U, float *SIGMA, float **D_HAT, int *K)
{
    float sum_eigenvalues = 0.0;
    for (int i = 0; i < P; i++)
    {
        sum_eigenvalues += SIGMA[i];
    }

    *K = 0;
    float retention_ = 0.0;
    int i = 0;
    while ((retention_ < retention) && (i < P))
    {
        retention_ += SIGMA[i] / sum_eigenvalues;
        (*K)++;
        i++;
    }

    *D_HAT = (float *)malloc(sizeof(float) * N * (*K));
    multiply(D, N, P, U, P, *K, *D_HAT);
}
