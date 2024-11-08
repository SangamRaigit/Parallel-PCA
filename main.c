#include "t_io.h"     // Include for read_matrix and write_result functions
#include "t_omp.h"    // Include for OpenMP-specific functionality
#include <stdlib.h>
#include <omp.h>
#include <stdio.h>

/*
	Arguments:
		arg1: input filename (consist M, N and D)
		arg2: retention (percentage of information to be retained by PCA)
*/

int main(int argc, char const *argv[])
{
    // Check the number of arguments
	if (argc < 3) {
		printf("\nLess Arguments\n");
		return 0;
	}

	if (argc > 3) {
		printf("\nTOO many Arguments\n");
		return 0;
	}

	//---------------------------------------------------------------------
	int M;			 // Number of rows (samples) in the input matrix D (input)
	int N;			 // Number of columns (features) in the input matrix D (input)
	float* D;		 // 1D array of M x N matrix to be reduced (input)
	float* U;		 // 1D array of N x N matrix U (to be computed by SVD)
	float* SIGMA;	 // 1D array of N x M diagonal matrix SIGMA (to be computed by SVD)
	float* V_T;		 // 1D array of M x M matrix V_T (to be computed by SVD)
	int K;			 // Number of columns (features) in the reduced matrix D_HAT (to be computed by PCA)
	float* D_HAT;	 // 1D array of M x K reduced matrix (to be computed by PCA)
	int retention;	 // Percentage of information to be retained by PCA (command-line input)
	//---------------------------------------------------------------------

	// Retention value (e.g., 90 means 90% of the information should be retained)
	retention = atoi(argv[2]);

	float start_time, end_time;
	double computation_time;

	/*
		-- Pre-defined function --
		Reads matrix and its dimensions from the input file and creates array D.
		The number of elements in D is M * N. 
        Format:
        --------------------------------------------------------------------------------------
        | D[0][0] | D[0][1] | ... | D[0][N-1] | D[1][0] | ... | D[1][N-1] | ... | D[M-1][N-1] |
        --------------------------------------------------------------------------------------
	*/
	read_matrix(argv[1], &M, &N, &D);   // Read the matrix from the file

	// Allocate memory for U, SIGMA, and V_T
	U = (float*) malloc(sizeof(float) * N * N);
	SIGMA = (float*) malloc(sizeof(float) * N);
	V_T = (float*) malloc(sizeof(float) * M * M);

	// Start timer
	start_time = omp_get_wtime();
	
	/*
		*****************************************************
			TODO -- You must implement these two functions:
			1. SVD: Singular Value Decomposition
			2. PCA: Principal Component Analysis
		*****************************************************
	*/
	SVD(M, N, D, &U, &SIGMA, &V_T);           // Perform Singular Value Decomposition
	PCA(retention, M, N, D, U, SIGMA, &D_HAT, &K);   // Perform Principal Component Analysis

	// End timer
	end_time = omp_get_wtime();
	computation_time = ((double)(end_time - start_time));   // Calculate computation time
	
	/*
		-- Pre-defined function --
		Checks the correctness of results computed by SVD and PCA
		and outputs the results.
	*/
	write_result(M, N, D, U, SIGMA, V_T, K, D_HAT, computation_time);

	// Free allocated memory
	free(U);
	free(SIGMA);
	free(V_T);
	free(D_HAT);
	free(D);

	return 0;
}
