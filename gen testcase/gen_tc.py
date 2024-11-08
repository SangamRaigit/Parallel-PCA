#!/usr/bin/python3

#########################################################################
# Generate M x N matrix of real numbers and store                       #
# the matrix in a file named 'testcase_<M>_<N>'                         #
# Parameters:                                                           #
#   M               : Number of rows (samples) in the matrix             #
#   N               : Number of columns (features) in the matrix        #
#   lrange, urange  : Range of matrix elements ie                        #
#                       0 <= i < M, 0 <= j < N                          #
#                       lrange <= matrix[i][j] <= urange                #
# Format of output file:                                                #
#   -------------------------------------------------------------------  #
#   | M N                                                               #
#   | D[0][0] D[0][1] ... D[0][N-1]                                       #
#   | D[1][0] D[1][1] ... D[1][N-1]                                       #
#   | ...                                                               #
#   | D[M-1][0] D[M-1][1] ... D[M-1][N-1]                               #
#   -------------------------------------------------------------------  #
#########################################################################

from random import uniform

M = 10000           # number of rows (samples) in input matrix D
N = 300             # number of columns (features) in input matrix
lrange = -100000    # lower range for matrix element
urange = 100000     # upper range for matrix element

# Output filename based on M and N dimensions
filename = f'testcase_{M}_{N}.txt'

# Open file for writing
with open(filename, 'w') as file:
    # Write size of matrix (M N) in the first line
    file.write(f"{M} {N}\n")

    # Write matrix elements row by row
    for i in range(M):
        # Generate a row with N random numbers in the specified range
        row = [str(uniform(lrange, urange)) for _ in range(N)]
        
        # Join the elements with a space and write to file, followed by a newline
        file.write(" ".join(row) + "\n")

print(f"Matrix saved to {filename}")
