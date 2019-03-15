#include <stdio.h>
#include <stdlib.h> 
#include <malloc.h>
#include <omp.h>

int min (int a, int b) {
    return (a < b) ? a : b;
}

void MatrixMultiply (double *A, double* B, double* C, int m, int n, int p, int parallelise=0)
{
    // C = A * B
    // dim (A) = m * n
    // dim (B) = n * p
    // dim (C) = m * p
    if (parallelise == 0) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < p; j++) {
                C[i*p + j] = 0.0;
                for (int k = 0; k < n; k++) {
                    C[i*p + j] += A[i*n + k] * B[k*p + j];
                }
            }
        }
    } else {
        // Parallelise the algorithm => Blocked Matrix Multiplication Algorithm
        int BLOCK_SIZE = 64;
        omp_set_num_threads(4);
        #pragma omp parallel for collapse(2) schedule(static, BLOCK_SIZE)
        for (int I = 0; I < m; I += BLOCK_SIZE) {
            for (int J = 0; J < p; J += BLOCK_SIZE) {
                int i_max = min (m, I + BLOCK_SIZE);
                int j_max = min (p, J + BLOCK_SIZE);

                for (int i = I; i < i_max; i++) {
                    for (int j = J; j < j_max; j++) {
                        C[i*p + j] = 0.0;
                    }
                }         
                
                for (int K = 0; K < n; K += BLOCK_SIZE) {
                    int k_max = min (n, K + BLOCK_SIZE);
                    for (int i = I; i < i_max; i++) {
                        for (int j = J; j < j_max; j++) {
                            for (int k = K; k < k_max; k++) {
                                C[i*p + j] += A[i*n + k] * B[k*p + j];
                            }
                        }
                    }
                }
            }
        }
    }
}

void SVD(int M, int N, float* D, float** U, float** SIGMA, float** V_T)
{
    srand (0); // Deterministically Random

    int m = 758, n = 4002, p = 920;
    double *A = (double *) malloc (sizeof(double) * m * n);
    double *B = (double *) malloc (sizeof(double) * n * p);
    double *C = (double *) malloc (sizeof(double) * m * p);
    double *Cp = (double *) malloc (sizeof(double) * m * p); 

    for (int a = 0; a < m*n; a++) {
        A[a] = (rand() % 100000) / 100000.0;
    }
    for (int b = 0; b < n*p; b++) {
        B[b] = (rand() % 100000) / 100000.0;
    }
    double start_time = omp_get_wtime();
    MatrixMultiply (A, B, C, m, n, p, 0);
    double end_time = omp_get_wtime();
    printf("Seq time: %f\n", end_time - start_time);

    start_time = omp_get_wtime();
    MatrixMultiply (A, B, Cp, m, n, p, 1);
    end_time = omp_get_wtime();
    printf("Par time: %f\n", end_time - start_time);

    for (int c = 0; c < m*p; c++) {
        if (abs(C[c] - Cp[c]) > 1e-5) {
            printf("Not Same!! c=%d | C=%f | Cp=%f \n", c, C[c], Cp[c]);
            break;
        }
    }
}

void PCA(int retention, int M, int N, float* D, float* U, float* SIGMA, float** D_HAT, int *K)
{
    
}
