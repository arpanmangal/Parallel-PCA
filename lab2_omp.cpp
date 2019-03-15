#include <stdio.h>
#include <stdlib.h> 
#include <malloc.h>
#include <omp.h>

int min (int a, int b) {
    return (a < b) ? a : b;
}

void MatrixMultiply (double *A, double* B, double* C, int m, int n, int p, int parallelise=1)
{
    // C = A * B
    // dim (A) = m * n
    // dim (B) = n * p
    // dim (C) = m * p
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++) {
    //         printf("%f ", A[i*n+j]);
    //     }
    //     printf("\n");
    // }
    // for (int i = 0; i < n; i++) {
    //     for (int j = 0; j < p; j++) {
    //         printf("%f ", B[i*p+j]);
    //     }
    //     printf("\n");
    // }

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
        int BLOCK_SIZE = 8;
        #pragma omp parallel for collapse(2) schedule(dynamic)
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

void MatrixSubtract (double *A, double *B, double *C, int m, int n) {
    int size = m * n;
    #pragma omp parallel for schedule (dynamic, 64)
    for (int i = 0; i < size; i++)
        C[i] = A[i] - B[i];
}

double VectorNorm (double *A, int m) {
    double norm = 0;
    #pragma omp parallel for schedule (dynamic, 64) reduction (+ : norm)
    for (int i = 0; i < m; i++)
        norm += A[i] * A[i];

    return norm;
}

void ScalarDivide (double *A, double *N, int m, int n, double alpha) {
    int size = m * n;
    #pragma omp parallel for schedule (dynamic, 64)
    for (int i = 0; i < size; i++)
        N[i] = A[i] / alpha;
}

void makeIdenMatrix (double *I, int m) {
    printf("%d\n", m);
    #pragma omp parallel for schedule (dynamic, 64) collapse (2)
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            I[i*m + j] = (i == j);
}

void SVD(int M, int N, float* D, float** U, float** SIGMA, float** V_T)
{
    omp_set_num_threads(4);
    srand (0); // Deterministically Random

    // int m = 2758, n = 4002, p = 2920;
    int m = 1, n = 10000, p = 1;
    int I = 5;
    double *A = (double *) malloc (sizeof(double) * m * n);
    double *B = (double *) malloc (sizeof(double) * n * p);
    double *C = (double *) malloc (sizeof(double) * I * I);
    double *Cp = (double *) malloc (sizeof(double) * m * p); 

    for (int a = 0; a < m*n; a++) {
        A[a] = (rand() % 100000) / 100000.0;
        A[a] = 10;
    }
    for (int b = 0; b < n*p; b++) {
        B[b] = (rand() % 100000) / 100000.0;
        // B[b] = b;
    }
    
    // double start_time = omp_get_wtime();
    // MatrixMultiply (A, B, C, m, n, p, 0);
    // double end_time = omp_get_wtime();
    // printf("Seq time: %f\n", end_time - start_time);

    double start_time = omp_get_wtime();
    double norm = VectorNorm (A, n);
    // MatrixSubtract (A, A, A, m, n);
    // ScalarDivide (A, A, m, n, 10);
    makeIdenMatrix (C, I);
    double end_time = omp_get_wtime();
    for (int i = 0; i < I; i++) {
        for (int j = 0; j < I; j++) 
            printf("%f ", C[i*I+j]);
        printf("\n");
    }
    printf ("%f\n", norm);
    MatrixMultiply (A, A, Cp, m, n, p, 0);
    printf("Par time: %f\n", end_time - start_time);

    for (int c = 0; c < m*p; c++) {
        printf("%f ", Cp[c]);
        // if (abs(C[c] - Cp[c]) > 1e-5) {
        //     printf("Not Same!! c=%d | C=%f | Cp=%f \n", c, C[c], Cp[c]);
        //     break;
        // }
    }
}

void PCA(int retention, int M, int N, float* D, float* U, float* SIGMA, float** D_HAT, int *K)
{
    
}
