#include <stdio.h>
#include <stdlib.h> 
#include <malloc.h>
#include <omp.h>
#include <math.h>
#include <vector>
#include <algorithm> // Used for sorting code

int min (int a, int b) {
    return (a < b) ? a : b;
}

void MatrixMultiply (float *A, float* B, float* C, int m, int n, int p, int parallelise=1)
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

void MatrixSubtract (float *A, float *B, float *C, int m, int n) {
    int size = m * n;
    #pragma omp parallel for schedule (dynamic, 64)
    for (int i = 0; i < size; i++)
        C[i] = A[i] - B[i];
}

float VectorNorm (float *A, int m) {
    float norm = 0;
    #pragma omp parallel for schedule (dynamic, 64) reduction (+:norm)
    for (int i = 0; i < m; i++)
        norm += A[i] * A[i];

    return sqrt(norm);
}

float InnerProduct (float *A, float *B, int m) {
    float IP = 0;
    #pragma omp parallel for schedule (dynamic, 64) reduction (+:IP)
    for (int i = 0; i < m; i++)
        IP += A[i] * B[i];

    return IP;
}

void ScalarMultiply (float *A, float *B, int m, int n, float alpha) {
    int size = m * n;
    #pragma omp parallel for schedule (dynamic, 64)
    for (int i = 0; i < size; i++)
        B[i] = A[i] * alpha;
}

void makeIdenMatrix (float *I, int m) {
    printf("%d\n", m);
    #pragma omp parallel for schedule (dynamic, 64) collapse (2)
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            I[i*m + j] = (i == j);
}

void MatrixTranspose (float *A, float *B, int m, int n) {
    #pragma omp parallel for schedule (dynamic, 64) collapse (2)
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            B[j*m + i] = A[i*n + j];
}

void MatrixAssign (float *A, float *B, int m, int n) {
    int size = m * n;
    #pragma omp parallel for schedule (dynamic, 64)
    for (int i = 0; i < size; i++)
        B[i] = A[i];
}

void MatrixProj (float *A, float *U, float *P, int N) {
    // All N * 1 Matrix
    // Project A on U
    float factor = InnerProduct (U, A, N) / InnerProduct (U, U, N);
    ScalarMultiply (U, P, N, 1, factor);
}

void makeTriangular (float *M, int N, int upper=1) {
    // Make the matrix triangular
    #pragma omp parallel for schedule (dynamic, 64) collapse (2)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if ((upper == 1 && j < i) || (upper == 0 && j > i))
                M[i*N + j] = 0;
}

void printMatrix (float *M, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", M[i*n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void GramSchmidt (float *A, int N, float *At, float *u, float *e, float *p, float *Q, float *R) {
    // Just allocate At, u, e matrices to be N*N; p matrix to be N * 1
    
    MatrixTranspose (A, At, N, N);

    for (int k = 0; k < N; k++) {
        MatrixAssign (At + k*N, u + k*N, N, 1);
        for (int j = 0; j < k; j++) {
            MatrixProj (At + k*N, u + j*N, p, N);
            MatrixSubtract (u + k*N, p, u + k*N, N, 1);
        }

        float uNorm = VectorNorm (u + k*N, N);
        ScalarMultiply (u + k*N, e + k*N, N, 1, 1 / uNorm);
    }

    MatrixTranspose (e, Q, N, N);
    MatrixMultiply (e, A, R, N, N, N);
    // printMatrix(R, N, N);
    makeTriangular (R, N, 1);
}

void QR (float *D, float *lD, int N, float *Evals, float *E) {
    MatrixAssign (D, lD, N, N);
    makeIdenMatrix (E, N);

    float *At = (float *) malloc (sizeof(float) * N * N);
    float *u = (float *) malloc (sizeof(float) * N * N);
    float *e = (float *) malloc (sizeof(float) * N * N);
    float *p = (float *) malloc (sizeof(float) * N);
    float *Q = (float *) malloc (sizeof(float) * N * N);
    float *R = (float *) malloc (sizeof(float) * N * N);
    float *E_next = (float *) malloc (sizeof(float) * N * N);

    for (int i = 0; i < 10000; i++) {
        // printMatrix (E, N, N);
        GramSchmidt (lD, N, At, u, e, p, Q, R);
        // printMatrix (Q, N, N);
        // printMatrix (R, N, N);
        MatrixMultiply (R, Q, lD, N, N, N);
        MatrixMultiply (E, Q, E_next, N, N, N);
        MatrixAssign (E_next, E, N, N);
    }
    // printMatrix (lD, N, N);
    // printMatrix (E, N, N);

    #pragma omp parallel for schedule (dynamic, 64)
    for (int i = 0; i < N; i++) {
        Evals[i] = lD[i*N+i];
    }

    // printMatrix (Evals, 1, N);
    free(At);
    free(u);
    free (e);
    free(p);
    free(Q);
    free(R);
}

bool sortPair (std::pair<float, int> &a, std::pair<float, int> &b)
{
    return (a.first > b.first);
}
void Decompose (float * Dt, float *E, float* E_vals, int M, int N, float *U, float *SIGMA, float *V_T)
{
    // D is M*M, U is N*N, SIGMA => N, V_T => M*M
    double start_time = omp_get_wtime();
    std::vector<std::pair<float, int>> EigenVals;
    for (int i = 0; i < M; i++) {
        EigenVals.push_back(std::make_pair(sqrt(abs(E_vals[i])), i));
    }
    std::sort (EigenVals.begin(), EigenVals.end(), sortPair);

    printf("$$$\n");
    for (int i = 0; i < M; i++) {
        printf("%f ", EigenVals[i].first);
    }
    printf ("\n");
    
    printMatrix (E, M, M);
    MatrixTranspose (E, V_T, M, M);
    printMatrix (V_T, M, M);
    MatrixAssign (V_T, E, M, M);

    // Possible parallelisation => no
    for (int i = 0; i < M; i++) {
        int rank = EigenVals[i].second;
        MatrixAssign (E + rank*N, V_T + i*M, M, 1);
        if (i < N)
            SIGMA [i] = EigenVals[i].first;
    }

    printMatrix (V_T, M, M);
    printMatrix (SIGMA, N, 1);

    float *SigmaInvMatrix = (float *) malloc (sizeof(float) * M * N);
    float *VSigmaInvMatrix = (float *) malloc (sizeof(float) * M * N);

    #pragma omp parallel for schedule (dynamic, 64) collapse (2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            SigmaInvMatrix [i*N + j] = (i == j) * (1 / SIGMA[j]);
        }
    }
    printMatrix (SigmaInvMatrix, M, N);
    MatrixTranspose (V_T, E, M, M);
    printMatrix (E, M, M);
    MatrixMultiply (E, SigmaInvMatrix, VSigmaInvMatrix, M, M, N);

    MatrixMultiply (Dt, VSigmaInvMatrix, U, N, M, N);
    printMatrix (U, N, N);
    printMatrix (SIGMA, N, 1);
    printMatrix (V_T, M, M);
    exit(0);

    double end_time = omp_get_wtime();
    printf("Decompose time: %f\n", end_time - start_time);
}

void SVD(int M, int N, float* D, float** U, float** SIGMA, float** V_T)
{
    omp_set_num_threads(4);
    srand (0); // Deterministically Random

    // N = 3, M = 5;
    // float a[] = {12, -51, 4, 6, 167, -68, -4, 24, -41, 10, 23, 10, 78, 90, 13};
    // // float a[] = {-2, -4, 2, -2, 1, 2, 4, 2, 5};
    // // float a[] = {24, -15, -15, 25};
    // float *A = (float *) malloc (sizeof(float) * M * N);
    // for (int i = 0; i < M*N; i++)
    //     A[i] = a[i];
    // float *B = (float *) malloc (sizeof(float) * N * M);

    // MatrixTranspose (A, B, M, N);
    // printMatrix (A, M, N);
    // printMatrix (B, N, M);
    // return;

    // N = 2;
    // M = 2;
    // // float a[] = {12, -51, 4, 6, 167, -68, -4, 24, -41};
    // // float a[] = {-2, -4, 2, -2, 1, 2, 4, 2, 5};
    // float a[] = {25, -15, -15, 25};
    // float *A = (float *) malloc (sizeof(float) * N * N);

    // for (int i = 0; i < N*N; i++)
    //     A[i] = a[i];

    // float *E_vals = (float *) malloc (sizeof(float) * N);
    // float *E = (float *) malloc (sizeof(float) * N * N);
    // float *lD = (float *) malloc (sizeof(float) * N * N);

    // QR (A, lD, N, E_vals, E);

    // return;
    printMatrix (D, M, N);
    float *Dt = (float *) malloc (sizeof(float) * N * M);
    MatrixTranspose (D, Dt, M, N);
    printMatrix (Dt, N, M);

    float *SVDMatrix = (float *) malloc (sizeof(float) * M * M);
    MatrixMultiply (D, Dt, SVDMatrix, M, N, M);

    printMatrix (SVDMatrix, M, M);

    float *E_vals = (float *) malloc (sizeof(float) * M);
    float *E = (float *) malloc (sizeof(float) * M * M);
    float *lD = (float *) malloc (sizeof(float) * M * M);

    // Finding all eigenvalues
    QR (SVDMatrix, lD, M, E_vals, E);

    printMatrix (SVDMatrix, M, M);
    printMatrix (lD, M, M);
    printMatrix (E_vals, 1, M);
    printMatrix (E, M, M);

    // Extracting first N eigenvalues and computing UVSigma
    Decompose (Dt, E, E_vals, M, N, *U, *SIGMA, *V_T);
    printMatrix (lD, M, M);
    printMatrix (E_vals, 1, M);
    printMatrix (*U, N, N);
    printMatrix (*SIGMA, 1, N);
    printMatrix (*V_T, M, M);
    return;


    // float *At = (float *) malloc (sizeof(float) * N * N);
    // float *u = (float *) malloc (sizeof(float) * N * N);
    // float *e = (float *) malloc (sizeof(float) * N * N);
    // float *p = (float *) malloc (sizeof(float) * N);
    // float *Q = (float *) malloc (sizeof(float) * N * N);
    // float *R = (float *) malloc (sizeof(float) * N * N);

    // GramSchmidt (A, N, At, u, e, p, Q, R);

    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%f ", Q[i*N + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    // for (int i = 0; i < N; i++) {
    //     for (int j = 0; j < N; j++) {
    //         printf("%f ", R[i*N + j]);
    //     }
    //     printf("\n");
    // }

    

    // // int m = 2758, n = 4002, p = 2920;
    // int m = 1, n = 10000, p = 1;
    // int I = 5;
    // float *A = (float *) malloc (sizeof(float) * m * n);
    // float *B = (float *) malloc (sizeof(float) * n * p);
    // float *C = (float *) malloc (sizeof(float) * I * I);
    // float *Cp = (float *) malloc (sizeof(float) * I * I); 

    // for (int a = 0; a < m*n; a++) {
    //     A[a] = (rand() % 100000) / 100000.0;
    //     A[a] = 10;
    // }
    // for (int b = 0; b < n*p; b++) {
    //     B[b] = (rand() % 100000) / 100000.0;
    //     // B[b] = b;
    // }
    // for (int i = 0; i < I; i++) 
    //     for (int j = 0; j < I; j++)
    //         C[i*I+j] = rand() % 100;
    
    // // float start_time = omp_get_wtime();
    // // MatrixMultiply (A, B, C, m, n, p, 0);
    // // float end_time = omp_get_wtime();
    // // printf("Seq time: %f\n", end_time - start_time);

    // for (int i = 0; i < I; i++) {
    //     for (int j = 0; j < I; j++)
    //         printf("%f ", C[i*I+j]);
    //     printf("\n");
    // }
    //     printf("\n");

    // float start_time = omp_get_wtime();
    // float norm = VectorNorm (A, n);
    // // MatrixSubtract (A, A, A, m, n);
    // // ScalarDivide (A, A, m, n, 10);
    // MatrixTranspose(C, Cp, I);
    // // makeIdenMatrix (C, I);
    // float end_time = omp_get_wtime();
    // for (int i = 0; i < I; i++) {
    //     for (int j = 0; j < I; j++)
    //         printf("%f ", Cp[i*I+j]);
    //     printf("\n");
    // }
    // for (int i = 0; i < I; i++) {
    //     for (int j = 0; j < I; j++) 
    //         printf("%f ", C[i*I+j]);
    //     printf("\n");
    // }
    // printf ("%f\n", norm);
    // MatrixMultiply (A, A, Cp, m, n, p, 0);
    // printf("Par time: %f\n", end_time - start_time);

    // for (int c = 0; c < m*p; c++) {
    //     printf("%f ", Cp[c]);
    //     // if (abs(C[c] - Cp[c]) > 1e-5) {
    //     //     printf("Not Same!! c=%d | C=%f | Cp=%f \n", c, C[c], Cp[c]);
    //     //     break;
    //     // }
    // }
}

void PCA(int retention, int M, int N, float* D, float* U, float* SIGMA, float** D_HAT, int *K)
{
    
}
