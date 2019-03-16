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
    #pragma omp parallel for schedule (dynamic, 64) reduction (+:norm)
    for (int i = 0; i < m; i++)
        norm += A[i] * A[i];

    return sqrt(norm);
}

double InnerProduct (double *A, double *B, int m) {
    double IP = 0;
    #pragma omp parallel for schedule (dynamic, 64) reduction (+:IP)
    for (int i = 0; i < m; i++)
        IP += A[i] * B[i];

    return IP;
}

void ScalarMultiply (double *A, double *B, int m, int n, double alpha) {
    int size = m * n;
    #pragma omp parallel for schedule (dynamic, 64)
    for (int i = 0; i < size; i++)
        B[i] = A[i] * alpha;
}

void makeIdenMatrix (double *I, int m) {
    printf("%d\n", m);
    #pragma omp parallel for schedule (dynamic, 64) collapse (2)
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++)
            I[i*m + j] = (i == j);
}

void MatrixTranspose (double *A, double *B, int m, int n) {
    #pragma omp parallel for schedule (dynamic, 64) collapse (2)
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            B[j*m + i] = A[i*n + j];
}

void MatrixAssign (double *A, double *B, int m, int n) {
    int size = m * n;
    #pragma omp parallel for schedule (dynamic, 64)
    for (int i = 0; i < size; i++)
        B[i] = A[i];
}

void MatrixAssignFtoD (float *A, double *B, int m, int n) {
    int size = m * n;
    #pragma omp parallel for schedule (dynamic, 64)
    for (int i = 0; i < size; i++)
        B[i] = A[i];
}

void MatrixAssignDtoF (double *A, float *B, int m, int n) {
    int size = m * n;
    #pragma omp parallel for schedule (dynamic, 64)
    for (int i = 0; i < size; i++)
        B[i] = A[i];
}

void MatrixProj (double *A, double *U, double *P, int N) {
    // All N * 1 Matrix
    // Project A on U
    double factor = InnerProduct (U, A, N) / InnerProduct (U, U, N);
    ScalarMultiply (U, P, N, 1, factor);
}

void makeTriangular (double *M, int N, int upper=1) {
    // Make the matrix triangular
    #pragma omp parallel for schedule (dynamic, 64) collapse (2)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            if ((upper == 1 && j < i) || (upper == 0 && j > i))
                M[i*N + j] = 0;
}

void printMatrix (double *M, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", M[i*n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void GramSchmidt (double *A, int N, double *At, double *u, double *e, double *p, double *Q, double *R) {
    // Just allocate At, u, e matrices to be N*N; p matrix to be N * 1
    MatrixTranspose (A, At, N, N);
    double epsilon = 1e-5;

    for (int k = 0; k < N; k++) {
        MatrixAssign (At + k*N, u + k*N, 1, N);
        for (int j = 0; j < k; j++) {
            double uNorm = VectorNorm (u + j*N, N);
            if (uNorm < epsilon) {
                // Don't take projection
                continue;
            }
            MatrixProj (At + k*N, u + j*N, p, N);
            MatrixSubtract (u + k*N, p, u + k*N, 1, N);
        }

        double uNorm = VectorNorm (u + k*N, N);
        // printf("%f\n", uNorm);
        double factor;
        if (uNorm < epsilon) {
            factor = 0;
        } else {
            factor = 1 / uNorm;
        }
        ScalarMultiply (u + k*N, e + k*N, 1, N, factor);
    }
    // printMatrix (A, N, N);
    // printMatrix (u, N, N);
    // printMatrix (e, N, N);
    // exit(0);

    MatrixTranspose (e, Q, N, N);
    MatrixMultiply (e, A, R, N, N, N);
    // printMatrix(R, N, N);
    makeTriangular (R, N, 1);
}

void QR (double *D, double *lD, int N, double *Evals, double *E) {
    MatrixAssign (D, lD, N, N);
    makeIdenMatrix (E, N);

    double *At = (double *) malloc (sizeof(double) * N * N);
    double *u = (double *) malloc (sizeof(double) * N * N);
    double *e = (double *) malloc (sizeof(double) * N * N);
    double *p = (double *) malloc (sizeof(double) * N);
    double *Q = (double *) malloc (sizeof(double) * N * N);
    double *R = (double *) malloc (sizeof(double) * N * N);
    double *E_next = (double *) malloc (sizeof(double) * N * N);

    for (int i = 0; i < 1000; i++) {
        // printMatrix (lD, N, N);
        GramSchmidt (lD, N, At, u, e, p, Q, R);
        // printMatrix (Q, N, N);
        // printMatrix (R, N, N);
        MatrixMultiply (R, Q, lD, N, N, N);
        MatrixMultiply (E, Q, E_next, N, N, N);
        MatrixAssign (E_next, E, N, N);
    }
    printMatrix (lD, N, N);
    printMatrix (E, N, N);
    // exit(0);

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

bool sortPair (std::pair<double, int> &a, std::pair<double, int> &b)
{
    return (a.first > b.first);
}
void Decompose (double * Dt, double *E, double* E_vals, int M, int N, double *U, double *SIGMA, double *V_T)
{
    // D is M*M, U is N*N, SIGMA => N, V_T => M*M
    double start_time = omp_get_wtime();
    std::vector<std::pair<double, int>> EigenVals;
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
    // exit(0);

    double *SigmaInvMatrix = (double *) malloc (sizeof(double) * M * N);
    double *VSigmaInvMatrix = (double *) malloc (sizeof(double) * M * N);

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

    // exit(0);

    double end_time = omp_get_wtime();
    printf("Decompose time: %f\n", end_time - start_time);
}

void SVD(int M, int N, float* Df, float** Uf, float** SIGMAf, float** V_Tf)
{
    omp_set_num_threads(4);
    srand (0); // Deterministically Random

    // Double Matrices
    double *D = (double *) malloc (sizeof(double) * M * N);
    double *U = (double*) malloc(sizeof(double) * N*N);
	double *SIGMA = (double*) malloc(sizeof(double) * N*M);
	double *V_T = (double*) malloc(sizeof(double) * M*M);

    MatrixAssignFtoD (Df, D, M, N);

    // N = 4;
    // double A[] = {4,1,0,2};
    // double u[] = {5, -1, 2, 3};
    // double *P = (double *) malloc (sizeof(double) * N);
    // printMatrix (A, 1, N);
    // printMatrix (u, 1, N);
    // MatrixProj (A, u, P, N);
    // printMatrix (P, 1, N);
    // exit(0);
    // N = 3, M = 5;
    // double a[] = {12, -51, 4, 6, 167, -68, -4, 24, -41, 10, 23, 10, 78, 90, 13};
    // // double a[] = {-2, -4, 2, -2, 1, 2, 4, 2, 5};
    // // double a[] = {24, -15, -15, 25};
    // double *A = (double *) malloc (sizeof(double) * M * N);
    // for (int i = 0; i < M*N; i++)
    //     A[i] = a[i];
    // double *B = (double *) malloc (sizeof(double) * N * M);

    // MatrixTranspose (A, B, M, N);
    // printMatrix (A, M, N);
    // printMatrix (B, N, M);
    // return;

    // N = 2;
    // M = 2;
    // // double a[] = {12, -51, 4, 6, 167, -68, -4, 24, -41};
    // // double a[] = {-2, -4, 2, -2, 1, 2, 4, 2, 5};
    // double a[] = {25, -15, -15, 25};
    // double *A = (double *) malloc (sizeof(double) * N * N);

    // for (int i = 0; i < N*N; i++)
    //     A[i] = a[i];

    // double *E_vals = (double *) malloc (sizeof(double) * N);
    // double *E = (double *) malloc (sizeof(double) * N * N);
    // double *lD = (double *) malloc (sizeof(double) * N * N);

    // QR (A, lD, N, E_vals, E);

    // return;
    printMatrix (D, M, N);
    double *Dt = (double *) malloc (sizeof(double) * N * M);
    MatrixTranspose (D, Dt, M, N);
    printMatrix (Dt, N, M);

    double *SVDMatrix = (double *) malloc (sizeof(double) * M * M);
    MatrixMultiply (D, Dt, SVDMatrix, M, N, M);

    printMatrix (SVDMatrix, M, M);
    // exit(0);

    double *E_vals = (double *) malloc (sizeof(double) * M);
    double *E = (double *) malloc (sizeof(double) * M * M);
    double *lD = (double *) malloc (sizeof(double) * M * M);
    
    // Finding all eigenvalues
    QR (SVDMatrix, lD, M, E_vals, E);
    // exit(0);

    printMatrix (SVDMatrix, M, M);
    printMatrix (lD, M, M);
    printMatrix (E_vals, 1, M);
    printMatrix (E, M, M);
    // exit(0);

    // Extracting first N eigenvalues and computing UVSigma
    Decompose (Dt, E, E_vals, M, N, U, SIGMA, V_T);

    double *SigmaMatrix = (double *) malloc (sizeof(double) * N * M);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            SigmaMatrix[i * M + j] = (i == j) * (SIGMA)[j];
        }
    }
    printMatrix (U, N, N);
    printMatrix (SigmaMatrix, N, M);
    double *USigmaMatrix = (double *) malloc (sizeof(double) * N * M);
    MatrixMultiply (U, SigmaMatrix, USigmaMatrix, N, N, M);

    double *USigmaMatrixVT = (double *) malloc (sizeof(double) * N * M);
    MatrixMultiply (USigmaMatrix, V_T, USigmaMatrixVT, N, M, M);

    printMatrix (USigmaMatrixVT, N, M);

    // printMatrix (lD, M, M);
    // printMatrix (E_vals, 1, M);
    // printMatrix (*U, N, N);
    // printMatrix (*SIGMA, 1, N);
    // printMatrix (*V_T, M, M);
    // return;


    // double *At = (double *) malloc (sizeof(double) * N * N);
    // double *u = (double *) malloc (sizeof(double) * N * N);
    // double *e = (double *) malloc (sizeof(double) * N * N);
    // double *p = (double *) malloc (sizeof(double) * N);
    // double *Q = (double *) malloc (sizeof(double) * N * N);
    // double *R = (double *) malloc (sizeof(double) * N * N);

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
    // double *A = (double *) malloc (sizeof(double) * m * n);
    // double *B = (double *) malloc (sizeof(double) * n * p);
    // double *C = (double *) malloc (sizeof(double) * I * I);
    // double *Cp = (double *) malloc (sizeof(double) * I * I); 

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
    
    // // double start_time = omp_get_wtime();
    // // MatrixMultiply (A, B, C, m, n, p, 0);
    // // double end_time = omp_get_wtime();
    // // printf("Seq time: %f\n", end_time - start_time);

    // for (int i = 0; i < I; i++) {
    //     for (int j = 0; j < I; j++)
    //         printf("%f ", C[i*I+j]);
    //     printf("\n");
    // }
    //     printf("\n");

    // double start_time = omp_get_wtime();
    // double norm = VectorNorm (A, n);
    // // MatrixSubtract (A, A, A, m, n);
    // // ScalarDivide (A, A, m, n, 10);
    // MatrixTranspose(C, Cp, I);
    // // makeIdenMatrix (C, I);
    // double end_time = omp_get_wtime();
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
