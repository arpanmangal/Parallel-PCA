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
        double factor;
        if (uNorm < epsilon) {
            factor = 0;
        } else {
            factor = 1 / uNorm;
        }
        ScalarMultiply (u + k*N, e + k*N, 1, N, factor);
    }

    MatrixTranspose (e, Q, N, N);
    MatrixMultiply (e, A, R, N, N, N);
    makeTriangular (R, N, 1);
}

bool hasConverged (double *E, double *E_next, int N) {
    double diff;
    #pragma omp parallel for collapse(2) schedule (dynamic, 64) reduction (+:diff)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            diff += abs(E[i*N+j] - E_next[i*N+j]);
        }
    }

    return (diff < 1e-6);
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

    for (int i = 0; i < 10000; i++) {
        GramSchmidt (lD, N, At, u, e, p, Q, R);
        MatrixMultiply (R, Q, lD, N, N, N);
        MatrixMultiply (E, Q, E_next, N, N, N);
        if (hasConverged(E, E_next, N)) {
            printf("converged: %d\n", i);
            break;
        }
        MatrixAssign (E_next, E, N, N);
    }

    // printMatrix (lD, N, N);
    // printMatrix (E, N, N);
    // exit(0);

    #pragma omp parallel for schedule (dynamic, 64)
    for (int i = 0; i < N; i++) {
        Evals[i] = lD[i*N+i];
    }

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
    
    MatrixTranspose (E, V_T, M, M);
    MatrixAssign (V_T, E, M, M);

    // Possible parallelisation => no
    for (int i = 0; i < M; i++) {
        int rank = EigenVals[i].second;
        MatrixAssign (E + rank*M, V_T + i*M, 1, M);
        if (i < N)
            SIGMA [i] = EigenVals[i].first;
    }

    double *SigmaInvMatrix = (double *) malloc (sizeof(double) * M * N);
    double *VSigmaInvMatrix = (double *) malloc (sizeof(double) * M * N);

    #pragma omp parallel for schedule (dynamic, 64) collapse (2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            SigmaInvMatrix [i*N + j] = (i == j) * (1 / SIGMA[j]);
        }
    }
    MatrixTranspose (V_T, E, M, M);
    MatrixMultiply (E, SigmaInvMatrix, VSigmaInvMatrix, M, M, N);

    MatrixMultiply (Dt, VSigmaInvMatrix, U, N, M, N);

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

    double *Dt = (double *) malloc (sizeof(double) * N * M);
    MatrixTranspose (D, Dt, M, N);

    double *SVDMatrix = (double *) malloc (sizeof(double) * M * M);
    MatrixMultiply (D, Dt, SVDMatrix, M, N, M);

    double *E_vals = (double *) malloc (sizeof(double) * M);
    double *E = (double *) malloc (sizeof(double) * M * M);
    double *lD = (double *) malloc (sizeof(double) * M * M);
    
    // Finding all eigenvalues
    QR (SVDMatrix, lD, M, E_vals, E);

    // Extracting first N eigenvalues and computing UVSigma
    Decompose (Dt, E, E_vals, M, N, U, SIGMA, V_T);

    double *SigmaMatrix = (double *) malloc (sizeof(double) * N * M);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            SigmaMatrix[i * M + j] = (i == j) * (SIGMA)[j];
        }
    }
    double *USigmaMatrix = (double *) malloc (sizeof(double) * N * M);
    MatrixMultiply (U, SigmaMatrix, USigmaMatrix, N, N, M);

    double *USigmaMatrixVT = (double *) malloc (sizeof(double) * N * M);
    MatrixMultiply (USigmaMatrix, V_T, USigmaMatrixVT, N, M, M);

    printMatrix (U, N, N);
    printMatrix (SigmaMatrix, N, M);
    printMatrix (V_T, M, M);
    printMatrix (USigmaMatrixVT, N, M);

    // Convert to their format
    MatrixAssignDtoF (U, *Uf, N, N);
    MatrixAssignDtoF (SIGMA, *SIGMAf, N, M);
    MatrixAssignDtoF (V_T, *V_Tf, M, M); 
}

void PCA(int retention, int M, int N, float* D, float* U, float* SIGMA, float** D_HAT, int *K)
{
    
}
