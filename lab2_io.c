#include "lab2_io.h"

void read_matrix (const char* input_filename, int* M, int* N, float** D){
	FILE *fin = fopen(input_filename, "r");

	fscanf(fin, "%d%d", M, N);
	
	int num_elements = (*M) * (*N);
	*D = (float*) malloc(sizeof(float)*(num_elements));
	
	for (int i = 0; i < num_elements; i++){
		fscanf(fin, "%f", (*D + i));
	}
	fclose(fin);
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

void write_result (int M, 
		int N, 
		float* D, 
		float* U, 
		float* SIGMA, 
		float* V_T,
		int K, 
		float* D_HAT,
		double computation_time){
	// Will contain output code
	printf("Time taken: %.4f\n", computation_time);
	printMatrix (U, N, N);
    printMatrix (SIGMA, 1, N);
    printMatrix (V_T, M, M);
}
