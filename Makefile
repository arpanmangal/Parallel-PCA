all: 
	g++ -fopenmp -lm lab2_io.c lab2_omp.cpp main_omp.c -o pca

clean:
	rm -rf pca