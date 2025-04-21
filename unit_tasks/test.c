#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

int main(int argc, char* argv[])
{
        int N = atoi(argv[1]);
	double* u = (double*) malloc(N*sizeof(double));
        int i;
        #pragma omp parallel for shared(N, u) private(i) num_threads(3) 
	for (i = 0; i < N; i++) {
		printf("u[%d] <- thread %d\n", i, omp_get_thread_num());
		u[i] = (double)i;
	}
	free(u);
        return 0;
}

