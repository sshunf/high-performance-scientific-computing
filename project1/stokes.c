#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
	// check number of arguments
	if (argc != 7) {
		printf("wrong number of arguments\n");
		return -1;
	}

	int N = atoi(argv[1]);
	int mu = atoi(argv[2]);
	int P = atoi(argv[3]);
	double omega = atof(argv[4]);
	double tol = atof(argv[5]);
	int K = atoi(argv[6]);

	printf("N: %d\n", N);
	printf("mu: %d\n", mu);
	printf("P: %d\n", P);
	printf("omega: %.2f\n", omega);
	printf("tol: %.10f\n", tol);
	printf("K: %d\n", K);
	
	float dx, dy;
      	dx = dy	= 1.0 / (N-1);

	// allocate memory for the x,y velocities and pressure
	double (*u)[N-1] = calloc(N, sizeof(*u)); // use calloc to initialize all memory to 0
	double (*v)[N] = calloc(N-1, sizeof(*v));
	double (*p)[N-1] = malloc(sizeof(*p)*(N-1));
	
	// apply all boundary conditions here
	for (int k = 0; i < K; ++k) {
		p[0][k] = 2*P - p[0][k];
		p[N-1][k] = 0.0;
	}

	float max_residual = 0;
	float u_res, v_res, p_res;
	for (int i = 0; i < K; ++i) {
		// complete the updates here
		for (int j = 1; j < K; ++j) {
			for (int k = 1; k < K-1; ++k) {
				u[j][k] = u[j][k] + omega *
					( mu*(u[j-1][k] + 2*u[j][k] + u[j+1][k])
					+ mu*(u[j][k+1] - 2*u[j][k] + u[j][k-1])
				       	- dy*(p[j][k] - p[j-1][k])
					);	
			}
		}	
		// break when we are within tolerance
		if (max_residual < tol) break;
	}

	free(u);
	free(v);
	free(p);

	return 0;
}



