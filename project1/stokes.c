#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static inline double max(double a, double b) {
	return (fabs(a) > b) ? a : b;
}

// functions to recompute boundary conditions
void recompute_bcs(double* u, double* v, double* p, int N) {
	// for u (top and bottom)
	for (int j = 0; j < N; ++j) {
		// TODO
	}

}

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
	// use calloc to initialize memory of u and v to 0
	double (*u)[N-1] = calloc(N, sizeof(*u)); // N x (N-1)
	double (*v)[N] = calloc(N-1, sizeof(*v)); // (N-1) x N
	double (*p)[N-1] = malloc(sizeof(*p)*(N-1)); // (N-1) x (N-1)
	
	// apply boundary conditions to p
	for (int k = 0; k < N-1; ++k) {
		p[0][k] = 2*P - p[0][k];
		p[N-2][k] = 0.0;
	}

	float max_residual = 0;
	float u_res, v_res, p_res;
	for (int i = 0; i < K; ++i) {
		// udpate the u values
		for (int j = 1; j < N-1; ++j) {
			for (int k = 1; k < N-2; ++k) {
			u_res = mu*(u[j-1][k] - 4*u[j][k] + u[j+1][k]
				+ u[j][k+1] + u[j][k-1])
				- dy*(p[j][k] - p[j-1][k]);	
			
			u[j][k] += omega*u_res;
			max_residual = max(u_res, max_residual);	
			}
		}	
		
		// update the v values
		for (int j = 1; j < N-2; ++j) {
			for (int k = 1; k < N-1; ++k) {
			      v_res = mu*(v[j-1][k] - 4*v[j][k] + v[j+1][k]
				      + v[j][k+1] + v[j][k-1])
				      - dx*(p[j][k] - p[j][k-1]);

			      v[j][k] += omega*v_res;
			      max_residual = max(v_res, max_residual);
			}
		}
		
		for (int j = 1; j < N-2; ++j) {
			for (int k = 1; k < N-2; ++k) {
				p_res = -(u[j+1][k] - u[j][k])
					- (v[j][k+1] - v[j][k]);

				p[j][k] += omega*p_res;
				max_residual = max(p_res, max_residual);
			}
		}

		// break when we are within tolerance
		if (max_residual < tol) break;
	}


	// delete later
	for (int j=0; j<N; ++j){
		for (int k=0; k<N-1; ++k){
			printf("%.f ", u[j][k]);
		}
		printf("\n");
	}

	free(u);
	free(v);
	free(p);

	return 0;
}



