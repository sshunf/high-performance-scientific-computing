#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static inline double max(double a, double b) {
	return (fabs(a) > b) ? fabs(a) : b;
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
	double (*p)[N-1] = calloc(N-1, sizeof(*p)); // (N-1) x (N-1)
	
	// apply boundary conditions to p
	for (int k = 0; k < N-1; ++k) {
		p[0][k] = P; // TODO: check this
		p[N-2][k] = 0.0;
	}

	double max_residual = 0.0;
	double u_res, v_res, p_res;

	// beginning of solve
	for (int i = 0; i < K; ++i) {
		
		// udpate the u values
		u_res = mu * (
            -4*u[0][0]
            + u[1][0]
            + u[0][1]
          )
        - 2*dy * (p[0][0] - P);

		u[0][0] += omega*u_res; // update bottom left
		max_residual = max(u_res, max_residual);

		for(int k = 1; k < N-2; ++k) {
			u_res = mu * (
						-3*u[0][k]
						+ u[1][k]
						+ u[0][k-1]
						+ u[0][k+1]
					)
					- 2*dy * (p[0][k] - P);
			u[0][k] += omega * u_res; // update left side (inlet)
			max_residual = max(u_res, max_residual);
		} 

		// update top left
		u_res = mu * (
			-4*u[0][N-2]
			+ u[1][N-2]
			+ u[0][N-3]
		  )
		- 2*dy * (p[1][N-2] - p[0][N-2]);
		
		for (int j = 1; j < N-1; ++j) {
			u_res = mu * (
				-5*u[j][0]
				+ u[j-1][0]
				+ u[j][1]
				+ u[j+1][0]
			  )
			- 2*dy * (p[j+1][0] - p[j][k]);
			u[j][0] += omega * u_res; // update bottom row
			max_residual = max(u_res, max_residual);
			
			for (int k = 1; k < N-2; ++k) {
				u_res = mu * (
					u[j-1][k]
					- 4*u[j][k]
					+ u[j+1][k]
					+ u[j][k-1]
					+ u[j][k+1]
				) 
				- dy * (p[j][k] - p[j-1][k]);

				u[j][k] += omega*u_res; // update interior
				max_residual = max(u_res, max_residual);	
			}

			u_res = mu * (
				-5*u[j][N-2]
				+ u[j-1][N-2]
				+ u[j][N-3]
				+ u[j+1][N-2]
			  )
			- 2*dy * (p[j+1][0] - p[j][k]);
			u[j][N-2] += omega * u_res; // update bottom row
			max_residual = max(u_res, max_residual);	
		}

		u_res = mu * (
			-4*u[N-1][0]
			+ u[N-2][0]
			+ u[N-1][1]
		  )
		- 2*dy * (p[N][0] - p[j][k]);
		u[N-1][0] += omega * u_res; // update bottom right
		max_residual = max(u_res, max_residual);	

		for (int k = 1; k < N-2; ++k) {
			u_res = mu * (
				-3*u[N-1][k]
				+ u[N-2][k]
				+ u[N-1][k-1]
				+ u[N-1][k+1]
			)
			- 2*dy * (p[0][k] - P);
			u[N-1][k] += omega * u_res; // update right side (outlet)
			max_residual = max(u_res, max_residual);	
		}

		u_res = mu * (
			-4*u[N-1][N-2]
			+ u[N-2][N-2]
			+ u[N-1][N-3]
		  )
		- 2*dy * (p[N][0] - p[j][k]);
		u[N-1][N-2] += omega * u_res; // update top right
		max_residual = max(u_res, max_residual);		

		
		// update the v values
		for (int j = 1; j < N-1; ++j) {
			for (int k = 1; k < N-2; ++k) {
			      v_res = mu*(v[j-1][k] - 4*v[j][k] + v[j+1][k]
				      + v[j][k+1] + v[j][k-1])
				      - dx*(p[j][k] - p[j][k-1]);

			      v[j][k] += omega*v_res;
			      max_residual = max(v_res, max_residual);
			}
		}
		
		// update the p values
		for (int j = 1; j < N-2; ++j) {
			for (int k = 1; k < N-2; ++k) {
				p_res = -(u[j+1][k] - u[j][k])
					- (v[j][k+1] - v[j][k]);

				p[j][k] += omega*p_res;
				max_residual = max(p_res, max_residual);
			}
		}

		// Recompute boundary conditions
		// for (int k = 1; k < N-1; ++k) {
        //         	// TODO
		// 	u[0][k] += omega*u_res;
        // 	}
		

		// break when we are within tolerance
		if (max_residual < tol) break;
	}


	// delete later
	for (int j=0; j<N-1; ++j){
		for (int k=0; k<N; ++k){
			printf("%.f ", u[j][k]);
		}
		printf("\n");
	}

	free(u);
	free(v);
	free(p);

	return 0;
}



