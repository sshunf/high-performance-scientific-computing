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
	
	double dx, dy;
	dx = dy	= 1.0 / (N-1.0);

	// allocate memory for the x,y velocities and pressure
	// use calloc to initialize memory of u and v to 0
	double (*u)[N-1] = calloc(N, sizeof(*u)); // N x (N-1)
	double (*v)[N] = calloc(N-1, sizeof(*v)); // (N-1) x N
	double (*p)[N-1] = calloc(N-1, sizeof(*p)); // (N-1) x (N-1)
	

	// apply boundary conditions TODO: check if we need this
	for (int k = 0; k < N-1; ++k){
		p[0][k] = P;
		p[N-2][k] = 0.0;
	}

	double max_residual = 0.0;
	double u_res, v_res, p_res;

	// beginning of SOR
	for (int i = 0; i < K; ++i) {
		
		// udpate the u values

		// update bottom left
		u_res = mu * (
            -4*u[0][0]
            + u[1][0]
            + u[0][1]
          )
        - 2*dy*(p[0][0] - P); // TODO: check this part

		u[0][0] += omega*u_res;
		max_residual = max(u_res, max_residual);

		// update left side (inlet)
		for(int k = 1; k < N-2; ++k) {
			u_res = mu * (
						-3*u[0][k]
						+ u[1][k]
						+ u[0][k-1]
						+ u[0][k+1]
					)
					- 2*dy*(p[0][k] - P);
			u[0][k] += omega * u_res; 
			max_residual = max(u_res, max_residual);
		} 

		// update top left
		u_res = mu * (
			-4*u[0][N-2]
			+ u[1][N-2]
			+ u[0][N-3]
		  )
		- 2*dy*(p[1][N-2] - p[0][N-2]);
		
		 // update bottom row
		for (int j = 1; j < N-1; ++j) {
			u_res = mu * (
				-5*u[j][0]
				+ u[j-1][0]
				+ u[j][1]
				+ u[j+1][0]
			  )
			- dy * (p[j][0] - p[j-1][0]);
			u[j][0] += omega * u_res; 
			max_residual = max(u_res, max_residual);
			
			 // update interior
			for (int k = 1; k < N-2; ++k) {
				u_res = mu * (
					- 4*u[j][k]
					+ u[j-1][k]
					+ u[j+1][k]
					+ u[j][k-1]
					+ u[j][k+1]
				) 
				- dy * (p[j][k] - p[j-1][k]);

				u[j][k] += omega*u_res;
				max_residual = max(u_res, max_residual);	
			}

			// update top row
			u_res = mu * (
				-5*u[j][N-2]
				+ u[j-1][N-2]
				+ u[j][N-3]
				+ u[j+1][N-2]
			  )
			- dy * (p[j][N-2] - p[j-1][N-2]);
			u[j][N-2] += omega * u_res; 
			max_residual = max(u_res, max_residual);	
		}

		// update bottom right
		u_res = mu * (
			-4*u[N-1][0]
			+ u[N-2][0]
			+ u[N-1][1]
		  )
		- dy * (p[N-1][0] - p[N-2][0]);
		u[N-1][0] += omega * u_res;
		max_residual = max(u_res, max_residual);	

		 // update right side (outlet)
		for (int k = 1; k < N-2; ++k) {
			u_res = mu * (
				-3*u[N-1][k]
				+ u[N-2][k]
				+ u[N-1][k-1]
				+ u[N-1][k+1]
			)
			- dy * (p[N-1][k] - p[N-2][k]);
			u[N-1][k] += omega * u_res;
			max_residual = max(u_res, max_residual);	
		}

		// update top right
		u_res = mu * (
			-4*u[N-1][N-2]
			+ u[N-2][N-2]
			+ u[N-1][N-3]
		  )
		- dy * (p[N-1][N-2] - p[N-2][N-2]);
		u[N-1][N-2] += omega * u_res;
		max_residual = max(u_res, max_residual);


		// update the v values

		// update bottom left TODO: check if these are right
		// v_res = mu * (
		// 	-3*v[0][0]
		// 	+ v[1][0]
		// 	+ v[0][1]
		// )
		// - 2 * dx * (p[0][0] - P);

		// v[0][0] += omega*v_res;
		// max_residual = max(u_res, max_residual);
		v[0][0] = 0;
		
		// update left side
		for (int k = 1; k < N-1; ++k) {
			v_res = mu * (
				-3*v[0][k]
				+ v[1][k]
				+ v[0][k+1]
				+ v[0][k-1]
			)
			- dx * (p[0][k] - p[0][k-1]);
			v[0][k] += omega*v_res;
			max_residual = max(v_res, max_residual);
		}

		// update top left
		// v_res = mu * (
		// 	-3*v[0][N-1]
		// 	+ v[1][N-1]
		// 	+ v[0][N-2]
		// )
		// - dx * 2 * (p[0][N-1] - P);

		// v[0][N-2] += omega*v_res;
		// max_residual = max(v_res, max_residual);
		v[0][N-2] = 0;

		for (int j = 1; j < N-2; ++j) {
			// update bottom
			// v_res = mu * (
			// 	-3*v[j][0]
			// 	+ v[j-1][0]
			// 	+ v[j+1][0]
			// 	+ v[j][1]
			// )
			// - dx * 2 * (p[j][0] - P);

			// v[j][0] += omega*v_res;
			// max_residual = max(v_res, max_residual);	
			v[j][0] = 0;

			// update interior
			for (int k = 1; k < N-1; ++k) {
				v_res = mu * (
					- 4*v[j][k] 
					+ v[j-1][k] 
					+ v[j+1][k]
					+ v[j][k+1] 
					+ v[j][k-1]
				)
				- dx*(p[j][k] - p[j][k-1]);

				v[j][k] += omega*v_res; 
				max_residual = max(v_res, max_residual);
			}

			// update top
			// v_res = mu * (
			// 	-3*v[j][N-1]
			// 	+ v[j-1][N-1]
			// 	+ v[j+1][N-1]
			// 	+ v[j][N-1]
			// )
			// - dx * (v[j][N-1] - v[j][N-2]);

			// v[j][N-1] += omega*v_res;
			// max_residual = max(v_res, max_residual);
			v[j][N-1] = 0;
		}

		// update bottom right
		v[N-2][0] = 0;

		// update right side
		for (int k = 1; k < N-2; ++k) {
			v_res = mu * (
				-3*v[N-2][k]
				+ v[N-3][k]
				+ v[N-2][k+1]
				+ v[N-2][k-1]
			)
			- dx * (p[N-2][k] - p[N-2][k-1]);

			v[N-2][k] += omega*v_res;
			max_residual = max(v_res, max_residual);	
		}

		// update top right
		v[N-2][N-1] = 0;

		// boundary conditions
		for(int j = 0; j < N; ++j){
			v[0  ][j] = 0.0;
			v[N-2][j] = 0.0;
		}
		for(int k = 0; k < N-1; ++k){
			v[k][0]   = v[k][1];
			v[k][N-1] = v[k][N-2];
		}

		// update the p values

		// update bottom left
		p_res = mu * (
			- (u[1][0] - u[0][0])
			- (v[0][1] - v[0][0])
		);
		// p_res = 2 * P - p[1][0] - p[0][0];

		p[0][0] += omega*p_res;
		max_residual = max(p_res, max_residual);

		// update the left values
		for (int k = 1; k < N-2; ++k) {
			p_res = 2 * P - p[1][k] - p[0][k];

			p[0][k] += omega*p_res;
			max_residual = max(p_res, max_residual);
		}

		// update the top left
		p_res = 2 * P - p[1][N-2] - p[0][N-2];

		p[0][N-2] += omega*p_res;
		max_residual = max(p_res, max_residual);

		// update the interior
		for (int j = 1; j < N-2; ++j) {
			// update the bottom
			p_res = -(u[j+1][0]	- u[j][0]) - (v[j][1] - v[j][0]);

			p[j][0] += omega*p_res;
			max_residual = max(p_res, max_residual);	

			// interior
			for (int k = 1; k < N-2; ++k) {
				p_res = -(u[j+1][k] - u[j][k])
					- (v[j][k+1] - v[j][k]);

				p[j][k] += omega*p_res;
				max_residual = max(p_res, max_residual);
			}

			// update the top
			p_res = -(u[j+1][N-2] - u[j][N-2]) + v[j][N-2];

			p[j][N-2] += omega*p_res;
			max_residual = max(p_res, max_residual);		
		}

		// update bottom right
		p_res = -(u[N-1][0] - u[N-2][0]) - (v[N-2][1] - v[N-2][0]);

		p[N-2][N-2] += omega*p_res;
		max_residual = max(p_res, max_residual);	

		// update right side
		for (int k = 1; k < N-2; ++k) {
			p_res = -(u[N-1][0] - u[N-2][0]) - (v[N-2][1] - v[N-2][0]);

			p[N-2][k] += omega*p_res;
			max_residual = max(p_res, max_residual);
		}

		// update top right
		p_res = -(u[N-1][N-2] - u[N-2][N-2]) - (v[N-2][N-1] - v[N-2][N-2]);

		p[N-2][N-2] += omega*p_res;
		max_residual = max(p_res, max_residual);	

		// TODO: Check if we need to apply BCs here
		for (int j = 0; j < N; ++j) {
			u[j][0] = 0.0;
			u[j][N-2] = 0.0;
		}

		for (int j = 0; j < N-1; ++j) {
			v[j][0] = 0.0;
			v[j][N-1] = 0.0;
		}

		for(int j = 0; j < N; ++j){
			v[0][j] = 0.0;
			v[N-2][j] = 0.0;
		}

		for (int k = 0; k < N-1; k++) {
			p[0][k] = P;
			p[N-2][k] = 0;
		}
		
		// break when we are within tolerance
		if (max_residual < tol) {
			printf("number of iterations: %d\n", i);
			break;
		}
	}


	// write data to files
	FILE *fU = fopen("StokesU.out", "wb");
	fwrite(u, sizeof(double), N*(N-1), fU);
	fclose(fU);

	FILE *fV = fopen("StokesV.out","wb");
	fwrite(v, sizeof(double), (N-1)*N, fV);
	fclose(fV);

	FILE *fP = fopen("StokesP.out","wb");
	fwrite(p, sizeof(double), (N-1)*(N-1), fP);
	fclose(fP);

	free(u);
	free(v);
	free(p);

	return 0;
}



