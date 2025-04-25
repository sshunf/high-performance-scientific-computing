/*
Shun Fujita
ESAPPM 344 - Spring 2025, Project 1

SOR Method to solve stokes flow, using OpenMP
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

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
	double mu = atof(argv[2]);
	double P = atof(argv[3]);
	double omega = atof(argv[4]);
	double tol = atof(argv[5]);
	int K = atoi(argv[6]);

	printf("N: %d\n", N);
	printf("mu: %.2f\n", mu);
	printf("P: %.2f\n", P);
	printf("omega: %.2f\n", omega);
	printf("tol: %.10f\n", tol);
	printf("K: %d\n", K);
	
	double dx, dy;
	dx = dy	= 1.0 / (N-1);

	// allocate memory for the x,y velocities and pressure
	// use calloc to initialize memory of u and v to 0
	double (*u)[N-1] = calloc(N, sizeof(*u)); // N x (N-1)
	double (*v)[N] = calloc(N-1, sizeof(*v)); // (N-1) x N
	double (*p)[N-1] = calloc(N-1, sizeof(*p)); // (N-1) x (N-1)
	
	double max_residual = 0.0;
	double u_res = 0.0, v_res = 0.0, p_res = 0.0;

	// load data into u, v, p here from matlab solution
	// make sure transpositions are correct


	// create timer
	clock_t start, end;
	double cpu_time_used;
	
	start = clock();

	// beginning of SOR
	for (int i = 0; i < K; ++i) {
		max_residual = 0.0;
		/** udpate the u values **/ 

		// update bottom left
		u_res = mu * (
            -4*u[0][0]
            + u[1][0]
            + u[0][1]
          )
        - 2*dy*(p[0][0] - P);

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
			u[0][k] += omega*u_res;
			max_residual = max(u_res, max_residual);
		}

		// update top left
		u_res = mu * (
			-4*u[0][N-2]
			+ u[1][N-2]
			+ u[0][N-3]
		  )
		- 2*dy*(p[0][N-2] - P);

		u[0][N-2] += omega*u_res; // fixed: previously did not update
		max_residual = max(u_res, max_residual);

		 // update bottom row
		for (int j = 1; j < N-1; ++j) {
			u_res = mu * (
				-5*u[j][0]
				+ u[j-1][0]
				+ u[j][1]
				+ u[j+1][0]
			)
			- dy * (p[j][0] - p[j-1][0]);
			u[j][0] += omega*u_res; 
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
		+ 2 * dy * (p[N-2][0]); // fixed this
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
			+ 2 * dy * (p[N-2][k]);
			u[N-1][k] += omega * u_res;
			max_residual = max(u_res, max_residual);
		}

		// update top right
		u_res = mu * (
			-4*u[N-1][N-2]
			+ u[N-2][N-2]
			+ u[N-1][N-3]
		  )
		+ 2 * dy * (p[N-2][N-2]);
		u[N-1][N-2] += omega * u_res;
		max_residual = max(u_res, max_residual);


		/** update v values **/

		// update bottom left
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
		v[0][N-1] = 0; // N-1 for y direction

		for (int j = 1; j < N-2; ++j) {
			// update bottom
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


		/** update p values **/

		// update bottom left
		p_res = - (u[1][0] - u[0][0]) - (v[0][1] - v[0][0]);

		p[0][0] += omega*p_res;
		max_residual = max(p_res, max_residual);

		// update the left values
		for (int k = 1; k < N-2; ++k) {
			p_res = -(u[1][k] - u[0][k]) - (v[0][k+1] - v[0][k]);
			p[0][k] += omega*p_res;
			max_residual = max(p_res, max_residual);
		}

		// update the top left
		p_res = -(u[1][N-2] - u[0][N-2]) - (v[0][N-1] - v[0][N-2]);

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
				p_res = -(u[j+1][k] - u[j][k]) - (v[j][k+1] - v[j][k]);

				p[j][k] += omega*p_res;
				max_residual = max(p_res, max_residual);
			}

			// update the top
			p_res = -(u[j+1][N-2] - u[j][N-2]) + v[j][N-2];

			p[j][N-2] += omega*p_res;
			max_residual = max(p_res, max_residual);
		}

		// update bottom right
		p_res = - (u[N-1][0] - u[N-2][0]) -(v[N-2][1] - v[N-2][0]); // fixed this

		p[N-2][0] += omega*p_res;
		max_residual = max(p_res, max_residual);	

		// update right side
		for (int k = 1; k < N-2; ++k) {
			p_res = - (u[N-1][k] - u[N-2][k]) - (v[N-2][k+1] - v[N-2][k]); // fixed this

			p[N-2][k] += omega*p_res;
			max_residual = max(p_res, max_residual);
		}

		// update top right
		p_res = - (u[N-1][N-2] - u[N-2][N-2]) - (v[N-2][N-1] - v[N-2][N-2]); // fixed this

		p[N-2][N-2] += omega*p_res;
		max_residual = max(p_res, max_residual);


		// debugging print
		// if (i % 100 == 0) {
		// 	printf("max residual at iteration %d: %.12f\n", i, max_residual);
		// }

		// break when we are within tolerance
		if (max_residual < tol) {
			// 
			printf("number of iterations: %d\n", i);
			break;
		}
	}

	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("Time taken: %f seconds\n", cpu_time_used);
	/*
	// print out u
	printf("u: \n");
	for (int j = 0; j < N-1; ++j) {
		for (int k = 0; k < N-2; ++k) {
			printf("%.3f, ", u[j][k]);
		}
		printf("\n");
	}

	// print out v 
	printf("v: \n");
	for (int j = 0; j < N-2; ++j) {
		for (int k = 0; k < N-1; ++k) {
			printf("%.3f, ", v[j][k]);
		}
		printf("\n");
	}

	// print out p
	printf("p: \n");
	for (int j = 0; j < N-2; ++j) {
		for (int k = 0; k < N-2; ++k) {
			printf("%.3f, ", p[j][k]);
		}
		printf("\n");
	}
	*/

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



