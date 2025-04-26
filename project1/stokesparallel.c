/*
Shun Fujita
ESAPPM 344 - Spring 2025, Project 1

SOR Method to solve stokes flow
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

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
	#pragma omp parallel shared(u, v, p, finished, max_residual) private(u_res, v_res, p_res)
	for (int i = 0; i < K; ++i) {
		max_residual = 0.0;

		/** udpate the u values **/ 
		// black pass for left boundary
		#pragma omp for
		for (int k=0;k<N-1; k+=2) {
			if (k == 0) { // bottom left
				u_res = mu * (-4*u[0][0] + u[1][0] + u[0][1]) - 2*dy*(p[0][0] - P);
			}
			else if (k == N-2) { // top left
				u_res = mu * (-4*u[0][N-2] + u[1][N-2] + u[0][N-3]) - 2*dy*(p[0][N-2] - P);
			}
			else { // interior
				u_res = mu * (-3*u[0][k] + u[1][k] + u[0][k-1] + u[0][k+1]) - 2*dy*(p[0][k] - P);
			}

			u[0][k] += omega*u_res;
			max_residual = fmax(max_residual, fabs(u_res));
		}
		
		// black pass for bottom and top boundary
		#pragma omp for
		for (int j=0; j<N; j+=2) {
			u_res = mu * (-5*u[j][0] + u[j-1][0] + u[j][1] + u[j+1][0]) - dy * (p[j][0] - p[j-1][0]);
			u[j][0] += omega*u_res;
			max_residual = fmax(max_residual, fabs(u_res));	

			u_res = mu * (-5*u[j][N-2] + u[j-1][N-2] + u[j][N-3] + u[j+1][N-2]) - dy * (p[j][N-2] - p[j-1][N-2]);
			u[j][N-2] += omega * u_res;
			max_residual = fmax(max_residual, fabs(u_res));	
		}

		// black interior pass
		#pragma omp for collapse(2)
		for (int j=1; j<N; ++j) {
			int k_start = (j & 1) ? 1 : 2;
			for (int k_start; k_start < N-2; k+=2) {
				u_res = mu * (- 4*u[j][k] + u[j-1][k] + u[j+1][k] + u[j][k-1] + u[j][k+1]) - dy * (p[j][k] - p[j-1][k]);
				u[j][k] += omega*u_res;
				max_residual = fmax(max_residual, fabs(u_res));		
			}
		} 

		// black pass for right boundary
		#pragma omp for
		for (int k=0; k<N-1; k+=2) { // bottom right
			if (k == 0) {
				u_res = mu * (-4*u[N-1][0] + u[N-2][0] + u[N-1][1]) + 2 * dy * (p[N-2][0]);
			}
			else if (k == N-2) { // top right
				u_res = mu * (-4*u[N-1][N-2] + u[N-2][N-2] + u[N-1][N-3]) + 2 * dy * (p[N-2][N-2]);
			}
			else {
				u_res = mu * (-3*u[N-1][k] + u[N-2][k] + u[N-1][k-1] + u[N-1][k+1]) + 2 * dy * (p[N-2][k]);
			}

			u[N-1][k] = omega*u_res;
			max_residual = fmax(max_residual, fabs(u_res));	
		}

		// red pass for left boundary
		#pragma omp for
		for (int k=1;k<N-1; k+=2) {
			if (k == N-2) { // top left
				u_res = mu * (-4*u[0][N-2] + u[1][N-2] + u[0][N-3]) - 2*dy*(p[0][N-2] - P);
			}
			else { // interior
				u_res = mu * (-3*u[0][k] + u[1][k] + u[0][k-1] + u[0][k+1]) - 2*dy*(p[0][k] - P);
			}

			u[0][k] += omega*u_res;
			max_residual = fmax(max_residual, fabs(u_res));
		}

		// red pass for bottom and top boundary
		#pragma omp for
		for (int j=1; j<N; j+=2) {
			u_res = mu * (-5*u[j][0] + u[j-1][0] + u[j][1] + u[j+1][0]) - dy * (p[j][0] - p[j-1][0]);
			u[j][0] += omega*u_res;
			max_residual = fmax(max_residual, fabs(u_res));	

			u_res = mu * (-5*u[j][N-2] + u[j-1][N-2] + u[j][N-3] + u[j+1][N-2]) - dy * (p[j][N-2] - p[j-1][N-2]);
			u[j][N-2] += omega * u_res;
			max_residual = fmax(max_residual, fabs(u_res));	
		}

		// red interior pass
		#pragma omp for collapse(2)
		for (int j=1; j<N; ++j) {
			int k_start = (j & 1) ? 2 : 1; // swap 1 and 2 
			for (int k = k_start; k<N-2; k+=2) {
				u_res = mu * (- 4*u[j][k] + u[j-1][k] + u[j+1][k] + u[j][k-1] + u[j][k+1]) - dy * (p[j][k] - p[j-1][k]);
				u[j][k] += omega*u_res;
				max_residual = fmax(max_residual, fabs(u_res));		
			}
		} 
		
		// red pass for right boundary
		#pragma omp for
		for (int k=1; k<N-1; k+=2) { // bottom right
			if (k == N-2) { // top right
				u_res = mu * (-4*u[N-1][N-2] + u[N-2][N-2] + u[N-1][N-3]) + 2 * dy * (p[N-2][N-2]);
			}
			else {
				u_res = mu * (-3*u[N-1][k] + u[N-2][k] + u[N-1][k-1] + u[N-1][k+1]) + 2 * dy * (p[N-2][k]);
			}

			u[N-1][k] = omega*u_res;
			max_residual = fmax(max_residual, fabs(u_res));	
		}


		/** udpate the v values **/ 
		// black pass for left boundary
		#pragma omp for
		for (int k=0;k<N; k+=2) {
			if (k == 0 || k == N-1) { // bottom left or top left
				v_res = 0;
			}
			else { // interior
				v_res = mu * (-3*v[0][k] + v[1][k] + v[0][k+1] + v[0][k-1]) - dx * (p[0][k] - p[0][k-1]);
			}

			v[0][k] += omega*v_res;
			max_residual = fmax(max_residual, fabs(v_res));	
		}	

		// black pass for bottom, top
		#pragma omp for
		for (int j=0; j<N-1; j+=2) {
			v[j][0] = 0; // bottom
			v[j][N-1] = 0; // top
		}
		
		// black pass for interior
		#pragma omp for collapse(2)
		for (int j=1; j<N-1; ++j) {
			int k_start = (j & 1) : 1 ? 2;
			for (int k = k_start; k<N-2; k+=2) {
				v_res = mu * (- 4*v[j][k] + v[j-1][k] + v[j+1][k]+ v[j][k+1] + v[j][k-1]) - dx*(p[j][k] - p[j][k-1]);
				v[j][k] += omega*v_res;
				max_residual = fmax(max_residual, fabs(v_res));		
			}
		}

		// black pass for right boundary
		#pragma omp for
		for (int k=0; k<N; k+=2) { // bottom right
			if (k == 0 || k == N-1) {
				v_res = 0;
			}
			else { // interior
				v_res = mu * (-3*v[N-2][k] + v[N-3][k] + v[N-2][k+1] + v[N-2][k-1]) - dx * (p[N-2][k] - p[N-2][k-1]);
			}

			v[j][k] += omega*v_res; 
			max_residual = fmax(max_residual, fabs(u_res));	
		}
		
		// red pass for bottom, top
		#pragma omp for
		for (int j=1; j<N-1; j+=2) {
			v[j][0] = 0; // bottom
			v[j][N-1] = 0; // top
		}

		// red pass for left boundary
		#pragma omp for
		for (int k=1;k<N; k+=2) {
			if (k == N-1) { // top left
				v_res = 0;
			}
			else { // interior
				v_res = mu * (-3*v[0][k] + v[1][k] + v[0][k+1] + v[0][k-1]) - dx * (p[0][k] - p[0][k-1]);
			}

			v[0][k] += omega*v_res;
			max_residual = fmax(max_residual, fabs(v_res));	
		}


		// red pass for bottom, interior, and top
		#pragma omp for collapse(2)
		for (int j=0; j<N-1; j+=2) {
			int k_start = (j & 1) : 2 ? 1;
			for (int k = k_start; k<N-2; k+=2) {
				v_res = mu * (- 4*v[j][k] + v[j-1][k] + v[j+1][k]+ v[j][k+1] + v[j][k-1]) - dx*(p[j][k] - p[j][k-1]);
				v[j][k] += omega*v_res;
				max_residual = fmax(max_residual, fabs(v_res));		
			}
		}




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



