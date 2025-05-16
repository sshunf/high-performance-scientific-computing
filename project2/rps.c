/*
ES_APPM 344 - Project 2
Shun Fujita

Solves Rock-paper-scissors model problem using ADI
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// helper function to transpose matrix
void transpose_matrix(int Nx, int Ny, double matrix[][Ny], double t_matrix[][Nx]) {
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            t_matrix[j][i] = matrix[i][j];
        }
    }
}
	
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    // check number of arguments
    if (argc < 4) {
        printf("Not enough arguments\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // parse arguments
    int N = atoi(argv[1]);
    double alpha = atof(argv[2]);
    int M = atoi(argv[3]);
    long seed;
    if (argc == 5) {
        seed = atoi(argv[4]);  
    } else {
        // generate a seed
        seed = 42;  // Example default seed FIX THIS
    }

    // initialize dimension [-L, L] x [-L, L] grid
    int L = 60;

    // initialize step sizes for x, y, t
    double dx = 2.0*L / (N - 1.0);
    double dy = dx;
    double dt = M / 200; // T = 200
    
    // get the MPI info (number of processes and rank)
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // seed the rng
    srand(seed + rank);
    double sigma;

    // print arguments
    printf("N: %d\n", N);
    printf("alpha: %.2f\n", alpha);
    printf("M: %d\n", M);
    printf("seed: %ld\n", seed + rank);

    // calculate number of rows each process is responsible for
    int local_N = N / world_size;

    // initialize local grid (localN x N)
    double (*local_U)[N] = malloc(sizeof(*local_U) * local_N);
    double (*local_V)[N] = malloc(sizeof(*local_V) * local_N);
    double (*local_W)[N] = malloc(sizeof(*local_W) * local_N);

    // initialize transposed grid
    double (*local_Ut)[local_N] = malloc(sizeof(*local_Ut) * N);
    double (*local_Vt)[local_N] = malloc(sizeof(*local_Vt) * N);
    double (*local_Wt)[local_N] = malloc(sizeof(*local_Wt) * N);

    // initialize receiving grids
    double (*local_Ur)[local_N] = malloc(sizeof(*local_Ur) * N);
    double (*local_Vr)[local_N] = malloc(sizeof(*local_Vr) * N);
    double (*local_Wr)[local_N] = malloc(sizeof(*local_Wr) * N);

    // set initial conditions
    for (int i = 0; i < local_N; i++) {
        for (int j = 0; j < N; j++) {
            sigma = (double)rand() / RAND_MAX;
            double init_val = 1.0 / (1 + alpha) * sigma;
            local_U[i][j] = init_val; 
            local_V[i][j] = init_val;
            local_W[i][j] = init_val;
        }
    }

    // initialize the tridiagonal matrix and scatter
    double (*D_x)[N] = malloc(sizeof(*D_x) * local_N);
    double (*D_y)[N] = malloc(sizeof(*D_y) * local_N);
    
    // initialize subdiagonal, diagonal, and superdiagonal
    // (I - dt/2 * D_x) and (I - dt/2 * D_y)
    double* ld = malloc(N*sizeof(double));
    double* d = malloc(N*sizeof(double));
    double* ud = malloc(N*sizeof(double));

    double dt2 = 0.5 * dt;
    MPI_Request request;
    MPI_Request D_request;

    if (rank == 0) {
        double lambda = alpha / (dx*dx);
        double (*D)[N] = malloc(sizeof(*D_x) * N);

        // first row
        D[0][0] = -lambda;
        D[0][1] = lambda;

        // initialize interior
        for (int i = 1; i < N-1; i++) {
            for (int j = i-1; j <= i+1; j++) {
                if (i == j) {
                    D[i][j] = -2*lambda;
                } else if (i-1 == j || i+1 == j) {
                    D[i][j] = lambda;
                }
            }
        }

        // last row
        D[N-1][N-2] = lambda;
        D[N-1][N-1] = -lambda;

        // scatter
        MPI_Iscatter(D, N*N, MPI_DOUBLE, D_x, N*local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD, &D_request);
        transpose_matrix(N, N, D, D_y);
        MPI_Iscatter(D, N*N, MPI_DOUBLE, D_y, N*local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD, &D_request);

        
        // fix this part
        // compute M_x, M_y
        for (int i = 0; i < N-1; i++) {
            for (int j = i-1; j <= i+1; j++) {
                if (i == j) {
                    M_x[i][j] = 1 - dt2 * D_x[i][j];
                    M_y[i][j] = 1 - dt2 * D_y[i][j];
                } else if (i-1 == j || i+1 == j) {
                    M_x[i][j] = dt2 * D_x[i][j];
                    M_y[i][j] = dt2 * D_y[i][j];
                }
            }
        }
        M_x[0][0] = 1 + dt2 * D[0][0];
        M_x[0][1] = dt2 * D[0][1];
        M_y[0][0] = 1 + dt2 * D[0][0];
        M_y[0][1] = dt2 * D[0][1];
        
        // unblocking broadcast of M_x, M_y
        MPI_Ibcast(M_x, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD, &M_request);
        MPI_Ibcast(M_y, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD, &M_request);
        free(D);
    }

    double rho_U; double rho_V; double rho_W;
    double Uxx; double Vxx; double Wxx;

    MPI_Wait(&D_request, MPI_STATUS_IGNORE);
    // start the time step loop
    for (int t = 0; t < M; t++) {
        for (int i = 0; i < local_N; i++) {
            // step 1 - explicit x update for U, V, W
            for (int j = 0; j < N; j++) {
                rho_U = local_U[i][j] * (1 - local_U[i][j] - alpha * local_W[i][j]);
                rho_V = local_V[i][j] * (1 - local_V[i][j] - alpha * local_U[i][j]);
                rho_W = local_W[i][j] * (1 - local_W[i][j] - alpha * local_V[i][j]);
                if (j == 0) {
                    Uxx = D_x[i][j] * local_U[i][j] + D_x[i][j+1] * local_U[i][j+1];
                    Vxx = D_x[i][j] * local_V[i][j] + D_x[i][j+1] * local_V[i][j+1];
                    Wxx = D_x[i][j] * local_W[i][j] + D_x[i][j+1] * local_W[i][j+1];
                } else if (j == N-1) {
                    Uxx = D_x[i][j-1] * local_U[i][j-1] + D_x[i][j] * local_U[i][j];
                    Vxx = D_x[i][j-1] * local_V[i][j-1] + D_x[i][j] * local_V[i][j];
                    Wxx = D_x[i][j-1] * local_W[i][j-1] + D_x[i][j] * local_W[i][j];
                } else {
                    Uxx = D_x[i][j-1] * local_U[i][j-1] + D_x[i][j] * local_U[i][j] + D_x[i][j+1] * local_U[i][j+1];
                    Vxx = D_x[i][j-1] * local_V[i][j-1] + D_x[i][j] * local_V[i][j] + D_x[i][j+1] * local_V[i][j+1];
                    Wxx = D_x[i][j-1] * local_W[i][j-1] + D_x[i][j] * local_W[i][j] + D_x[i][j+1] * local_W[i][j+1];
                }
                
                local_U[i][j] += dt2 * (Uxx + rho_U);
                local_V[i][j] += dt2 * (Vxx + rho_V);
                local_W[i][j] += dt2 * (Wxx + rho_W);
            }
        }

        // tranpose array
        transpose_matrix(local_N, N, local_U, local_Ut);
        transpose_matrix(local_N, N, local_V, local_Vt);
        transpose_matrix(local_N, N, local_W, local_Wt);

        // blocking scatter + gather for all processes TODO: change to scatter
        int block = (local_N * N) / world_size;
        MPI_Scatter(local_Ut, block, MPI_DOUBLE, &local_Ur[rank*local_N][0], block, MPI_DOUBLE, rank, MPI_COMM_WORLD);
        MPI_Scatter(local_Vt, block, MPI_DOUBLE, &local_Vr[rank*local_N][0], block, MPI_DOUBLE, rank, MPI_COMM_WORLD);
        MPI_Scatter(local_Wt, block, MPI_DOUBLE, &local_Wr[rank*local_N][0], block, MPI_DOUBLE, rank, MPI_COMM_WORLD);

        // transpose back
        transpose_matrix(N, local_N, local_Ur, local_U);
        transpose_matrix(N, local_N, local_Vr, local_V);
        transpose_matrix(N, local_N, local_Wr, local_W);

        // step 2 - implicit y update (using lapack)


        // three calls to dgtsv_ here using nrhs = localN? solution overwrites local_U, local_V, local_W


        // step 3 - explicit y update

        // transpose array + MPI scatter

        // step 4 implicit x update (using lapack)
    }

    // MPI Gather to rank 0 + write to files

    
    // free memory
    free(local_U);
    free(local_V);
    free(local_W);
    free(local_Ut);
    free(local_Vt);
    free(local_Wt);
    free(local_Ur);
    free(local_Vr);
    free(local_Wr);
    free(D_x);
    free(D_y);
    free(M_x);
    free(M_y);

    MPI_Finalize();
    return 0;
}