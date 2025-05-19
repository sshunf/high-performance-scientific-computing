/*
ES_APPM 344 - Project 2
Shun Fujita

Solves the Rock-paper-scissors model problem using ADI + MPI on distributed cluster
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

// lapack function prototypes
extern void dgttrf_(int *n, double *dl, double *d, double *du, double *uud, int *pivot, int *info);
extern void dgttrs_(char *trans, int *n, int *nrhs,
                    double *dl, double *d, double *du, double *uud,
                    int *pivot, double *b, int *ldb, int *info);


// helper function to transpose matrix
void transpose_matrix(int Nx, int Ny, double (*matrix)[Ny], double (*t_matrix)[Nx]) {
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            t_matrix[j][i] = matrix[i][j];
        }
    }
}

// Helper to move square blocks into a row-wise layout for column-major reconstruction
// N: total number of columns in destination matrix
void move_blocks(int N, int block_width, int num_blocks, double original[][block_width], double dest[][N]) {
    for (int b = 0; b < num_blocks; b++) {
        int offset = b * block_width;

        for (int i = 0; i < block_width; i++) {
            for (int j = 0; j < block_width; j++) {
                dest[i][offset + j] = original[offset + i][j];
            }
        }
    } 
}
	
int main(int argc, char *argv[]) {
    double start_time, end_time;

    MPI_Init(&argc, &argv);
    #ifdef DO_ERROR_CHECKING
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    #endif

    // start timer
    start_time = MPI_Wtime();

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
    double T = 200.0;
    double dt = T / M;
    
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

    // printf("\ninitialization of local_U\n");
    // set initial conditions
    double coeff = 1.0 / (1 + alpha); 
    for (int i = 0; i < local_N; i++) {
        for (int j = 0; j < N; j++) {
            sigma = (double)rand() / RAND_MAX;
            local_U[i][j] = coeff * sigma;

            sigma = (double)rand() / RAND_MAX;
            local_V[i][j] = coeff * sigma;

            sigma = (double)rand() / RAND_MAX;
            local_W[i][j] = coeff * sigma;
        }
    }

    // initialize subdiagonal, diagonal, and superdiagonal
    // (I - dt/2 * D_x) and (I - dt/2 * D_y)
    double* ld = malloc(N*sizeof(double));
    double* d = malloc(N*sizeof(double));
    double* ud = malloc(N*sizeof(double));
    
    double dt2 = 0.5 * dt;
    double lambda = alpha / (dx*dx);

    if (rank == 0) {// initilaize the diagonals
        double a = dt2 * lambda; /* a = (Δt/2)·λ */
        for (int i = 0; i < N; ++i) {
            d[i] = 1 + 2*a; /* 1 – (-2λ)Δt/2 */
            if (i < N-1) {
                ld[i] = -a; /* –λΔt/2 */
                ud[i] = -a;
            }
        }
    }
    // broadcast of d, ld, ud
    MPI_Bcast(d, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(ld, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(ud, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // initialize variables for explicit steps
    double rho_U; double rho_V; double rho_W;
    double U2; double V2; double W2;

    // initialize parameters for implicit steps
    double ld_copy[N], d_copy[N], ud_copy[N], uud[N];
    int pivot[N];

    double diag = -lambda;

    // initialize 2d grid for gathering
    double (*U)[N] = NULL;
    double (*V)[N] = NULL;
    double (*W)[N] = NULL;

    // initialize solver parameters
    char tr = 'N';
    int nrhs = local_N;
    int ldb  = N;
    double *rhs_list[3] = { &local_U[0][0], &local_V[0][0], &local_W[0][0] };

    // MPI Gather to rank 0 to write initial state to files
    if (rank == 0) {
        U = malloc(N*sizeof(*U));
        V = malloc(N*sizeof(*V));
        W = malloc(N*sizeof(*W));
    }

    MPI_Gather(local_U, N*local_N,MPI_DOUBLE, U, N*local_N,MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(local_V, N*local_N,MPI_DOUBLE, V, N*local_N,MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(local_W, N*local_N,MPI_DOUBLE, W, N*local_N,MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        FILE *f;
        f=fopen("RPSU.out","wb"); fwrite(U, sizeof(double), N*N, f); fclose(f);
        f=fopen("RPSV.out","wb"); fwrite(V, sizeof(double), N*N, f); fclose(f);
        f=fopen("RPSW.out","wb"); fwrite(W, sizeof(double), N*N, f); fclose(f);
    }

    // start the time step loop
    for (int t = 1; t <= M; t++) {
        for (int i = 0; i < local_N; i++) {
            // step 1 - explicit x update for U, V, W
            for (int j = 0; j < N; j++) {
                rho_U = local_U[i][j] * (1 - local_U[i][j] - alpha * local_W[i][j]);
                rho_V = local_V[i][j] * (1 - local_V[i][j] - alpha * local_U[i][j]);
                rho_W = local_W[i][j] * (1 - local_W[i][j] - alpha * local_V[i][j]);
                if (j == 0) {
                    U2 = diag * local_U[i][j] + -diag * local_U[i][j+1];
                    V2 = diag * local_V[i][j] + -diag * local_V[i][j+1];
                    W2 = diag * local_W[i][j] + -diag * local_W[i][j+1];
                } else if (j == N-1) {
                    U2 = -diag * local_U[i][j-1] + diag * local_U[i][j];
                    V2 = -diag * local_V[i][j-1] + diag * local_V[i][j];
                    W2 = -diag * local_W[i][j-1] + diag * local_W[i][j];
                } else {
                    U2 = lambda * local_U[i][j-1] + diag * local_U[i][j] + lambda * local_U[i][j+1];
                    V2 = lambda * local_V[i][j-1] + diag * local_V[i][j] + lambda * local_V[i][j+1];
                    W2 = lambda * local_W[i][j-1] + diag * local_W[i][j] + lambda * local_W[i][j+1];
                }
                
                local_U[i][j] += dt2 * (U2 + rho_U);
                local_V[i][j] += dt2 * (V2 + rho_V);
                local_W[i][j] += dt2 * (W2 + rho_W);
            }
        }

        // tranpose array
        transpose_matrix(local_N, N, local_U, local_Ut);
        transpose_matrix(local_N, N, local_V, local_Vt);
        transpose_matrix(local_N, N, local_W, local_Wt);

        // blocking scatter + gather for all processes 
        int block = local_N * local_N;

        // scatter + gather
        MPI_Alltoall(&local_Ut[0][0], block, MPI_DOUBLE, &local_Ur[0][0], block, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Alltoall(&local_Vt[0][0], block, MPI_DOUBLE, &local_Vr[0][0], block, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Alltoall(&local_Wt[0][0], block, MPI_DOUBLE, &local_Wr[0][0], block, MPI_DOUBLE, MPI_COMM_WORLD);

        move_blocks(N, local_N, world_size, local_Ur, local_U);
        move_blocks(N, local_N, world_size, local_Vr, local_V);
        move_blocks(N, local_N, world_size, local_Wr, local_W);

        // step 2 - implicit y update
        memcpy(ld_copy, ld, N*sizeof(double));
        memcpy(d_copy , d , N*sizeof(double));
        memcpy(ud_copy, ud, N*sizeof(double));

        int info;

        dgttrf_(&N, ld_copy, d_copy, ud_copy, uud, pivot, &info);
        if (info != 0) {
            fprintf(stderr,"dgttrf failed (info=%d)\n",info);
            MPI_Abort(MPI_COMM_WORLD,-1);
        }

        for (int s=0; s<3; ++s) {
            dgttrs_(&tr, &N, &nrhs, ld_copy, d_copy, ud_copy, uud, pivot, rhs_list[s], &ldb, &info);
            if (info != 0) {
                fprintf(stderr,"dgttrs failed (species %d, info=%d)\n",s,info);
                MPI_Abort(MPI_COMM_WORLD,-1);
            }
        }

        // step 3 - explicit y update
        for (int i = 0; i < local_N; i++) {
            for (int j = 0; j < N; j++) {
                rho_U = local_U[i][j] * (1 - local_U[i][j] - alpha * local_W[i][j]);
                rho_V = local_V[i][j] * (1 - local_V[i][j] - alpha * local_U[i][j]);
                rho_W = local_W[i][j] * (1 - local_W[i][j] - alpha * local_V[i][j]);
                if (j == 0) {
                    U2 = diag * local_U[i][j] + -diag * local_U[i][j+1];
                    V2 = diag * local_V[i][j] + -diag * local_V[i][j+1];
                    W2 = diag * local_W[i][j] + -diag * local_W[i][j+1];
                } else if (j == N-1) {
                    U2 = -diag * local_U[i][j-1] + diag * local_U[i][j];
                    V2 = -diag * local_V[i][j-1] + diag * local_V[i][j];
                    W2 = -diag * local_W[i][j-1] + diag * local_W[i][j];
                } else {
                    U2 = lambda * local_U[i][j-1] + diag * local_U[i][j] + lambda * local_U[i][j+1];
                    V2 = lambda * local_V[i][j-1] + diag * local_V[i][j] + lambda * local_V[i][j+1];
                    W2 = lambda * local_W[i][j-1] + diag * local_W[i][j] + lambda * local_W[i][j+1];
                }
                
                local_U[i][j] += dt2 * (U2 + rho_U);
                local_V[i][j] += dt2 * (V2 + rho_V);
                local_W[i][j] += dt2 * (W2 + rho_W);
            }
        }

        // transpose array + MPI scatter
        transpose_matrix(local_N, N, local_U, local_Ut);
        transpose_matrix(local_N, N, local_V, local_Vt);
        transpose_matrix(local_N, N, local_W, local_Wt);

        // scatter + gather
        MPI_Alltoall(&local_Ut[0][0], block, MPI_DOUBLE, &local_Ur[0][0], block, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Alltoall(&local_Vt[0][0], block, MPI_DOUBLE, &local_Vr[0][0], block, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Alltoall(&local_Wt[0][0], block, MPI_DOUBLE, &local_Wr[0][0], block, MPI_DOUBLE, MPI_COMM_WORLD);

        move_blocks(N, local_N, world_size, local_Ur, local_U);
        move_blocks(N, local_N, world_size, local_Vr, local_V);
        move_blocks(N, local_N, world_size, local_Wr, local_W);

        // step 4 implicit x update (using lapack)
        memcpy(ld_copy, ld, N*sizeof(double));
        memcpy(d_copy , d , N*sizeof(double));
        memcpy(ud_copy, ud, N*sizeof(double));

        dgttrf_(&N, ld_copy, d_copy, ud_copy, uud, pivot, &info);
        if (info != 0) {
            fprintf(stderr,"dgttrf failed (info=%d)\n",info);
            MPI_Abort(MPI_COMM_WORLD,-1);
        }

        for (int s=0; s<3; ++s) {
            dgttrs_(&tr, &N, &nrhs, ld_copy, d_copy, ud_copy, uud, pivot, rhs_list[s], &ldb, &info);
            if (info != 0) {
                fprintf(stderr,"dgttrs failed (species %d, info=%d)\n",s,info);
                MPI_Abort(MPI_COMM_WORLD,-1);
            }
        }

        // gather to write to files
        MPI_Gather(local_U, N*local_N, MPI_DOUBLE, U, N*local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(local_V, N*local_N, MPI_DOUBLE, V, N*local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(local_W, N*local_N, MPI_DOUBLE, W, N*local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0 && (t % (M / 10)) == 0) {
            FILE *fU = fopen("RPSU.out", "ab");
            fwrite(U, sizeof(double), N * N, fU);
            fclose(fU);
        
            FILE *fV = fopen("RPSV.out", "ab");
            fwrite(V, sizeof(double), N * N, fV);
            fclose(fV);
        
            FILE *fW = fopen("RPSW.out", "ab");
            fwrite(W, sizeof(double), N * N, fW);
            fclose(fW);
        }
    }
    
    // record time elapsed
    end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Elasped time: %f seconds \n", end_time - start_time);
    }

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
    free(ld);
    free(d);
    free(ud);

    if (rank == 0) {
        free(U);
        free(V);
        free(W);
    }

    MPI_Finalize();
    return 0;
}