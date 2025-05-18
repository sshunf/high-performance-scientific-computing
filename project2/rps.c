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

// helper function to transpose the blocks within a matrix - assumes more rows that columns
// inputs:
//  N: block_width (same as block_height) we assume square here
//  num_blocks: number of processes or blocks in matrix
void transpose_blocks(int N, double matrix[][N], int num_blocks) {
    for (int b = 0; b < num_blocks; b++) {
        int row_offset = b * N;

        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < N; j++) {
                double temp = matrix[row_offset + i][j];
                matrix[row_offset + i][j] = matrix[row_offset + j][i];
                matrix[row_offset + j][i] = temp;
            }
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
    MPI_Init(&argc, &argv);
    #ifdef DO_ERROR_CHECKING
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    #endif

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
    double dt = (double) M / 200; // T = 200
    
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

    printf("\ninitialization of local_U\n");
    // set initial conditions
    for (int i = 0; i < local_N; i++) {
        for (int j = 0; j < N; j++) {
            sigma = (double)rand() / RAND_MAX;
            double coeff = 1.0 / (1 + alpha);
            local_U[i][j] = coeff * sigma;

            sigma = (double)rand() / RAND_MAX;
            local_V[i][j] = coeff * sigma;

            sigma = (double)rand() / RAND_MAX;
            local_W[i][j] = coeff * sigma;

            printf("local_U[%d][%d]: %.2f | ", i, j, local_U[i][j]);
        }
        printf("\n");
    }

    // initialize subdiagonal, diagonal, and superdiagonal
    // (I - dt/2 * D_x) and (I - dt/2 * D_y)
    double* ld = malloc(N*sizeof(double));
    double* d = malloc(N*sizeof(double));
    double* ud = malloc(N*sizeof(double));
    
    double dt2 = 0.5 * dt;
    double (*D)[N] = malloc(sizeof(*D) * N);;
    double lambda = alpha / (dx*dx);

    double D_init[N][N];
    // initialize D matrix and diagonals and broadcast
    if (rank == 0) {
        // first row
        memset(D_init, 0, sizeof D_init);
        for (int i = 0; i < N; ++i) {
            D_init[i][i] = -2.0 * lambda;
            if (i > 0)   D_init[i][i-1] = lambda;
            if (i < N-1) D_init[i][i+1] = lambda;
        }
    }

    // scatter
    MPI_Scatter(&D_init, N*local_N, MPI_DOUBLE, D, N*local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {// initilaize the diagonals
        double a = dt2 * lambda;          /*  a =  (Δt/2)·λ            */
        for (int i = 0; i < N; ++i) {
            d[i] = 1 + 2*a;               /* main diagonal: 1 – (-2λ)Δt/2 */
            if (i < N-1) {
                ld[i] = -a;                /* sub- & super- diagonal: –λΔt/2 */
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

    // start the time step loop
    for (int t = 0; t < 20; t++) { // TODO: change this back to M iterations
        for (int i = 0; i < local_N; i++) {
            // step 1 - explicit x update for U, V, W
            for (int j = 0; j < N; j++) {
                rho_U = local_U[i][j] * (1 - local_U[i][j] - alpha * local_W[i][j]);
                rho_V = local_V[i][j] * (1 - local_V[i][j] - alpha * local_U[i][j]);
                rho_W = local_W[i][j] * (1 - local_W[i][j] - alpha * local_V[i][j]);
                if (j == 0) {
                    U2 = D[i][j] * local_U[i][j] + D[i][j+1] * local_U[i][j+1];
                    V2 = D[i][j] * local_V[i][j] + D[i][j+1] * local_V[i][j+1];
                    W2 = D[i][j] * local_W[i][j] + D[i][j+1] * local_W[i][j+1];
                } else if (j == N-1) {
                    U2 = D[i][j-1] * local_U[i][j-1] + D[i][j] * local_U[i][j];
                    V2 = D[i][j-1] * local_V[i][j-1] + D[i][j] * local_V[i][j];
                    W2 = D[i][j-1] * local_W[i][j-1] + D[i][j] * local_W[i][j];
                } else {
                    U2 = D[i][j-1] * local_U[i][j-1] + D[i][j] * local_U[i][j] + D[i][j+1] * local_U[i][j+1];
                    V2 = D[i][j-1] * local_V[i][j-1] + D[i][j] * local_V[i][j] + D[i][j+1] * local_V[i][j+1];
                    W2 = D[i][j-1] * local_W[i][j-1] + D[i][j] * local_W[i][j] + D[i][j+1] * local_W[i][j+1];
                }
                
                local_U[i][j] += dt2 * (U2 + rho_U);
                local_V[i][j] += dt2 * (V2 + rho_V);
                local_W[i][j] += dt2 * (W2 + rho_W);
            }
        }

        printf("\nafter explicit step (step 1)\n");
        for (int i = 0; i < local_N; ++i) {
            for (int j = 0; j < N; ++j) {
                printf("local_U[%d][%d]: %.3f | ", i, j, local_U[i][j]);
            }
            printf("\n");
        }

        // tranpose array
        transpose_matrix(local_N, N, local_U, local_Ut);
        transpose_matrix(local_N, N, local_V, local_Vt);
        transpose_matrix(local_N, N, local_W, local_Wt);

        printf("\n");
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < local_N; ++j) {
                printf("local_Ut[%d][%d]: %.3f | ", i, j, local_Ut[i][j]);
            }
            printf("\n");
        }
        
        // blocking scatter + gather for all processes 
        int block = local_N * local_N;

        // scatter + gather
        MPI_Alltoall(&local_Ut[0][0], block, MPI_DOUBLE, &local_Ur[0][0], block, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Alltoall(&local_Vt[0][0], block, MPI_DOUBLE, &local_Vr[0][0], block, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Alltoall(&local_Wt[0][0], block, MPI_DOUBLE, &local_Wr[0][0], block, MPI_DOUBLE, MPI_COMM_WORLD);

        printf("\nafter gather step 1\n");
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < local_N; ++j) {
                printf("local_Ur[%d][%d]: %.3f | ", i, j, local_Ur[i][j]);
            }
            printf("\n");
        }

        move_blocks(N, local_N, world_size, local_Ur, local_U);
        move_blocks(N, local_N, world_size, local_Vr, local_V);
        move_blocks(N, local_N, world_size, local_Wr, local_W);

        printf("\nafter moving blocks step 1\n");
        for (int i = 0; i < local_N; ++i) {
            for (int j = 0; j < N; ++j) {
                printf("local_U[%d][%d]: %.3f | ", i, j, local_U[i][j]);
            }
            printf("\n");
        }

        // step 2 - implicit y update (using lapack)

        // make a copy of ld, d, ud
        int info;
        char tr = 'N';

        // for each RHS matrix (bU, bV, bW), refactor A and solve
        for (int i = 0; i < 3; i++) {
            printf("solve loop\n");
            memcpy(ld_copy, ld, N * sizeof(double));
            memcpy(d_copy, d, N * sizeof(double));
            memcpy(ud_copy, ud, N * sizeof(double));

            double *rhs;
            if (i == 0) rhs = &local_U[0][0];
            else if (i == 1) rhs = &local_V[0][0];
            else rhs = &local_W[0][0];

            dgttrf_(&N, ld_copy, d_copy, ud_copy, uud, pivot, &info);
            if (info != 0) {
                fprintf(stderr, "dgttrf_ failed with info = %d\n", info);
                MPI_Abort(MPI_COMM_WORLD, -1);
            }

            dgttrs_(&tr, &N, &local_N, ld_copy, d_copy, ud_copy, uud, pivot, rhs, &N, &info);
            if (info != 0) {
                fprintf(stderr, "dgttrs_ failed with info = %d\n", info);
                MPI_Abort(MPI_COMM_WORLD, -1); 
            }
        }


        printf("\nafter solve step 2\n");
        for (int i = 0; i < local_N; ++i) {
            for (int j = 0; j < N; ++j) {
                printf("local_U[%d][%d]: %.3f | ", i, j, local_U[i][j]);
            }
            printf("\n");
        }

        // step 3 - explicit y update
        for (int i = 0; i < local_N; i++) {
            for (int j = 0; j < N; j++) {
                rho_U = local_U[i][j] * (1 - local_U[i][j] - alpha * local_W[i][j]);
                rho_V = local_V[i][j] * (1 - local_V[i][j] - alpha * local_U[i][j]);
                rho_W = local_W[i][j] * (1 - local_W[i][j] - alpha * local_V[i][j]);
                if (j == 0) {
                    U2 = D[i][j] * local_U[i][j] + D[i][j+1] * local_U[i][j+1];
                    V2 = D[i][j] * local_V[i][j] + D[i][j+1] * local_V[i][j+1];
                    W2 = D[i][j] * local_W[i][j] + D[i][j+1] * local_W[i][j+1];
                } else if (j == N-1) {
                    U2 = D[i][j-1] * local_U[i][j-1] + D[i][j] * local_U[i][j];
                    V2 = D[i][j-1] * local_V[i][j-1] + D[i][j] * local_V[i][j];
                    W2 = D[i][j-1] * local_W[i][j-1] + D[i][j] * local_W[i][j];
                } else {
                    U2 = D[i][j-1] * local_U[i][j-1] + D[i][j] * local_U[i][j] + D[i][j+1] * local_U[i][j+1];
                    V2 = D[i][j-1] * local_V[i][j-1] + D[i][j] * local_V[i][j] + D[i][j+1] * local_V[i][j+1];
                    W2 = D[i][j-1] * local_W[i][j-1] + D[i][j] * local_W[i][j] + D[i][j+1] * local_W[i][j+1];
                }
                
                local_U[i][j] += dt2 * (U2 + rho_U);
                local_V[i][j] += dt2 * (V2 + rho_V);
                local_W[i][j] += dt2 * (W2 + rho_W);
            }
        }

        printf("\nafter explicit step (step 3)\n");
        for (int i = 0; i < local_N; ++i) {
            for (int j = 0; j < N; ++j) {
                printf("local_U[%d][%d]: %.3f | ", i, j, local_U[i][j]);
            }
            printf("\n");
        }

        // transpose array + MPI scatter
        transpose_matrix(local_N, N, local_U, local_Ut);
        transpose_matrix(local_N, N, local_V, local_Vt);
        transpose_matrix(local_N, N, local_W, local_Wt);

        printf("after transpose step 3\n");
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < local_N; ++j) {
                printf("local_Ut[%d][%d]: %.3f | ", i, j, local_Ut[i][j]);
            }
            printf("\n");
        }

        // scatter + gather
        MPI_Alltoall(&local_Ut[0][0], block, MPI_DOUBLE, &local_Ur[0][0], block, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Alltoall(&local_Vt[0][0], block, MPI_DOUBLE, &local_Vr[0][0], block, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Alltoall(&local_Wt[0][0], block, MPI_DOUBLE, &local_Wr[0][0], block, MPI_DOUBLE, MPI_COMM_WORLD);

        printf("\nafter gather step 3\n");
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < local_N; ++j) {
                printf("local_Ur[%d][%d]: %.3f | ", i, j, local_Ur[i][j]);
            }
            printf("\n");
        }

        move_blocks(N, local_N, world_size, local_Ur, local_U);
        move_blocks(N, local_N, world_size, local_Vr, local_V);
        move_blocks(N, local_N, world_size, local_Wr, local_W);

        printf("\nafter moving blocks step 3\n");
        for (int i = 0; i < local_N; ++i) {
            for (int j = 0; j < N; ++j) {
                printf("local_U[%d][%d]: %.3f | ", i, j, local_U[i][j]);
            }
            printf("\n");
        }

        // step 4 implicit x update (using lapack)
        for (int i = 0; i < 3; i++) {
            printf("solve loop\n");
            memcpy(ld_copy, ld, N * sizeof(double));
            memcpy(d_copy, d, N * sizeof(double));
            memcpy(ud_copy, ud, N * sizeof(double));

            double *rhs;
            if (i == 0) rhs = &local_U[0][0];
            else if (i == 1) rhs = &local_V[0][0];
            else rhs = &local_W[0][0];

            dgttrf_(&N, ld_copy, d_copy, ud_copy, uud, pivot, &info);
            if (info != 0) {
                fprintf(stderr, "dgttrf_ failed with info = %d\n", info);
                MPI_Abort(MPI_COMM_WORLD, -1);
            }

            dgttrs_(&tr, &N, &local_N, ld_copy, d_copy, ud_copy, uud, pivot, rhs, &N, &info);
            if (info != 0) {
                fprintf(stderr, "dgttrs_ failed with info = %d\n", info);
                MPI_Abort(MPI_COMM_WORLD, -1); 
            }
        }

        printf("after solve step 4\n");
        for (int i = 0; i < local_N; ++i) {
            for (int j = 0; j < N; ++j) {
                printf("local_U[%d][%d]: %.3f | ", i, j, local_U[i][j]);
            }
            printf("\n");
        }
    }
    
    double (*U)[N] = NULL;
    double (*V)[N] = NULL;
    double (*W)[N] = NULL;

    // MPI Gather to rank 0
    if (rank == 0) {
        U = malloc(N*sizeof(*U));
        V = malloc(N*sizeof(*V));
        W = malloc(N*sizeof(*W));
    }

    MPI_Gather(local_U, N*local_N, MPI_DOUBLE, U, N*local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(local_V, N*local_N, MPI_DOUBLE, V, N*local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(local_W, N*local_N, MPI_DOUBLE, W, N*local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // write result to files
    // FILE *fU = fopen("RPSU.out", "wb");
    // fwrite(U, sizeof(double), N*N, fU);
    // fclose(fU);

    // FILE *fV = fopen("RPSV.out", "wb");
    // fwrite(V, sizeof(double), N*N, fV);
    // fclose(fU);

    // FILE *fW = fopen("RSPW.out", "wb");
    // fwrite(W, sizeof(double), N*N, fW);
    // fclose(fW);


    if (rank == 0) {
        // Write U to text file
        FILE *fU = fopen("RPSU.txt", "w");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                fprintf(fU, "%.10f ", U[i][j]);
            }
            fprintf(fU, "\n");
        }
        fclose(fU);

        // Write V to text file
        FILE *fV = fopen("RPSV.txt", "w");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                fprintf(fV, "%.10f ", V[i][j]);
            }
            fprintf(fV, "\n");
        }
        fclose(fV);

        // Write W to text file
        FILE *fW = fopen("RPSW.txt", "w");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                fprintf(fW, "%.10f ", W[i][j]);
            }
            fprintf(fW, "\n");
        }
        fclose(fW);
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
    free(D);
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