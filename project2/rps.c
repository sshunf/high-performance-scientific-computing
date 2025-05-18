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
            double coeff = 1.0 / (1 + alpha);
            local_U[i][j] = coeff * sigma;

            sigma = (double)rand() / RAND_MAX;
            local_V[i][j] = coeff * sigma;

            sigma = (double)rand() / RAND_MAX;
            local_W[i][j] = coeff * sigma;
        }
    }
    
    double (*D_x)[N] = malloc(sizeof(*D_x) * local_N);
    double (*D_y)[N] = malloc(sizeof(*D_y) * local_N);

    // initialize subdiagonal, diagonal, and superdiagonal
    // (I - dt/2 * D_x) and (I - dt/2 * D_y)
    double* ld = malloc(N*sizeof(double));
    double* d = malloc(N*sizeof(double));
    double* ud = malloc(N*sizeof(double));
    
    double dt2 = 0.5 * dt;
    double (*D)[N] = NULL;
    double lambda = alpha / (dx*dx);
    // initialize D matrix and diagonals and broadcast
    if (rank == 0) {
        D = malloc(sizeof(*D) * N);

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
    }

    // scatter
    MPI_Scatter(D, N*local_N, MPI_DOUBLE, D_x, N*local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // TODO: maybe allocate a another D 2d array and then call transpose_matrix to D
    if (rank == 0) {
        D[0][0] = lambda; D[0][1] = -lambda; D[N-1][N-2] = -lambda; D[N-1][N-1] = lambda; // transpose the matrix --> these act as a transpose
    }

    MPI_Scatter(D, N*local_N, MPI_DOUBLE, D_y, N*local_N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // initilaize the diagonals
        for (int i = 0; i < N; i++) {
            d[i] = 1 - dt2 * D[i][i];
            if (i < N-1) {
                ld[i] = dt2 * D[i+1][i]; 
                ud[i] = dt2 * D[i][i+1];
            }
        }
    }
    // broadcast of d, ld, ud
    MPI_Bcast(d, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(ld, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(ud, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // initialize variables for explicit steps
    double rho_U; double rho_V; double rho_W;
    double Uxx; double Vxx; double Wxx; double Uyy; double Vyy; double Wyy;

    // initialize parameters for implicit steps
    double ld_copy[N], d_copy[N], ud_copy[N], uud[N];
    int pivot[N];

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

        for (int i = 0; i < local_N; ++i) {
            for (int j = 0; j < N; ++j) {
                printf("local_U[%d][%d]: %.3f | ", i, j, local_U[i][j]);
            }
            printf("\n");
        }

        printf("\n");
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < local_N; ++j) {
                printf("local_Ut[%d][%d]: %.3f | ", i, j, local_Ut[i][j]);
            }
            printf("\n");
        }
        
        // blocking scatter + gather for all processes 
        int block = local_N * local_N;
        // MPI_Scatter(local_Ut, block, MPI_DOUBLE, &local_Ur[rank*local_N][0], block, MPI_DOUBLE, rank, MPI_COMM_WORLD);
        // MPI_Scatter(local_Vt, block, MPI_DOUBLE, &local_Vr[rank*local_N][0], block, MPI_DOUBLE, rank, MPI_COMM_WORLD);
        // MPI_Scatter(local_Wt, block, MPI_DOUBLE, &local_Wr[rank*local_N][0], block, MPI_DOUBLE, rank, MPI_COMM_WORLD);

        // flatten the 2d arrays into pointers for MPI
        double *sendU = &local_Ut[0][0];
        double *recvU = &local_Ur[0][0];
        double *sendV = &local_Vt[0][0];
        double *recvV = &local_Vr[0][0];
        double *sendW = &local_Wt[0][0];
        double *recvW = &local_Wr[0][0];

        // scatter + gather
        MPI_Alltoall(sendU, block, MPI_DOUBLE, recvU, block, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Alltoall(sendV, block, MPI_DOUBLE, recvV, block, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Alltoall(sendW, block, MPI_DOUBLE, recvW, block, MPI_DOUBLE, MPI_COMM_WORLD);

        printf("after gather\n");
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < local_N; ++j) {
                printf("local_Ur[%d][%d]: %.3f | ", i, j, local_Ur[i][j]);
            }
            printf("\n");
        }
        
        // transpose each block in received buffer
        // transpose_blocks(local_N, local_Ur, world_size);
        // transpose_blocks(local_N, local_Vr, world_size);
        // transpose_blocks(local_N, local_Wr, world_size);

        move_blocks(N, local_N, world_size, local_Ur, local_U);
        move_blocks(N, local_N, world_size, local_Vr, local_V);
        move_blocks(N, local_N, world_size, local_Wr, local_W);

        printf("after moving blocks\n");
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


        printf("after solve\n");
        for (int i = 0; i < local_N; ++i) {
            for (int j = 0; j < N; ++j) {
                printf("local_Ur[%d][%d]: %.10f | ", i, j, local_U[i][j]);
            }
            printf("\n");
        }

        // step 3 - explicit y update
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < local_N; j++) {
                rho_U = local_Ur[i][j] * (1 - local_Ur[i][j] - alpha * local_Wr[i][j]);
                rho_V = local_Vr[i][j] * (1 - local_Vr[i][j] - alpha * local_Ur[i][j]);
                rho_W = local_Wr[i][j] * (1 - local_Wr[i][j] - alpha * local_Vr[i][j]);
                // continue
            }
        }

        // transpose array + MPI scatter

        // step 4 implicit x update (using lapack)
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
    free(D_x);
    free(D_y);
    free(ld);
    free(d);
    free(ud);

    if (rank == 0) {
        free(U);
        free(V);
        free(W);
        free(D);
    }

    MPI_Finalize();
    return 0;
}