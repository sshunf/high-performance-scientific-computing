#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void transpose_matrix(double (*matrix)[N], int N) {
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            double temp = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = temp;
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

    double (*D)[N] = malloc(sizeof(*D) * local_N);
    // initialize the tridiagonal matrix and scatter - how to account for boundary conditions?
    if (rank == 0) {
        double lambda = alpha * dt / (dx*dx);
        double (*D_x)[N] = calloc(N, sizeof(*D_x));
        D_x[0][0] = 1;
        D_x[N-1][N-2] = 0; // TODO: fix these boundary conditions
        D_x[N-1][N-1] = 0;
        for (int i = 1; i < N-1; i++) {
            for (int j = 0; j < N; j++) {
                if (i == j) {
                    D_x[i][j] = -2*lambda;
                } else if (i - 1 == j || i + 1 == j) {
                    D_x[i][j] = lambda;
                }
            }
        }

        // scatter
        MPI_Scatter(D_x, N*N, MPI_DOUBLE, D, N*local_N, MPI_DOUBLE, MPI_COMM_WORLD);
    }

    // calculate number of rows each process is responsible for
    int local_N = N / world_size;

    // initialize local arrays (localN x N)
    double (*local_U)[N] = malloc(sizeof(*local_U) * local_N);
    double (*local_V)[N] = malloc(sizeof(*local_V) * local_N);
    double (*local_W)[N] = malloc(sizeof(*local_W) * local_N);

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

    // start the time step loop
    for (int t = 0; t < M; t++) {
        // step 1 - explicit x update
        for (int i = 0; i < local_N; i++) {
            for (int j = 0; j < N; j++) {
                continue;
            }
        }

        // tranpose array + MPI scatter

        // step 2 - implicit y update (using lapack)

        // step 3 - explicit y update

        // transpose array + MPI scatter

        // step 4 implicit x update (using lapack)
    }

    // MPI Gather to rank 0 + write to files

    
    MPI_Finalize();
    return 0;
}