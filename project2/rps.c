#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    if (argc < 4) {
        printf("Not enough arguments\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // parse inputs
    int N = atoi(argv[1]);
    double alpha = atof(argv[2]);
    int M = atoi(argv[3]);
    int seed;
    if (argc == 5) {
        seed = atoi(argv[4]);
    } else {
        // generate a seed
        seed = 42;  // Example default seed
    }

    // get the MPI info (number of processes and world rank)
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // create the subgroups for U, V, and W
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    // compute number of processors per group
    int u_num_proc = world_size / N;
    int v_num_proc = u_num_proc;
    int w_num_proc = u_num_proc;

    // Adjust the number of processes for groups based on remainder
    if (u_num_proc % 3 == 1) {
        u_num_proc++;
    } else if (u_num_proc % 3 == 2) {
        u_num_proc++;
        v_num_proc++;
    }

    int uranks[u_num_proc];
    int vranks[v_num_proc];
    int wranks[w_num_proc];

    // allocate processes to U, V, and W
    int offset = 0;
    for (int i = 0; i < u_num_proc; ++i) uranks[i] = offset++;
    for (int i = 0; i < v_num_proc; ++i) vranks[i] = offset++;
    for (int i = 0; i < w_num_proc; ++i) wranks[i] = offset++;

    // Define groups for each communicator
    MPI_Group u_group, v_group, w_group;
    MPI_Group_incl(world_group, u_num_proc, uranks, &u_group);
    MPI_Group_incl(world_group, v_num_proc, vranks, &v_group);
    MPI_Group_incl(world_group, w_num_proc, wranks, &w_group);

    // Create communicators for U, V, W
    MPI_Comm u_comm, v_comm, w_comm;
    MPI_Comm_create(MPI_COMM_WORLD, u_group, &u_comm);
    MPI_Comm_create(MPI_COMM_WORLD, v_group, &v_comm);
    MPI_Comm_create(MPI_COMM_WORLD, w_group, &w_comm);

    // Finalize MPI
    MPI_Finalize();
    return 0;
}