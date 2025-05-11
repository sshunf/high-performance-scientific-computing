#include <stdio.h>
#include <mpi.h>

int main(int argc, char* argv[]) {
	if (argc != 4) {
		printf("Not enough arguments\n");
		return -1;
	}	
	MPI_Init(&argc, &argv);

	int N = atoi(argv[1]);
	double  alpha = atod(argv[2]);
	int M = atoi(argv[3]);

	MPI_Finalize();
	return 0;
}