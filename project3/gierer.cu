#include <stdio.h>
#include <stdlib.h>
#include <cufft.h>
#include <cuda_runtime.h>

#define NUM_BLOCKS 8;
#define NUM_THREADS_PER_BLOCK 16;
#define REAL_SIZE = N*N;
#define COMPLEX_SIZE = N*(N/2+1);

// error handler
static void HANDLEERROR( cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file,line);
        exit(1);
    }
}
#define HandleError(err) (HANDLEERROR(err, __FILE__, __LINE__))

__global__ void runge_kutta_advance(double* dev_u, double* dev_v, int step) {
    // TODO
    continue;
}

// helper function to rescale matrix after inverse fft
__global__ void rescale(double* dev_matrix, double* factor, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dev_matrix[i * cols + j] *= scalar;
        }
    }
}

// function to swap pointers for runge-kutta step
inline void swap(double*& A, double*& B) {
    double* temp = A;
    A = B;
    B = temp;
}

int main(int argc, char* argv[]) {
    if (argc < 9) {
        printf("not enough arguments!\n");
    }

    // parse arguments
    int N = atoi(argv[1]);
    int D_u = atoi(argv[2]);
    int D_v = atoi(argv[3]);
    double a = atof(argv[4]);
    double b = atof(argv[5]);
    double c = atof(argv[6]);
    double eps = atof(argv[7]);
    int K = atoi(argv[8]);

    // parse seed
    long seed;
    if (argc == 9) {
        seed = atol(argv[9]);
    } else {
        seed = 42;
    }

    printf("N: %d\n", N);
    printf("D_u: %d\n", D_u);
    printf("D_v: %d\n", D_v);
    printf("a: %.2f\n", a);
    printf("b: %.2f\n", b);
    printf("c: %.2f\n", c);
    printf("eps: %.2f\n", eps);
    printf("K: %d\n", K);
    printf("seed: %ld\n", seed);

    // set the seed
    srand48(seed);
    double omega;
    printf("omega: %.2f\n", omega);

    // time steps
    int T = 100; // terminal time
    double dt = (double)T / (double)K;

    // initialize matrices
    double* u = (double*)malloc(REAL_SIZE*sizeof(double));
    double* v = (double*)malloc(REAL_SIZE*sizeof(double));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            omega = drand48();
            u[i*cols + j] = (a + c)/b - 4.5*omega;
            v[i*cols + j] = (a + c)*(a + c)/(c * b)/(c * b);
        }
    }

    // initialize matrices on device
    double* dev_u, dev_v;
    cudaMalloc(&dev_u, sizeof(double) * REAL_SIZE);
    cudaMalloc(&dev_v, sizeof(double) * REAL_SIZE);

    cudaMemcpy(dev_u, sizeof(double) * REAL_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v, sizeof(double) * REAL_SIZE, cudaMemcpyHostToDevice);

    cufftDoubleComplex* dev_cu, dev_cv;
    cudaMalloc((void**)&dev_cu, sizeof(cufftDoubleComplex) * COMPLEX_SIZE);
    cudaMalloc((void**)&dev_cv, sizeof(cufftDoubleComplex) * COMPLEX_SIZE);

    // create fft plans
    cufftHandle plan_r2c, plan_c2r;
    cufftPlan2d(&plan_r2c, N, N, CUFFT_R2C);
    cufftPlan2d(&plan_c2r, N, N, CUFFT_C2R);
    
    // cufft does not normalize so make sure to divide by N*N after the inverse

    // time step loop
    for (int t = 0; t < K; t++) {
        // forward fft (t1)
        // backward fft (t1)
        // runge-kutta time step (t1)
        // swap pointers

        // forward fft (t2)
        // backward fft (t2)
        // runge-kutta time step (t2)
        // swap pointers

        // forward fft (t3)
        // backward fft (t3)
        // runge-kutta time step (t3)
        // swap pointers

        // forward fft (t4)
        // backward fft (t4)
        // runge-kutta time step (t4)
        // swap pointers
    }

    cufftDestroy(plan_r2c); cufftDestroy(plan_c2r);
    cudaFree(dev_u); cudaFree(dev_v); cudaFree(dev_cu); cudaFree(dev_cv);
    return 0;
}