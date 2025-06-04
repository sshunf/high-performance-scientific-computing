#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>

// useful definitions
#define NUM_BLOCKS 8
#define NUM_THREADS_PER_BLOCK 16
#define REAL_SIZE (N*N)
#define COMPLEX_SIZE (N*(N/2+1))
#define PI 3.14159265358979323846
#define L 200.0
#define DIFF_COEFF (2.0 * M_PI / L)

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
}

// helper function to rescale matrix after inverse fft
__global__ void rescale(double* dev_matrix, double factor, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalSize = rows * cols;
    if (idx < totalSize) {
        dev_matrix[idx] *= factor;
    }
}


// function to swap pointers for runge-kutta step
inline void swap(double*& A, double*& B) {
    double* temp = A;
    A = B;
    B = temp;
}

// consider computing only u instead of both u and v
__global__ void compute_derivative(double* dev_cu, double* dev_cv, int N, double D_u, double D_v) {
    // TODO
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N || j >= N/2 + 1) return;

    int kx = (i <= N/2) ? i : i - N;
    int ky = j;

    double k_s = DIFF_COEFF * (kx * kx + ky * ky);

    int idx = i * (N/2 + 1) + j;

    cufftDoubleComplex old_u = dev_cu[idx];
    cufftDoubleComplex old_v = dev_cv[idx];

    dev_cu[idx].x = -D_u * k_s * old_u.x;
    dev_cu[idx].y = -D_u * k_s * old_u.y;

    dev_cv[idx].x = -D_v * k_s * old_v.x;
    dev_cv[idx].y = -D_v * k_s * old_v.y;
}

void fft(cufftHandle plan_r2c, cufftHandle* plan_c2r, double* dev_u, cufftDoubleComplex* dev_cu, double* dev_v, cufftDoubleComplex* dev_cv) {
    cufftExecD2Z(plan_r2c, dev_u, dev_cu);
    cufftExecD2Z(plan_r2c, dev_v, dev_cv);
}

// main loop
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

    // time steps
    int T = 100; // terminal time
    double dt = (double)T / (double)K;

    // initialize matrices
    double* u = (double*)malloc(REAL_SIZE*sizeof(double));
    double* v = (double*)malloc(REAL_SIZE*sizeof(double));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            omega = drand48();
            u[i*N + j] = (a + c)/b - 4.5*omega;
            v[i*N + j] = (a + c)*(a + c)/(c * b)/(c * b);
        }
    }

    // initialize matrices on device
    double *dev_u, *dev_v;
    cudaMalloc((void**)&dev_u, sizeof(double) * REAL_SIZE);
    cudaMalloc((void**)&dev_v, sizeof(double) * REAL_SIZE);

    // Assuming you have host arrays `u` and `v` allocated and initialized
    cudaMemcpy(dev_u, u, sizeof(double) * REAL_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v, v, sizeof(double) * REAL_SIZE, cudaMemcpyHostToDevice);

    cufftDoubleComplex *dev_cu, *dev_cv;
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
        continue;
    }

    cufftDestroy(plan_r2c); cufftDestroy(plan_c2r);
    cudaFree(dev_u); cudaFree(dev_v); cudaFree(dev_cu); cudaFree(dev_cv);
    return 0;
}