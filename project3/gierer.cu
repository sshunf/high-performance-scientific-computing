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
#define DIFF_COEFF ((2.0 * M_PI / L) * (2.0 * M_PI / L))

// error handler
static void HANDLEERROR( cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file,line);
        exit(1);
    }
}
#define HandleError(err) (HANDLEERROR(err, __FILE__, __LINE__))

// advances 
__global__ void runge_kutta_advance(double* dev_u, double* dev_v, double* dev_d2u, double* dev_d2v, int step) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N || j >= N) return;

    int idx = i * N + j;

    double old_u = dev_u[idx];
    double old_v = dev_v[idx];

    dev_u[idx] = old_u + dt / step * dev_d2u[idx];
    dev_v[idx] = old_v + dt / step * dev_d2v[idx];

    __syncthreads();
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
__global__ void compute_derivative(cufftDoubleComplex* dev_au, cufftDoubleComplex* dev_av, double D_u, double D_v) {
    // TODO
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N || j >= N/2 + 1) return;

    int kx = (i <= N/2) ? i : i - N;
    int ky = j;

    double k_s = DIFF_COEFF * (kx * kx + ky * ky);

    int idx = i * (N/2 + 1) + j;

    cufftDoubleComplex old_au = dev_au[idx];
    cufftDoubleComplex old_av = dev_av[idx];

    dev_au[idx].x = -D_u * k_s * old_au.x;
    dev_au[idx].y = -D_u * k_s * old_au.y;

    dev_av[idx].x = -D_v * k_s * old_av.x;
    dev_av[idx].y = -D_v * k_s * old_av.y;

    __syncthreads(); // probably dont need this
}

// void fft(cufftHandle plan_r2c, cufftHandle* plan_c2r, double* dev_u, cufftDoubleComplex* dev_au, double* dev_v, cufftDoubleComplex* dev_av) {
//     cufftExecD2Z(plan_r2c, dev_u, dev_au);
//     cufftExecD2Z(plan_r2c, dev_v, dev_av);
// }

// main loop
int main(int argc, char* argv[]) {
    if (argc < 9) {
        printf("not enough arguments!\n");
    }

    // parse arguments
    const int N = atoi(argv[1]);
    const int D_u = atoi(argv[2]);
    const int D_v = atoi(argv[3]);
    const double a = atof(argv[4]);
    const double b = atof(argv[5]);
    const double c = atof(argv[6]);
    const double eps = atof(argv[7]);
    const int K = atoi(argv[8]);

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

    // declare 
    __constant__ int dev_N;
    __constant__ double dev_dt;
    __constant__ int dev_Du;
    __constant__ int dev_Dv;

    cudaMemcpyToSymbol(dev_N, &N, sizeof(int));
    cudaMemcpyToSymbol(dev_dt, &dt, sizeof(double));
    cudaMemcpyToSymbol(dev_Du, &D_u, sizeof(int));
    cudaMemcpyToSymbol(dev_Dv, &D_v, sizeof(int));

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
    double *dev_u, *dev_v, *dev_d2u, *dev_d2v;
    cudaMalloc((void**)&dev_u, sizeof(double) * REAL_SIZE);
    cudaMalloc((void**)&dev_v, sizeof(double) * REAL_SIZE);

    cudaMalloc((void**)&dev_d2u, sizeof(double) * REAL_SIZE);
    cudaMalloc((void**)&dev_d2v, sizeof(double) * REAL_SIZE);

    // Assuming you have host arrays `u` and `v` allocated and initialized
    cudaMemcpy(dev_u, u, sizeof(double) * REAL_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v, v, sizeof(double) * REAL_SIZE, cudaMemcpyHostToDevice);

    cufftDoubleComplex *dev_au, *dev_av;
    cudaMalloc((void**)&dev_au, sizeof(cufftDoubleComplex) * COMPLEX_SIZE);
    cudaMalloc((void**)&dev_av, sizeof(cufftDoubleComplex) * COMPLEX_SIZE);

    // create fft plans
    cufftHandle plan_r2c, plan_c2r;
    cufftPlan2d(&plan_r2c, N, N, CUFFT_R2C);
    cufftPlan2d(&plan_c2r, N, N, CUFFT_C2R);
    
    // cufft does not normalize so make sure to divide by N*N after the inverse

    dim3 blockDim(16, 16);
    dim3 gridDim(((N/2 + 1) + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    // time step loop
    for (int t = 0; t < K; t++) {
        // forward fft (t1)
        cufftExecD2Z(plan_r2c, dev_u, dev_au);
        cufftExecD2Z(plan_r2c, dev_v, dev_av);
    
        // compute derivative
        compute_derivative<<<blockDim, gridDim>>>(dev_au, dev_av, N, D_u, D_v);
        
        // backward fft (t1)
        cufftExecZ2D(plan_c2r, dev_d2u, dev_au);
        cufftExecZ2D(plan_c2r, dev_d2v, dev_av);

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
    cudaFree(dev_u); cudaFree(dev_v); cudaFree(dev_au); cudaFree(dev_av);
    return 0;
}