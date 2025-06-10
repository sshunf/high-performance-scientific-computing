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

__constant__ int dev_N;
__constant__ double dev_dt;
__constant__ double dev_Du;
__constant__ double dev_Dv;
__constant__ double dev_a;
__constant__ double dev_b;
__constant__ double dev_c;
__constant__ double dev_eps;

// error handler
// static void HANDLEERROR( cudaError_t err, const char* file, int line) {
//     if (err != cudaSuccess) {
//         printf("%s in %s at line %d\n", cudaGetErrorString(err), file,line);
//         exit(1);
//     }
// }
// #define HandleError(err) (HANDLEERROR(err, __FILE__, __LINE__))

// advances time step 
__global__ void runge_kutta_step(double* dev_u_new, double* dev_v_new, double* dev_u, double* dev_v, double* dev_d2u, double* dev_d2v, int step) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= dev_N || j >= dev_N) return;

    int idx = i * dev_N + j;

    dev_u_new[idx] = dev_u[idx] + dev_dt / step * dev_d2u[idx];
    dev_v_new[idx] = dev_v[idx] + dev_dt / step * dev_d2v[idx];
}

// helper function to rescale matrix after inverse fft
__global__ void rescale(double* dev_matrix, double factor) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;  
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dev_N || j >= dev_N) return;

    int idx = i * dev_N + j;
    dev_matrix[idx] *= factor;
}

__global__ void add_reaction_terms(double* dev_u, double* dev_v, double* dev_d2u, double* dev_d2v) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dev_N || j >= dev_N) return;

    int idx = i * dev_N + j;

    double u0 = dev_u[idx];
    double v0 = dev_v[idx];

    // Compute the reaction‐terms
    double reactionU = dev_a + (u0*u0 / (v0 * (1.0 + dev_eps * u0*u0))) - dev_b * u0; // a + u^2/(v(1+εu^2)) – b u
    double reactionV = u0*u0 - dev_c * v0; // u^2 – c v

    dev_d2u[idx] += reactionU;
    dev_d2v[idx] += reactionV; 
}

// function to swap pointers for runge-kutta step
inline void swap_pointers(double*& A, double*& B) {
    double* temp = A;
    A = B;
    B = temp;
}

// consider computing only u instead of both u and v
__global__ void compute_derivative(cufftDoubleComplex* dev_au, cufftDoubleComplex* dev_av) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= dev_N || j >= dev_N/2 + 1) return;

    int kx = (i <= dev_N/2) ? i : i - dev_N;
    int ky = j;

    double k2 = double(kx*kx + ky*ky);
    double fac = -k2 / double(dev_N * dev_N);

    int idx = i * (dev_N/2 + 1) + j;

    dev_au[idx].x *= fac;
    dev_au[idx].y *= fac;
    dev_av[idx].x *= fac;
    dev_av[idx].y *= fac;
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
    const double D_u = atof(argv[2]);
    const double D_v = atof(argv[3]);
    const double a = atof(argv[4]);
    const double b = atof(argv[5]);
    const double c = atof(argv[6]);
    const double eps = atof(argv[7]);
    const int K = atoi(argv[8]);

    // parse seed
    long seed;
    if (argc >= 10) {
        seed = atol(argv[9]);
    } else {
        seed = 42;
    }

    printf("N: %d\n", N);
    printf("D_u: %.2f\n", D_u);
    printf("D_v: %.2f\n", D_v);
    printf("a: %.2f\n", a);
    printf("b: %.2f\n", b);
    printf("c: %.2f\n", c);
    printf("eps: %.3f\n", eps);
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

    cufftDoubleComplex* au = (cufftDoubleComplex*)malloc(COMPLEX_SIZE*sizeof(cufftDoubleComplex));
    cufftDoubleComplex* av = (cufftDoubleComplex*)malloc(COMPLEX_SIZE*sizeof(cufftDoubleComplex));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            omega = drand48();
            u[i*N + j] = (a + c)/b - 4.5*omega;
            v[i*N + j] = (a + c)*(a + c)/(c * b)/(c * b);
        }
    }

    // copy variables to device
    cudaMemcpyToSymbol(dev_dt, &dt, sizeof(double));
    cudaMemcpyToSymbol(dev_Du, &D_u, sizeof(double));
    cudaMemcpyToSymbol(dev_Dv, &D_v, sizeof(double));
    cudaMemcpyToSymbol(dev_N, &N, sizeof(int));
    cudaMemcpyToSymbol(dev_a, &a, sizeof(double));
    cudaMemcpyToSymbol(dev_b, &b, sizeof(double));
    cudaMemcpyToSymbol(dev_c, &c, sizeof(double));
    cudaMemcpyToSymbol(dev_eps, &eps, sizeof(double));

    // initialize matrices on device
    double *dev_u, *dev_v, *dev_d2u, *dev_d2v, *dev_u_new, *dev_v_new;
    cudaMalloc((void**)&dev_u, sizeof(double) * REAL_SIZE);
    cudaMalloc((void**)&dev_v, sizeof(double) * REAL_SIZE);
    cudaMalloc((void**)&dev_d2u, sizeof(double) * REAL_SIZE);
    cudaMalloc((void**)&dev_d2v, sizeof(double) * REAL_SIZE);
    cudaMalloc((void**)&dev_u_new, sizeof(double) * REAL_SIZE);
    cudaMalloc((void**)&dev_v_new, sizeof(double) * REAL_SIZE);

    cudaMemcpy(dev_u, u, sizeof(double) * REAL_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_v, v, sizeof(double) * REAL_SIZE, cudaMemcpyHostToDevice);

    cufftDoubleComplex *dev_au, *dev_av;
    cudaMalloc((void**)&dev_au, sizeof(cufftDoubleComplex) * COMPLEX_SIZE);
    cudaMalloc((void**)&dev_av, sizeof(cufftDoubleComplex) * COMPLEX_SIZE);

    // create fft plans
    cufftHandle plan_r2c, plan_c2r;
    cufftPlan2d(&plan_r2c, N, N, CUFFT_D2Z);
    cufftPlan2d(&plan_c2r, N, N, CUFFT_Z2D);
    
    // dim3 blockDim(16, 16); // number of threads per block
    // dim3 gridDim(((N/2 + 1) + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y); // number of blocks (5, 8)

    int numBlocks = 16;
    int gridCX = ((N/2 + 1) + numBlocks - 1) / numBlocks;
    int gridX = (N + numBlocks - 1) / numBlocks;
    int gridY = (N + numBlocks - 1) / numBlocks;
    double normal_factor = 1./(N*N);

    FILE* fid = fopen("GiererU.out","w");
    fwrite(u, sizeof(double), N*N, fid);
    fclose(fid);

    fid = fopen("GiererV.out","w");
    fwrite(v, sizeof(double), N*N, fid);
    fclose(fid);

    // time step loop
    for (int t = 0; t < K; t++) {
        // forward fft (t1)
        cufftExecD2Z(plan_r2c, dev_u, dev_au);
        cufftExecD2Z(plan_r2c, dev_v, dev_av);
    
        // compute derivative
        compute_derivative<<<dim3(gridCX, gridY), dim3(numBlocks, numBlocks)>>>(dev_au, dev_av);
        cudaDeviceSynchronize();

        // write out for debug
        cudaMemcpy(au, dev_au, sizeof(cufftDoubleComplex)*COMPLEX_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(av, dev_av, sizeof(cufftDoubleComplex)*COMPLEX_SIZE, cudaMemcpyDeviceToHost);

        FILE* fid = fopen("da.out","w");
        fwrite(au, sizeof(cufftDoubleComplex), COMPLEX_SIZE, fid);
        fwrite(av, sizeof(cufftDoubleComplex), COMPLEX_SIZE, fid);
        fclose(fid);

        // backward fft (t1)
        cufftExecZ2D(plan_c2r, dev_au, dev_d2u);
        cufftExecZ2D(plan_c2r, dev_av, dev_d2v);

        rescale<<<dim3(gridX, gridY), dim3(numBlocks, numBlocks)>>>(dev_d2u, D_u * DIFF_COEFF);
        rescale<<<dim3(gridX, gridY), dim3(numBlocks, numBlocks)>>>(dev_d2v, D_v * DIFF_COEFF);
        add_reaction_terms<<<dim3(gridX, gridY), dim3(numBlocks, numBlocks)>>>(dev_u, dev_v, dev_d2u, dev_d2v);

        // runge-kutta time step (t1)
        runge_kutta_step<<<dim3(gridX, gridY), dim3(numBlocks, numBlocks)>>>(dev_u_new, dev_v_new, dev_u, dev_v, dev_d2u, dev_d2v, 4);
        cudaDeviceSynchronize();
        
        cudaMemcpy(u, dev_u_new, sizeof(double)*REAL_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(v, dev_v_new, sizeof(double)*REAL_SIZE, cudaMemcpyDeviceToHost);

        // write out for debug
        fid = fopen("stage1.out","w");
        fwrite(u, sizeof(double), N*N, fid);
        fwrite(v, sizeof(double), N*N, fid);
        fclose(fid);

        // stage 2 
        cufftExecD2Z(plan_r2c, dev_u_new, dev_au);
        cufftExecD2Z(plan_r2c, dev_v_new, dev_av);

        compute_derivative<<<dim3(gridCX, gridY), dim3(numBlocks, numBlocks)>>>(dev_au, dev_av);

        cufftExecZ2D(plan_c2r, dev_au, dev_d2u);
        cufftExecZ2D(plan_c2r, dev_av, dev_d2v);

        rescale<<<dim3(gridX, gridY), dim3(numBlocks, numBlocks)>>>(dev_d2u, D_u * DIFF_COEFF);
        rescale<<<dim3(gridX, gridY), dim3(numBlocks, numBlocks)>>>(dev_d2v, D_v * DIFF_COEFF);
        add_reaction_terms<<<dim3(gridX, gridY), dim3(numBlocks, numBlocks)>>>(dev_u_new, dev_v_new, dev_d2u, dev_d2v);

        runge_kutta_step<<<dim3(gridX, gridY), dim3(numBlocks, numBlocks)>>>(dev_u_new, dev_v_new, dev_u, dev_v, dev_d2u, dev_d2v, 3);
        cudaDeviceSynchronize();
        
        cudaMemcpy(u, dev_u_new, sizeof(double)*REAL_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(v, dev_v_new, sizeof(double)*REAL_SIZE, cudaMemcpyDeviceToHost);

        fid = fopen("stage2.out","w");
        fwrite(u, sizeof(double), N*N, fid);
        fwrite(v, sizeof(double), N*N, fid);
        fclose(fid);

        // stage 3
        cufftExecD2Z(plan_r2c, dev_u_new, dev_au);
        cufftExecD2Z(plan_r2c, dev_v_new, dev_av);

        compute_derivative<<<dim3(gridCX, gridY), dim3(numBlocks, numBlocks)>>>(dev_au, dev_av);

        cufftExecZ2D(plan_c2r, dev_au, dev_d2u);
        cufftExecZ2D(plan_c2r, dev_av, dev_d2v);

        rescale<<<dim3(gridX, gridY), dim3(numBlocks, numBlocks)>>>(dev_d2u, D_u * DIFF_COEFF);
        rescale<<<dim3(gridX, gridY), dim3(numBlocks, numBlocks)>>>(dev_d2v, D_v * DIFF_COEFF);
        add_reaction_terms<<<dim3(gridX, gridY), dim3(numBlocks, numBlocks)>>>(dev_u_new, dev_v_new, dev_d2u, dev_d2v);

        runge_kutta_step<<<dim3(gridX, gridY), dim3(numBlocks, numBlocks)>>>(dev_u_new, dev_v_new, dev_u, dev_v, dev_d2u, dev_d2v, 2);
        cudaDeviceSynchronize();
        
        cudaMemcpy(u, dev_u_new, sizeof(double)*REAL_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(v, dev_v_new, sizeof(double)*REAL_SIZE, cudaMemcpyDeviceToHost);

        fid = fopen("stage3.out","w");
        fwrite(u, sizeof(double), N*N, fid);
        fwrite(v, sizeof(double), N*N, fid);
        fclose(fid);

        // stage 4
        cufftExecD2Z(plan_r2c, dev_u_new, dev_au);
        cufftExecD2Z(plan_r2c, dev_v_new, dev_av);

        compute_derivative<<<dim3(gridCX, gridY), dim3(numBlocks, numBlocks)>>>(dev_au, dev_av);

        cufftExecZ2D(plan_c2r, dev_au, dev_d2u);
        cufftExecZ2D(plan_c2r, dev_av, dev_d2v);

        rescale<<<dim3(gridX, gridY), dim3(numBlocks, numBlocks)>>>(dev_d2u, D_u * DIFF_COEFF);
        rescale<<<dim3(gridX, gridY), dim3(numBlocks, numBlocks)>>>(dev_d2v, D_v * DIFF_COEFF);
        add_reaction_terms<<<dim3(gridX, gridY), dim3(numBlocks, numBlocks)>>>(dev_u_new, dev_v_new, dev_d2u, dev_d2v);

        runge_kutta_step<<<dim3(gridX, gridY), dim3(numBlocks, numBlocks)>>>(dev_u_new, dev_v_new, dev_u, dev_v, dev_d2u, dev_d2v, 1);
        cudaDeviceSynchronize();
        
        cudaMemcpy(u, dev_u_new, sizeof(double)*REAL_SIZE, cudaMemcpyDeviceToHost);
        cudaMemcpy(v, dev_v_new, sizeof(double)*REAL_SIZE, cudaMemcpyDeviceToHost);

        fid = fopen("stage4.out","w");
        fwrite(u, sizeof(double), N*N, fid);
        fwrite(v, sizeof(double), N*N, fid);
        fclose(fid);


        break;
        swap_pointers(dev_u, dev_u_new);
        swap_pointers(dev_v, dev_v_new);
    }

    cudaMemcpy(u, dev_u, sizeof(double)*REAL_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(v, dev_v, sizeof(double)*REAL_SIZE, cudaMemcpyDeviceToHost);

    // FILE* fid = fopen("GiererU.out","w");
    // fwrite(u, sizeof(double), N*N, fid);
    // fclose(fid);

    // fid = fopen("GiererV.out","w");
    // fwrite(v, sizeof(double), N*N, fid);
    // fclose(fid);

    cufftDestroy(plan_r2c); cufftDestroy(plan_c2r);
    cudaFree(dev_u); cudaFree(dev_v); cudaFree(dev_au); cudaFree(dev_av); cudaFree(dev_d2u); cudaFree(dev_d2v);
    return 0;
}