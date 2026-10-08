#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>

#define RUNS 5
#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    printf("CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void addKernel(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int verify(const float *c, const float *ref, int n) {
    for (int i = 0; i < n; i++)
        if (fabsf(c[i] - ref[i]) > 1e-5f) return 0;
    return 1;
}

int main() {
    int sizes[] = {1000000, 10000000, 50000000};
    int threads[] = {1, 2, 4};
    CHECK(cudaFree(0));

    printf("N,method,time_ms,speedup_vs_serial,correct\n");
    for (int s = 0; s < 3; s++) {
        int n = sizes[s];
        size_t bytes = (size_t)n * sizeof(float);
        float *a = (float *)malloc(bytes), *b = (float *)malloc(bytes);
        float *c = (float *)malloc(bytes), *ref = (float *)malloc(bytes);
        for (int i = 0; i < n; i++) { a[i] = i * 0.5f; b[i] = i * 0.25f; c[i] = 0; ref[i] = 0; }

        // warm-up
        for (int i = 0; i < n; i++) ref[i] = a[i] + b[i];

        // serial
        double serial = 1e9;
        for (int r = 0; r < RUNS; r++) {
            double t0 = omp_get_wtime();
            for (int i = 0; i < n; i++) ref[i] = a[i] + b[i];
            double ms = (omp_get_wtime() - t0) * 1000.0;
            if (ms < serial) serial = ms;
        }
        printf("%d,serial,%.3f,1.00,YES\n", n, serial);

        // OpenMP
        for (int t = 0; t < 3; t++) {
            omp_set_num_threads(threads[t]);
            double best = 1e9;
            for (int r = 0; r < RUNS; r++) {
                double t0 = omp_get_wtime();
                #pragma omp parallel for
                for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
                double ms = (omp_get_wtime() - t0) * 1000.0;
                if (ms < best) best = ms;
            }
            printf("%d,openmp_%d_threads,%.3f,%.2f,%s\n", n, threads[t], best,
                   serial / best, verify(c, ref, n) ? "YES" : "NO");
        }

        // CUDA
        float *da, *db, *dc;
        CHECK(cudaMalloc(&da, bytes));
        CHECK(cudaMalloc(&db, bytes));
        CHECK(cudaMalloc(&dc, bytes));
        int block = 256, grid = (n + block - 1) / block;
        cudaEvent_t st, en;
        cudaEventCreate(&st); cudaEventCreate(&en);
        float ms;

        // warm-up
        CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));
        addKernel<<<grid, block>>>(da, db, dc, n);
        CHECK(cudaDeviceSynchronize());

        // kernel only
        float kBest = 1e9;
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(st);
            addKernel<<<grid, block>>>(da, db, dc, n);
            cudaEventRecord(en);
            cudaEventSynchronize(en);
            cudaEventElapsedTime(&ms, st, en);
            if (ms < kBest) kBest = ms;
        }

        // total = copy in + kernel + copy out
        float tBest = 1e9;
        memset(c, 0, bytes);
        for (int r = 0; r < RUNS; r++) {
            cudaEventRecord(st);
            CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
            CHECK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));
            addKernel<<<grid, block>>>(da, db, dc, n);
            CHECK(cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost));
            cudaEventRecord(en);
            cudaEventSynchronize(en);
            cudaEventElapsedTime(&ms, st, en);
            if (ms < tBest) tBest = ms;
        }
        int ok = verify(c, ref, n);
        printf("%d,cuda_kernel_only,%.3f,%.2f,%s\n", n, kBest, serial / kBest, ok ? "YES" : "NO");
        printf("%d,cuda_total_with_copy,%.3f,%.2f,%s\n", n, tBest, serial / tBest, ok ? "YES" : "NO");

        cudaFree(da); cudaFree(db); cudaFree(dc);
        free(a); free(b); free(c); free(ref);
    }
    return 0;
}
