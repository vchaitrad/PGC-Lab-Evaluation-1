#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define RUNS 5

int main() {
    int sizes[] = {1000000, 10000000, 50000000};
    int threads[] = {1, 2, 4};

    printf("N,threads,time_ms,speedup\n");
    for (int s = 0; s < 3; s++) {
        int n = sizes[s];
        float *a = malloc(n * sizeof(float));
        float *b = malloc(n * sizeof(float));
        float *c = malloc(n * sizeof(float));
        for (int i = 0; i < n; i++) { a[i] = i * 0.5f; b[i] = i * 0.25f; c[i] = 0.0f; }

        // warm-up (not timed)
        for (int i = 0; i < n; i++) c[i] = a[i] + b[i];

        // serial: best of RUNS
        double serial = 1e9;
        for (int r = 0; r < RUNS; r++) {
            double t0 = omp_get_wtime();
            for (int i = 0; i < n; i++) c[i] = a[i] + b[i];
            double ms = (omp_get_wtime() - t0) * 1000.0;
            if (ms < serial) serial = ms;
        }
        printf("%d,serial,%.3f,1.00\n", n, serial);

        // OpenMP: best of RUNS
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
            printf("%d,%d,%.3f,%.2f\n", n, threads[t], best, serial / best);
        }
        free(a); free(b); free(c);
    }
    return 0;
}
