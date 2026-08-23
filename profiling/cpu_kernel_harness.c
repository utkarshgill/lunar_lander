#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "kernel_artifacts/tiny_cpu_10000x128x128.c"

enum { M = 10000, K = 128, N = 128, WARMUP = 10, SAMPLES = 50 };

static int compare_double(const void *left, const void *right) {
  double a = *(const double *)left;
  double b = *(const double *)right;
  return (a > b) - (a < b);
}

static double now_seconds(void) {
  struct timespec value;
  clock_gettime(CLOCK_MONOTONIC_RAW, &value);
  return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

static void fill(float *data, size_t count, uint32_t seed) {
  uint32_t state = seed;
  for (size_t index = 0; index < count; index++) {
    state = state * 1664525u + 1013904223u;
    data[index] = ((float)(state >> 8) / 16777216.0f) * 2.0f - 1.0f;
  }
}

static void run_cblas(float *output, const float *left, const float *right) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
              1.0f, left, K, right, N, 0.0f, output, N);
}

static void run_cblas_column(float *output, const float *left, const float *right) {
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, M, K,
              1.0f, right, N, left, K, 0.0f, output, N);
}

static void run_tiny(float *output, const float *left, const float *right) {
  r_2500_32_4_4_32_4(output, (float *)left, (float *)right);
}

int main(int argc, char **argv) {
  if (argc != 2 || (strcmp(argv[1], "cblas") && strcmp(argv[1], "cblas_column") && strcmp(argv[1], "tiny"))) {
    fprintf(stderr, "usage: %s cblas|cblas_column|tiny\n", argv[0]);
    return 2;
  }
  float *left = NULL, *right = NULL, *output = NULL, *reference = NULL;
  if (posix_memalign((void **)&left, 64, M * K * sizeof(float)) ||
      posix_memalign((void **)&right, 64, K * N * sizeof(float)) ||
      posix_memalign((void **)&output, 64, M * N * sizeof(float)) ||
      posix_memalign((void **)&reference, 64, M * N * sizeof(float))) return 3;
  fill(left, M * K, 1);
  fill(right, K * N, 2);
  run_cblas(reference, left, right);
  void (*operation)(float *, const float *, const float *) = run_tiny;
  if (!strcmp(argv[1], "cblas")) operation = run_cblas;
  if (!strcmp(argv[1], "cblas_column")) operation = run_cblas_column;
  operation(output, left, right);

  double max_abs = 0.0, sum_square = 0.0;
  for (size_t index = 0; index < M * N; index++) {
    double difference = (double)output[index] - reference[index];
    if (fabs(difference) > max_abs) max_abs = fabs(difference);
    sum_square += difference * difference;
  }
  for (int index = 0; index < WARMUP; index++) operation(output, left, right);
  double times[SAMPLES];
  for (int index = 0; index < SAMPLES; index++) {
    double start = now_seconds();
    operation(output, left, right);
    times[index] = (now_seconds() - start) * 1000.0;
  }
  qsort(times, SAMPLES, sizeof(double), compare_double);
  double median_ms = (times[SAMPLES / 2 - 1] + times[SAMPLES / 2]) * 0.5;
  double operations = 2.0 * (double)M * (double)N * (double)K;
  printf("{\"mode\":\"%s\",\"median_ms\":%.9f,\"p10_ms\":%.9f,"
         "\"p90_ms\":%.9f,\"gflops\":%.6f,\"max_abs\":%.9g,\"rms\":%.9g}\n",
         argv[1], median_ms, times[SAMPLES / 10], times[SAMPLES * 9 / 10],
         operations / (median_ms * 1e6), max_abs, sqrt(sum_square / (M * N)));
  volatile float checksum = output[0];
  free(left); free(right); free(output); free(reference);
  return checksum == 123456789.0f;
}
