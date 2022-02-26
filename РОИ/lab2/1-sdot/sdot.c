#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pmmintrin.h>
#include <sys/time.h>

#include <immintrin.h>

enum { n = 1000003 };

float sdot(float *x, float *y, int n)
{
    float s = 0;
    for (int i = 0; i < n; i++)
        s += x[i] * y[i];
    return s;
}

double sdot_d(double *x, double *y, int n)
{
	double s = 0;
	for (int i = 0; i < n; ++i)
		s += x[i] * y[i];
	return s;
}

float sdot_sse(float * restrict x, float * restrict y, int n)
{
    // TODO
    __m128 *xx = (__m128 *)x;
    __m128 *yy = (__m128 *)y;
    
    int k = n / 4;
    __m128 sumv = _mm_setzero_ps();
    for (int i = 0; i < k; ++i) {
    	__m128 t1 = _mm_mul_ps(xx[i], yy[i]);
    	sumv = _mm_add_ps(sumv, t1);
	}
	
//	float t[4] __attribute__ ((aligned (16)));
//	_mm_store_ps(t, sumv);
//	float s = t[0] + t[1] + t[2] + t[3];

	sumv = _mm_hadd_ps(sumv, sumv);
	sumv = _mm_hadd_ps(sumv, sumv);
	float s __attribute__ ((aligned (16))) = 0;
	_mm_store_ss(&s, sumv);
	
	for (int i = k * 4; i < n; ++i)
		s += x[i] * y[i];
	return s;
}

double sdot_sse_d(double * restrict x, double * restrict y, int n)
{
    // TODO
    __m128d *xx = (__m128d *)x;
    __m128d *yy = (__m128d *)y;
    
    int k = n / 2;
    __m128d sumv = _mm_setzero_pd();
    for (int i = 0; i < k; ++i) {
    	__m128d t1 = _mm_mul_pd(xx[i], yy[i]);
    	sumv = _mm_add_pd(sumv, t1);
	}
	
//	double t[2] __attribute__ ((aligned (16)));
//	_mm_store_pd(t, sumv);
//	double s = t[0] + t[1];

	sumv = _mm_hadd_pd(sumv, sumv);
	double s __attribute__ ((aligned (16))) = 0;
	_mm_store_sd(&s, sumv);
	
	for (int i = k * 2; i < n; ++i)
		s += x[i] * y[i];
	return s;
}

float sdot_avx(float * restrict x, float * restrict y, int n)
{
    // TODO
    __m256 *xx = (__m256 *)x;
    __m256 *yy = (__m256 *)y;
    
    int k = n / 8;
    __m256 sumv = _mm256_setzero_ps();
    for (int i = 0; i < k; ++i) {
    	__m256 t1 = _mm256_mul_ps(xx[i], yy[i]);
    	sumv = _mm256_add_ps(sumv, t1);
	}
	
//	float t[8] __attribute__ ((aligned (32)));
//	_mm256_store_ps(t, sumv);
//	float s = t[0] + t[1] + t[2] + t[3] + t[4] + t[5] + t[6] + t[7];

	sumv = _mm256_hadd_ps(sumv, sumv);
	sumv = _mm256_hadd_ps(sumv, sumv);
	__m256 sumv_permuted = _mm256_permute2f128_ps(sumv, sumv, 1);
	sumv = _mm256_add_ps(sumv_permuted, sumv);
	
	float s __attribute__ ((aligned (32)));
	_mm256_store_ps(&s, sumv);
	
	for (int i = k * 8; i < n; ++i)
		s += x[i] * y[i];
	return s;
}

double sdot_avx_d(double * restrict x, double * restrict y, int n)
{
    // TODO
    __m256d *xx = (__m256d *)x;
    __m256d *yy = (__m256d *)y;
    
    int k = n / 4;
    __m256d sumv = _mm256_setzero_pd();
    for (int i = 0; i < k; ++i) {
    	__m256d t1 = _mm256_mul_pd(xx[i], yy[i]);
    	sumv = _mm256_add_pd(sumv, t1);
	}
	
//	double t[4] __attribute__ ((aligned (32)));
//	_mm256_store_pd(t, sumv);
//	double s = t[0] + t[1] + t[2] + t[3];

	sumv = _mm256_hadd_pd(sumv, sumv);
	__m256d sumv_permuted = _mm256_permute2f128_pd(sumv, sumv, 1);
	sumv = _mm256_add_pd(sumv_permuted, sumv);
	
	double s __attribute__ ((aligned (32)));
	_mm256_store_pd(&s, sumv);
	
	for (int i = k * 4; i < n; ++i)
		s += x[i] * y[i];
	return s;
}

void *xmalloc(size_t size)
{
    void *p = malloc(size);
    if (!p) {
        fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }
    return p;
}

double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

double run_scalar()
{
	float *x, *y;
//    double *x, *y;
    
	x = xmalloc(sizeof(*x) * n);
    y = xmalloc(sizeof(*y) * n);
    for (int i = 0; i < n; i++) {
        x[i] = 2.0;
        y[i] = 3.0;
    }        
    
    double t = wtime();
    float res = sdot(x, y, n);
//    double res = sdot_d(x, y, n);
    t = wtime() - t;    
    
    float valid_result = 2.0 * 3.0 * n;
//    double valid_result = 2.0 * 3.0 * n;
    printf("Result (scalar): %.6f err = %f\n", res, fabsf(valid_result - res));
    printf("Elapsed time (scalar): %.6f sec.\n", t);
    free(x);
    free(y);
    return t;
}

double run_vectorized_sse()
{
	float *x, *y;
//	double *x, *y;

    x = _mm_malloc(sizeof(*x) * n, 16);
    y = _mm_malloc(sizeof(*y) * n, 16);//sse
    for (int i = 0; i < n; i++) {
        x[i] = 2.0;
        y[i] = 3.0;
    }        
    
    double t = wtime();
    float res = sdot_sse(x, y, n);
//	double res = sdot_sse_d(x, y, n);
    t = wtime() - t;    
    
    float valid_result = 2.0 * 3.0 * n;
//	double valid_result = 2.0 * 3.0 * n;
    printf("Result (vectorized SSE): %.6f err = %f\n", res, fabsf(valid_result - res));
    printf("Elapsed time (vectorized SSE): %.6f sec.\n", t);
    _mm_free(x);
    _mm_free(y);
    return t;
}

double run_vectorized_avx()
{
	float *x, *y;
//	double *x, *y;

	x = _mm_malloc(sizeof(*x) * n, 32);
    y = _mm_malloc(sizeof(*y) * n, 32);//avx
    for (int i = 0; i < n; i++) {
        x[i] = 2.0;
        y[i] = 3.0;
    }        
    
    double t = wtime();
    float res = sdot_avx(x, y, n);
//    double res = sdot_avx_d(x, y, n);
    t = wtime() - t;    
    
    float valid_result = 2.0 * 3.0 * n;
//	double valid_result = 2.0 * 3.0 * n;
    printf("Result (vectorized AVX): %.6f err = %f\n", res, fabsf(valid_result - res));
    printf("Elapsed time (vectorized AVX): %.6f sec.\n", t);
    _mm_free(x);
    _mm_free(y);
    return t;
}

int main(int argc, char **argv)
{
    printf("SDOT: n = %d\n", n);
    float tscalar = run_scalar();
    float tvec_avx = run_vectorized_avx();
    float tvec_sse = run_vectorized_sse();
    
    printf("Speedup AVX: %.2f\n", tscalar / tvec_avx);
    printf("Speedup SSE: %.2f\n", tscalar / tvec_sse);
        
    return 0;
}
