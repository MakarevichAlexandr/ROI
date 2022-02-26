#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <xmmintrin.h>
#include <sys/time.h>

#define EPS 1E-6

enum {
    n = 1000003
};

void init_particles(float *x, float *y, float *z, int n)
{
    for (int i = 0; i < n; i++) {
        x[i] = cos(i + 0.1);
        y[i] = cos(i + 0.2);
        z[i] = cos(i + 0.3);
    }
}

void init_particles_d(double *x, double *y, double *z, int n)
{
    for (int i = 0; i < n; i++) {
        x[i] = cos(i + 0.1);
        y[i] = cos(i + 0.2);
        z[i] = cos(i + 0.3);
    }
}

void distance(float *x, float *y, float *z, float *d, int n)
{
    for (int i = 0; i < n; i++) {
        d[i] = sqrtf(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
    }
}

void distance_d(double *x, double *y, double *z, double *d, int n)
{
    for (int i = 0; i < n; i++) {
        d[i] = sqrtf(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
    }
}

void distance_vec_sse(float *x, float *y, float *z, float *d, int n)
{
	__m128 *xx = (__m128 *)x;
	__m128 *yy = (__m128 *)y;
	__m128 *zz = (__m128 *)z;
	__m128 *dd = (__m128 *)d;
	
	int k = n / 4;
	for (int i = 0; i < k; ++i) {
		__m128 t1 = _mm_mul_ps(xx[i], xx[i]);
		__m128 t2 = _mm_mul_ps(yy[i], yy[i]);
		__m128 t3 = _mm_mul_ps(zz[i], zz[i]);
		t1 = _mm_add_ps(t1, t2);
		t1 = _mm_add_ps(t1, t3);
		dd[i] = _mm_sqrt_ps(t1);
	}
	
	for (int i = k * 4; i < n; ++i) {
		d[i] = sqrtf(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
	}
		
}

void distance_vec_sse_d(double *x, double *y, double *z, double *d, int n)
{
	__m128d *xx = (__m128d *)x;
	__m128d *yy = (__m128d *)y;
	__m128d *zz = (__m128d *)z;
	__m128d *dd = (__m128d *)d;
	
	int k = n / 2;
	for (int i = 0; i < k; ++i) {
		__m128d t1 = _mm_mul_pd(xx[i], xx[i]);
		__m128d t2 = _mm_mul_pd(yy[i], yy[i]);
		__m128d t3 = _mm_mul_pd(zz[i], zz[i]);
		t1 = _mm_add_pd(t1, t2);
		t1 = _mm_add_pd(t1, t3);
		dd[i] = _mm_sqrt_pd(t1);
	}
	
	for (int i = k * 2; i < n; ++i) {
		d[i] = sqrtf(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
	}
		
}

void distance_vec_avx(float *x, float *y, float *z, float *d, int n)
{
    /* TODO */
    __m256 *xx = (__m256 *)x;
	__m256 *yy = (__m256 *)y;
	__m256 *zz = (__m256 *)z;
	__m256 *dd = (__m256 *)d;
	
	int k = n / 8;
	for (int i = 0; i < k; ++i) {
		__m256 t1 = _mm256_mul_ps(xx[i], xx[i]);
		__m256 t2 = _mm256_mul_ps(yy[i], yy[i]);
		__m256 t3 = _mm256_mul_ps(zz[i], zz[i]);
		t1 = _mm256_add_ps(t1, t2);
		t1 = _mm256_add_ps(t1, t3);
		dd[i] = _mm256_sqrt_ps(t1);
	}
	
	for (int i = k * 8; i < n; ++i) {
		d[i] = sqrtf(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
	}
}

void distance_vec_avx_d(double *x, double *y, double *z, double *d, int n)
{
    /* TODO */
    __m256d *xx = (__m256d *)x;
	__m256d *yy = (__m256d *)y;
	__m256d *zz = (__m256d *)z;
	__m256d *dd = (__m256d *)d;
	
	int k = n / 4;
	for (int i = 0; i < k; ++i) {
		__m256d t1 = _mm256_mul_pd(xx[i], xx[i]);
		__m256d t2 = _mm256_mul_pd(yy[i], yy[i]);
		__m256d t3 = _mm256_mul_pd(zz[i], zz[i]);
		t1 = _mm256_add_pd(t1, t2);
		t1 = _mm256_add_pd(t1, t3);
		dd[i] = _mm256_sqrt_pd(t1);
	}
	
	for (int i = k * 4; i < n; ++i) {
		d[i] = sqrtf(x[i] * x[i] + y[i] * y[i] + z[i] * z[i]);
	}
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
    float *d, *x, *y, *z;
//    double *d, *x, *y, *z;

    x = xmalloc(sizeof(*x) * n);
    y = xmalloc(sizeof(*y) * n);
    z = xmalloc(sizeof(*z) * n);
    d = xmalloc(sizeof(*d) * n);    
    
    init_particles(x, y, z, n);
//    init_particles_d(x, y, z, n);
    
    double t = wtime();
    for (int iter = 0; iter < 100; iter++) {
        distance(x, y, z, d, n);
//        distance_d(x, y, z, d, n);
    }
    t = wtime() - t;    

    printf("Elapsed time (scalar): %.6f sec.\n", t);
    free(x);
    free(y);    
    free(z);
    free(d);
    return t;
}

double run_vectorized_sse()
{
    float *d, *x, *y, *z;
//    double *d, *x, *y, *z;

	x = _mm_malloc(sizeof(*x) * n, 16);
    y = _mm_malloc(sizeof(*y) * n, 16);
    z = _mm_malloc(sizeof(*z) * n, 16);
    d = _mm_malloc(sizeof(*d) * n, 16);//sse
    
    init_particles(x, y, z, n);
//    init_particles_d(x, y, z, n);
    
    double t = wtime();
    for (int iter = 0; iter < 100; iter++) {
    	distance_vec_sse(x, y, z, d, n);
//    	distance_vec_sse_d(x, y, z, d, n);
    }
    t = wtime() - t;    

    /* Verification */
    for (int i = 0; i < n; i++) {
        float x = cos(i + 0.1);
        float y = cos(i + 0.2);
        float z = cos(i + 0.3);
        float dist = sqrtf(x * x + y * y + z * z);
        
//        double x = cos(i + 0.1);
//        double y = cos(i + 0.2);
//        double z = cos(i + 0.3);
//        double dist = sqrtf(x * x + y * y + z * z);
        if (fabs(d[i] - dist) > EPS) {
            fprintf(stderr, "Verification failed (SSE): d[%d] = %f != %f\n", i, d[i], dist);
            break;
        }
    }

    printf("Elapsed time (vectorized SSE): %.6f sec.\n", t);
    _mm_free(x);
    _mm_free(y);    
    _mm_free(z);
    _mm_free(d);
    return t;
}

double run_vectorized_avx()
{
    float *d, *x, *y, *z;
//    double *d, *x, *y, *z;

    x = _mm_malloc(sizeof(*x) * n, 32);
    y = _mm_malloc(sizeof(*y) * n, 32);
    z = _mm_malloc(sizeof(*z) * n, 32);
    d = _mm_malloc(sizeof(*d) * n, 32);//avx
    
    init_particles(x, y, z, n);
//    init_particles_d(x, y, z, n);
    
    double t = wtime();
    for (int iter = 0; iter < 100; iter++) {
        distance_vec_avx(x, y, z, d, n);
//        distance_vec_avx_d(x, y, z, d, n);
    }
    t = wtime() - t;    

    /* Verification */
    for (int i = 0; i < n; i++) {
        float x = cos(i + 0.1);
        float y = cos(i + 0.2);
        float z = cos(i + 0.3);
        float dist = sqrtf(x * x + y * y + z * z);
        
//        double x = cos(i + 0.1);
//        double y = cos(i + 0.2);
//        double z = cos(i + 0.3);
//        double dist = sqrtf(x * x + y * y + z * z);
        if (fabs(d[i] - dist) > EPS) {
            fprintf(stderr, "Verification failed (AVX): d[%d] = %f != %f\n", i, d[i], dist);
            break;
        }
    }

    printf("Elapsed time (vectorized AVX): %.6f sec.\n", t);
    _mm_free(x);
    _mm_free(y);    
    _mm_free(z);
    _mm_free(d);
    return t;
}

int main(int argc, char **argv)
{
    printf("Particles: n = %d)\n", n);
    double tscalar = run_scalar();
    //double tvec_avx = run_vectorized_avx();
    double tvec_sse = run_vectorized_sse();
    
    //printf("Speedup AVX: %.2f\n", tscalar / tvec_avx);
    printf("Speedup SSE: %.2f\n", tscalar / tvec_sse);

    return 0;
}
