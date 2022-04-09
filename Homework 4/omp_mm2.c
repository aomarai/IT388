/* File: omp_mv.c
 *
 * Purpose:
 *     Computes a parallel matrix-vector product with openMP
 * Linux Servers:
 *  Compile:  gcc -g -Wall -fopenmp -o omp_mv omp_mv.c
 *  Run: ./omp_mv <thread_count> <matrix dimension m=n>
 * Expanse Cluster:
 *  1) load intel compiler 
        module load intel mvapich2
    2) compile code with 
        icc -o mv omp_mv.c -qopenmp
    3) submit job script:
        sbatch ompScript.sb
 *
 * IT 388 - Illinois State University
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

/* Parallel function */
/*------------------------------------------------------------------*/
int main(int argc, char* argv[]) {
    double start, finish, elapsed;
    int i, j, k, l, nThreads;
    int     m, n;
    double* A;
    double* x;
    double* y;
    
    /* Get number of threads from command line */
    nThreads = atoi(argv[1]);
    omp_set_num_threads(nThreads);
    m=n=atoi(argv[2]); // get matrix dimension
    
    A = malloc(m*n*sizeof(double));
    x = malloc(m*n*sizeof(double));
    y = malloc(m*sizeof(double));
    
        start = omp_get_wtime();
        // generate matrix
        for (i=0;i<m;i++){
            #pragma omp parallel for num_threads(nThreads)
            for (j=0;j<n;j++){
                A[i*n + j] = (i+j)%6;
            }
        }
    
        // Generate another matrix
         for (k=0;k<m;k++){
            #pragma omp parallel for num_threads(nThreads)
            for (l=0;l<n;l++){
                x[k*n + l] = (k+l)%6;
            }
        }
        
    
        // Matrix-vector multiplication

        for (i = 0; i <= m; i++) {
            y[i] = 0.0;
            #pragma omp parallel for num_threads(nThreads)
            for (j = 0; j < n; j++)
                y[i] += A[i*n+j] * x[k*n+l];
        }
    
    finish = omp_get_wtime();
    elapsed = finish - start;
    
    
    printf("A[%d,%d] x[%d], #threads: %d Elapsed time: %f\n",m,n,n,nThreads,elapsed);
    
    free(A);
    free(x);
    free(y);
    return 0;
}  /* main */



