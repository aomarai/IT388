
/* File: serial_mm.c
 Compile with: gcc -g -Wall -o serial_mm serial_mm.c
 Run with: ./serial_mm <Matrix dimension>
 
 Square matrix multiplication
 
 IT 388 - HW03
 */
#include<stdio.h>
#include<stdlib.h>
#include<sys/time.h>

double Get_time();
void printMatrix(long *C,int N);

/*  Main starts here */
int main(int argc,char* argv[])
{

    long i,j,k;
    double start, elapsed;
    if (argv[1]==0){
        fprintf(stderr,"\n \t USAGE: ./mpi_mm <matrix dimension> \n\n");
        exit(1);
    }
    int N = atoi(argv[1]); // if read from comand line
    
    //allocates memory for an array of N*N long
    long* A = malloc(N*N*sizeof(long));
    long* B = malloc(N*N*sizeof(long));
    long* C = malloc(N*N*sizeof(long));
    
    
    // Initialize matrices.
    for ( i = 0; i < N; ++i) {
        for ( j = 0; j < N; ++j) {
            A[i*N+j] = (i%2) + j%3;
            B[i*N+j] = (i%3) - j%2;
        }
    }
    
    // Compute matrix multiplication.
    // C <- C + A x B
    start = Get_time();
    
    for ( i = 0; i < N; ++i) {
        for ( j = 0; j < N; ++j) {
            C[i*N+j]=0;
            for ( k = 0; k < N; ++k) {
                C[i*N+j] += A[i*N+k] * B[k*N+j];
            }
        }
    }
    
    elapsed = Get_time()-start;
    printf("Matrix Dimension: C[%d,%d]\tElapsed time: %f sec\n",N,N, elapsed);
    if (N<=20)
        printMatrix(C,N);
    
    return 0;
} /* end main */


/** Returns the wall clock time  */
double Get_time(){
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec/1000000.0;
}

/** Print Matrix to screen */
void printMatrix(long *C,int N){
    int i,j;
    for ( i = 0; i < N; ++i) {
        for ( j = 0; j < N; ++j) {
            printf("%ld ",C[i*N+j]);
        }
        printf("\n");
    }
}

