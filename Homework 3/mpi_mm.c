/* File: mpi_mm.c
 Compile with: gcc -g -Wall -o serial_mm serial_mm.c
 Run with: ./mpi_mm <Matrix dimension>
 
 Square matrix multiplication
 
 IT 388 - HW03 Ashkan Omaraie
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mpi.h>

double Get_time();
void printMatrix(long *C, int N);

/*  Main starts here */
int main(int argc, char* argv[])
{
    long localN;
    int myrank, nproc, N;

    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &myrank);

    long i,j,k;
    double start, elapsed;

    if (myrank == 0)
    {
        if (argv[1]==0)
        {
            fprintf(stderr,"\n \t USAGE: ./mpi_mm <matrix dimension> \n\n");
            exit(1);
        }
        N = atoi(argv[1]); // if read from comand line
    }

    localN = N/nproc;
    start = MPI_Wtime();
    
    //allocates memory for an array of N*N long for the matrices
    long* A = malloc(N*N*sizeof(long));
    long* B = malloc(N*N*sizeof(long));
    long* C = malloc(N*N*sizeof(long));
    long* localA = malloc(localN * N * sizeof(long));
    long* localRow = malloc(localN * sizeof(long));
    
    if (myrank == 0)
    {
        // Initialize matrices.
        for ( i = 0; i < N; ++i) 
        {
            for ( j = 0; j < N; ++j) {
                A[i*N+j] = (i%2) + j%3;
                B[i*N+j] = (i%3) - j%2;
            }
        }
    }

    // Scatter rows of matrix A to all processes
    MPI_Scatter(A, localN * N, MPI_LONG, localA, localN * N, MPI_LONG, 0, comm);
    //Broadcast the second matrix to all processes
    MPI_Bcast(B, N * N, MPI_LONG, 0, comm);

    MPI_Barrier(comm);
    
    // Compute matrix multiplication.
    // C <- C + A x B
    for (i = 0; i < N; ++i) 
    {
        for ( j = 0; j < N; ++j) 
        {
            C[i*localN+j]=0;
            for ( k = 0; k < N; ++k) 
            {
                C[i*localN+j] += localA[i*localN+k] * B[k*localN+j];
            }
        }
    }
    
    elapsed = MPI_Wtime() - start;
    printf("Matrix Dimension: C[%d,%d]\tElapsed time: %f sec\n",N,N, elapsed);
    if (N<=20)
    {
        printMatrix(C,N);
    }
        
    MPI_Finalize();
    return 0;
} /* end main */

/** Print Matrix to screen */
void printMatrix(long *C, int N)
{
    int i,j;
    for ( i = 0; i < N; ++i) {
        for ( j = 0; j < N; ++j) {
            printf("%ld ",C[i*N+j]);
        }
        printf("\n");
    }
}