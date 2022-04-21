// Ashkan Omaraie IT388 Homework 5
#include <stdio.h>
#include <stdlib.h>
//#include <sys/time.h>
#include <math.h>

//Add matrix a and matrix b into matrix c
__global__ void addMatrix(int A[], int B[], int C[], int m, int n)
{
   int i= blockDim.x * blockIdx.x + threadIdx.x;

   if (blockIdx.x < m && threadIdx.x < n) 
      C[i] = A[i] + B[i];
}

//Fill the matrices with non-random numbers
void generateMatrices(int A[], int B[], int m, int n)
{
   int i, j;
   for (i = 0; i < m; i++)
   {
      for (j = 0; j < n; j++)
      {
         A[i * n + j] = i * j;
         B[i * n + j] = i * j;
      }
   }
}

void printMatrix(int A[], int m, int n)
{
   int i, j;
   for (i = 0; i < m; i++)
   {
      for (j = 0; j < n; j++)
         printf("%d ", A[i * n + j]);
      printf("\n");
   }
}

// double getTime()
// {
//     struct timeval t;
//     gettimeofday(&t, NULL);
//     return t.tv_sec + t.tv_usec / 1000000.0;
// }

/* Host code */
int main(int argc, char *argv[])
{
   int m, n;
   int *h_A, *h_B, *h_C;
   int *d_A, *d_B, *d_C;
   size_t size;
   int blockSize, numBlocks;

   //Get size of matrices
   if (argc != 3)
   {
      fprintf(stderr, "usage: %s <matrix size> <threads per block>\n", argv[0]);
      exit(0);
   }
   m = n = atoi(argv[1]);
   blockSize = atoi(argv[2]);
   numBlocks = ceil(double(n) / blockSize);
   printf("Matrix Sizes: %d Block Size: %d\n", m, blockSize);
   size = m * n * sizeof(int);

   h_A = (int *)malloc(size);
   h_B = (int *)malloc(size);
   h_C = (int *)malloc(size);

   generateMatrices(h_A, h_B, m, n);

   // Allocate matrices in device memory
   cudaMalloc(&d_A, size);
   cudaMalloc(&d_B, size);
   cudaMalloc(&d_C, size);

   //Copy matrices from host memory to device memory
   cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

   addMatrix<<<numBlocks, blockSize>>>(d_A, d_B, d_C, m, n);

   //Wait for the kernel to complete
   cudaDeviceSynchronize();

   //Copy result from device memory to host memory
   cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

   printf("Sum:\n");
   printMatrix(h_C, m, n);

   //Free device memory
   cudaFree(d_A);
   cudaFree(d_B);
   cudaFree(d_C);

   //Free host memory
   free(h_A);
   free(h_B);
   free(h_C);

   return 0;
} /* main */