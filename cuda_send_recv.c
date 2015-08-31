#include "mpi.h"

#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <unistd.h>
#include <cuda_runtime.h>

#define CUDA_TEST
//#define MPI_ASYNC

void parse_argv(int argc, char **argv, int *length, int *iter)
{
    opterr = 0;
    int c;
    
    while ((c = getopt (argc, argv, "s:i:")) != -1) {
        switch (c) {
          case 's':
            *length = atoi(optarg);
            break;
          case 'i':
            *iter = atoi(optarg);
            break;
          case '?':
            if (optopt == 's' || optopt == 'i')
              fprintf (stderr, "Option -%c requires an argument.\n", optopt);
            else if (isprint (optopt))
              fprintf (stderr, "Unknown option `-%c'.\n", optopt);
            else
              fprintf (stderr,"Unknown option character `\\x%x'.\n", optopt);
            exit(1);
          default:
            exit(1);
          }
      }
}

int main(int argc, char **argv)
{
    MPI_Comm comm;  
    
    MPI_Status status;
#if defined (MPI_ASYNC) 
    MPI_Request request;
#endif
    
    int ierr, tag = 0;
    
    int rank, size;
    
    int length, iterations, i;
    
    double *buffer_pingpong, *buffer_cuda, *buffer_host;
    
    int root, dest;
    
    double t1, t2;

    length = 0;
    iterations = 0;
    
    parse_argv(argc, argv, &length, &iterations);

    if (length == 0) {
        length = 4000;
    }
    if (iterations == 0) {
        iterations = 1;
    }
    
    ierr = MPI_Init(&argc, &argv);
    if (ierr != MPI_SUCCESS) {
        printf("MPI_Init() returned %d", ierr);
        return 1;
    }
    
    /* init root, dest */
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);
    root = 0;
    if (size == 1) {
        dest = 0;
    } else if (size == 2) {
        dest = 1;
    } else {
        printf("not support yet\n");
        return 1;
    }

#if defined (CUDA_TEST)        
    cudaMalloc((void **)&buffer_cuda, sizeof(double)*length);
    cudaMemset(buffer_cuda, 0, sizeof(double)*length);
    buffer_pingpong = buffer_cuda;
#else
    cudaMallocHost((void **)&buffer_host, sizeof(double)*length);
    memset(buffer_host, 0, sizeof(double)*length);
    buffer_pingpong = buffer_host;
#endif
    
    sleep(15);
    
    printf("I am rank %d, addr %p, total_size %lu\n", rank, buffer_pingpong, sizeof(double)*length);
    
    if (rank == root) {
        /* send */
        t1 = MPI_Wtime();
        for (i = 0; i < iterations; i++) {        
#if defined (MPI_ASYNC)
            ierr = MPI_Isend(buffer_pingpong, length, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &request);
#else
            ierr = MPI_Send(buffer_pingpong, length, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
#endif
            if (ierr != MPI_SUCCESS) {
                printf("MPI_Send() returned %d", ierr);
            }

#if defined (MPI_ASYNC)
            ierr = MPI_Irecv(buffer_pingpong, length, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &request);
#else
            ierr = MPI_Recv(buffer_pingpong, length, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD, &status);
#endif
            if (ierr != MPI_SUCCESS) {
                printf("MPI_Recv() returned %d", ierr);
            }
#if defined (MPI_ASYNC)
            MPI_Wait(&request, &status);
#endif 
        }
        t2 = MPI_Wtime();
        printf("root send&recv time %f\n", (t2-t1)*1e6/iterations);    
        
    }
    
    if (rank == dest) {
        /* receive */
        for (i = 0; i < iterations; i++) {   
#if defined (MPI_ASYNC)
            ierr = MPI_Irecv(buffer_pingpong, length, MPI_DOUBLE, root, tag, MPI_COMM_WORLD, &request);
#else
            ierr = MPI_Recv(buffer_pingpong, length, MPI_DOUBLE, root, tag, MPI_COMM_WORLD, &status);
#endif
            if (ierr != MPI_SUCCESS) {
                printf("MPI_Recv() returned %d", ierr);
            }

#if defined (MPI_ASYNC)
            ierr = MPI_Isend(buffer_pingpong, length, MPI_DOUBLE, root, tag, MPI_COMM_WORLD, &request);
#else
            ierr = MPI_Send(buffer_pingpong, length, MPI_DOUBLE, root, tag, MPI_COMM_WORLD);
#endif

            if (ierr != MPI_SUCCESS) {
                printf("MPI_Send() returned %d", ierr);
            }
#if defined (MPI_ASYNC)
            MPI_Wait(&request, &status);
#endif
        }
    }

    //sleep(5);

#if defined (CUDA_TEST)   
    cudaFree(buffer_cuda);
#else
    cudaFreeHost(buffer_host);
#endif
    MPI_Finalize();
    return 0;
}
