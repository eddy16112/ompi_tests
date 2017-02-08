#include "mpi.h"

#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

int main(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    
    opterr = 0;
    int c;
    int j, k;
    size_t length;
    double *buffer_cuda = NULL;
    double *buffer_host = NULL;
    int root = 0;
    MPI_Datatype root_type;
    int tag = 0;
    size_t root_size;
    double t1, t2;
    cudaError_t err;
    MPI_Status status;
    int ierr;
    
    while ((c = getopt (argc, argv, "l:")) != -1) {
        switch (c) {
            case 'l':
                length = atoi(optarg);
                break;
            case '?':
                if (optopt == 'l')
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

    // Print off a hello world message
    printf("Hello world from processor %s, rank %d"
           " out of %d processors, size %ld\n",
           processor_name, my_rank, world_size, length);
    
    for (k = 0; k < world_size; k++) {
        cudaSetDevice(k);
        for (j = 0; j < world_size; j++) {
            if (j != k) {
                err = cudaDeviceEnablePeerAccess(j, 0);
                if (err != cudaSuccess  && err != cudaErrorPeerAccessAlreadyEnabled) {
                    printf("peer access error\n");
                    exit(0);
                }
            }
        }
    }  
    
    cudaSetDevice(my_rank);      
    cudaMalloc((void **)&buffer_cuda, sizeof(char)*length);
    
    if (my_rank % 2 == 0) {
        ierr = MPI_Send(buffer_cuda, length, MPI_CHAR, my_rank+1, tag, MPI_COMM_WORLD);
        if (ierr != MPI_SUCCESS) {
            printf("MPI_Send() returned %d", ierr);
        }
        t1 = MPI_Wtime();  
        for (j = 0; j < 10; j++) {
            ierr = MPI_Send(buffer_cuda, length, MPI_CHAR, my_rank+1, tag, MPI_COMM_WORLD);
            if (ierr != MPI_SUCCESS) {
                printf("MPI_Send() returned %d", ierr);
            }
            ierr = MPI_Recv(buffer_cuda, length, MPI_CHAR, my_rank+1, tag, MPI_COMM_WORLD, &status);
            if (ierr != MPI_SUCCESS) {
                printf("MPI_Send() returned %d", ierr);
            }
        }
        t2 = MPI_Wtime();
        printf("rank %d send&recv time %fs, BW %f GB/s\n", my_rank, (t2-t1)/2, length*sizeof(char)/1.0E9/(t2-t1)*10*2);
    } else {
        ierr = MPI_Recv(buffer_cuda, length, MPI_CHAR, my_rank-1, tag, MPI_COMM_WORLD, &status);
        if (ierr != MPI_SUCCESS) {
            printf("MPI_Send() returned %d", ierr);
        }
        
        for (j = 0; j < 10; j++) {
            ierr = MPI_Recv(buffer_cuda, length, MPI_CHAR, my_rank-1, tag, MPI_COMM_WORLD, &status);
            if (ierr != MPI_SUCCESS) {
                printf("MPI_Send() returned %d", ierr);
            }
            ierr = MPI_Send(buffer_cuda, length, MPI_CHAR, my_rank-1, tag, MPI_COMM_WORLD);
            if (ierr != MPI_SUCCESS) {
                printf("MPI_Send() returned %d", ierr);
            }
        }
    }
    if (buffer_cuda != NULL) cudaFree(buffer_cuda);
    MPI_Finalize();
}
