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
    char *buffer_cuda = NULL;
    char *buffer_host = NULL;
    char *buffer_tmp = NULL;
    int root = 0;
    MPI_Datatype root_type;
    int tag = 0;
    size_t root_size;
    double t1, t2;
    cudaError_t err;
    MPI_Status status;
    int ierr;
    int nb_segs;
    int seg_size = 2000000;
    MPI_Request req[1000];
    MPI_Status sta_array[1000];
    
    while ((c = getopt (argc, argv, "l:s:")) != -1) {
        switch (c) {
            case 'l':
                length = atoi(optarg);
                break;
            case 's':
                seg_size = atoi(optarg);
                break;
            case '?':
                if (optopt == 'l')
                    fprintf (stderr, "Option -%c requires an argument.\n", optopt);
                else if (optopt == 's')
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
           " out of %d processors, size %ld, seg_size %ld\n",
           processor_name, my_rank, world_size, length, seg_size);
           /*
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
    }  */
    
    cudaSetDevice(my_rank);      
    cudaMalloc((void **)&buffer_cuda, sizeof(char)*length);
    cudaMallocHost((void **)&buffer_host, sizeof(char)*length);
    buffer_tmp = buffer_cuda;
    
    nb_segs = length / seg_size;
    
    MPI_Bcast(buffer_cuda, length, MPI_CHAR, root, MPI_COMM_WORLD);
    
    cudaMemset(buffer_cuda, 0, sizeof(char)*length);
    cudaDeviceSynchronize();
    if (my_rank == 0) {
        for (j = 0; j < length; j++) {
            buffer_host[j] = 'a';
        }
        cudaMemcpy(buffer_cuda, buffer_host, sizeof(char)*length, cudaMemcpyHostToDevice);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    

    if (my_rank == 0) {
        t1 = MPI_Wtime();  
    }
    
    printf("nb_segs %d\n", nb_segs);
    
    
    
    if (my_rank == 0) {
        for (j = 0; j < nb_segs; j++) {
            tag = j;
            printf("send_buffer %p, size %d\n", buffer_tmp, seg_size);
            ierr = MPI_Isend(buffer_tmp, seg_size, MPI_CHAR, my_rank+1, tag, MPI_COMM_WORLD, &req[j]);
            if (ierr != MPI_SUCCESS) {
                printf("MPI_Send() returned %d", ierr);
            }
            buffer_tmp += seg_size;
        }
        MPI_Waitall(nb_segs, req, sta_array);
    } else if (my_rank == (world_size - 1)){
        for (j = 0; j < nb_segs; j++) {
            tag = j;
            ierr = MPI_Irecv(buffer_tmp, seg_size, MPI_CHAR, my_rank-1, tag, MPI_COMM_WORLD, &req[j]);
            if (ierr != MPI_SUCCESS) {
                printf("MPI_Send() returned %d", ierr);
            }
            buffer_tmp += seg_size;
        }
        MPI_Waitall(nb_segs, req, sta_array);
    } else {
        tag = 0;
        ierr = MPI_Recv(buffer_tmp, seg_size, MPI_CHAR, my_rank-1, tag, MPI_COMM_WORLD, &status);
        if (ierr != MPI_SUCCESS) {
            printf("MPI_Send() returned %d", ierr);
        }
        tag = 0;
        ierr = MPI_Isend(buffer_tmp, seg_size, MPI_CHAR, my_rank+1, tag, MPI_COMM_WORLD, &req[0]);
        if (ierr != MPI_SUCCESS) {
            printf("MPI_Send() returned %d", ierr);
        }
        buffer_tmp += seg_size;
        for (j = 1; j < nb_segs; j++) {
            tag = j;
            ierr = MPI_Irecv(buffer_tmp, seg_size, MPI_CHAR, my_rank-1, tag, MPI_COMM_WORLD, &req[1]);
            if (ierr != MPI_SUCCESS) {
                printf("MPI_Send() returned %d", ierr);
            }
            MPI_Waitall(2, req, sta_array);
            ierr = MPI_Isend(buffer_tmp, seg_size, MPI_CHAR, my_rank+1, tag, MPI_COMM_WORLD, &req[0]);
            if (ierr != MPI_SUCCESS) {
                printf("MPI_Send() returned %d", ierr);
            }
            buffer_tmp += seg_size;
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == 0) {
        t2 = MPI_Wtime();
        printf("rank %d send&recv time %fs, BW %f GB/s\n", my_rank, (t2-t1), length*sizeof(char)/1.0E9/(t2-t1));
    }
    
    if (my_rank != 0) {
        cudaMemcpy(buffer_host, buffer_cuda, sizeof(char)*length, cudaMemcpyDeviceToHost);
        for (j = 0; j < length; j++) {
            if (buffer_host[j] != 'a') {
                printf("error find , val %c\n", buffer_host[j]);
                goto cleanup;
            }
        }
        printf("no error is found\n");
    }
    
    
cleanup:
    if (buffer_cuda != NULL) cudaFree(buffer_cuda);
    if (buffer_host != NULL) cudaFreeHost(buffer_host);
    MPI_Finalize();
}