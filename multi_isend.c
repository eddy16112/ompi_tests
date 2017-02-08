#include <mpi.h>
#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>

int main(int argc, char **argv)
{
    MPI_Comm comm;  
    
    MPI_Status status;
    MPI_Request request[50];
    MPI_Status req_status[50];
    
    int length;
    int iter;
    int nb_send;
    
    int c;
    opterr = 0;
    
    int root = 0;
    int dest = 1;
    
    int i, j;
    
    int ierr;
    int tag = 0;
    char *buffer_cuda = NULL;
    double t1, t2;
    
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
    
    length = 1000000;
    iter = 10;
    nb_send = 10;
    
    while ((c = getopt (argc, argv, "s:i:m:")) != -1) {
        switch (c) {
          case 's':
            length = atoi(optarg);
            break;
          case 'i':
            iter = atoi(optarg);
            break;
          case 'm':
            nb_send = atoi(optarg);
            break;
          case '?':
            if (optopt == 's' || optopt == 'i' || optopt == 'm')
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
    
    printf("Hello world from processor %s, rank %d"
           " out of %d processors, length %d, iter %d, nb_send %d\n",
           processor_name, my_rank, world_size, length, iter, nb_send);
    
    cudaSetDevice(my_rank);
    cudaMalloc((void **)&buffer_cuda, sizeof(char)*length*nb_send);
    
    if (my_rank == root) {
        
        /* warm up */
        ierr = MPI_Send(buffer_cuda, length, MPI_CHAR, dest, tag, MPI_COMM_WORLD);
        if (ierr != MPI_SUCCESS) {
            printf("MPI_Send() returned %d", ierr);
        }
        ierr = MPI_Recv(buffer_cuda, length, MPI_CHAR, dest, tag, MPI_COMM_WORLD, &status);
        if (ierr != MPI_SUCCESS) {
            printf("MPI_Recv() returned %d", ierr);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        
        t1 = MPI_Wtime();
        for (i = 0; i < iter; i++) {
            for (j = 0; j < nb_send; j++) {
                ierr = MPI_Isend(buffer_cuda + j * length, length, MPI_CHAR, dest, tag, MPI_COMM_WORLD, &request[j]);
                if (ierr != MPI_SUCCESS) {
                    printf("MPI_Isend() returned %d", ierr);
                }
            }
            MPI_Waitall(nb_send, request, req_status);
            
            for (j = 0; j < nb_send; j++) {
                ierr = MPI_Irecv(buffer_cuda + j * length, length, MPI_CHAR, dest, tag, MPI_COMM_WORLD, &request[j]);
                if (ierr != MPI_SUCCESS) {
                    printf("MPI_Isend() returned %d", ierr);
                }
            }
            MPI_Waitall(nb_send, request, req_status);
        }
        t2 = MPI_Wtime();
        printf("root each send&recv time %fs, BW %f GB/s\n", (t2-t1)*1e6/iter/nb_send, length*sizeof(char)/1.0E9/(t2-t1)*iter*nb_send*2);
        
        
    } else {
        
        /* warm up */
        ierr = MPI_Recv(buffer_cuda, length, MPI_CHAR, root, tag, MPI_COMM_WORLD, &status);
        if (ierr != MPI_SUCCESS) {
            printf("MPI_Recv() returned %d", ierr);
        }
        ierr = MPI_Send(buffer_cuda, length, MPI_CHAR, root, tag, MPI_COMM_WORLD);
        if (ierr != MPI_SUCCESS) {
            printf("MPI_Send() returned %d", ierr);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        
        for (i = 0; i < iter; i++) {
            for (j = 0; j < nb_send; j++) {
                ierr = MPI_Irecv(buffer_cuda + j * length, length, MPI_CHAR, root, tag, MPI_COMM_WORLD, &request[j]);
                if (ierr != MPI_SUCCESS) {
                    printf("MPI_Isend() returned %d", ierr);
                }
            }
            MPI_Waitall(nb_send, request, req_status);
            
            for (j = 0; j < nb_send; j++) {
                ierr = MPI_Isend(buffer_cuda + j * length, length, MPI_CHAR, root, tag, MPI_COMM_WORLD, &request[j]);
                if (ierr != MPI_SUCCESS) {
                    printf("MPI_Isend() returned %d", ierr);
                }
            }
            MPI_Waitall(nb_send, request, req_status);
        }
    }
    
    if (buffer_cuda != NULL) cudaFree(buffer_cuda);
    // Finalize the MPI environment.
    MPI_Finalize();
    return 0;
}