#include <mpi.h>
#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>

//#define DDT_TEST
#define CPU_TEST    0

void create_vector(int count, int blocklength, int stride, MPI_Datatype *vector)
{
    int i, ierr;

    ierr = MPI_Type_vector(count, blocklength, stride,
                            MPI_DOUBLE,
                            vector);
    if (ierr != MPI_SUCCESS) { 
        printf("MPI_Type_vector() returned %d", ierr);
    }
    ierr = MPI_Type_commit (vector);
    if (ierr != MPI_SUCCESS) { 
        printf("MPI_Type_commit() returned %d", ierr);
    }
}

void fill_vectors(double* vp, int count, int blocklength, int stride)
{
    int i, j;
    for (i = 0; i < count-1; i++ ){
        for (j = i*stride; j < (i+1)*stride; j++) {
            if (j >= i*stride && j < i*stride+blocklength) {
                vp[j] = 1.0;
            } else {
                vp[j] = 0.0;
            }
        }
    }
    for (i = (count-1)*stride; i < (count-1)*stride+blocklength; i++) {
        vp[i] = 1.0;
    }
}

void verify_vectors(double *vp, int count, int blocklength, int stride)
{
    int i, j;
    int error = 0;
    for (i = 0; i < count-1; i++) {
        for (j = i*stride; j < (i+1)*stride; j++) {
            if (j >= i*stride && j < i*stride+blocklength) {
                if (vp[j] != 1.0) {
                    error ++;
                }
            } 
        }
    }
    for (i = (count-1)*stride; i < (count-1)*stride+blocklength; i++) {
        if (vp[i] != 1.0) {
            error ++;
        }
    }
    if (error != 0) {
        printf("%d error is found\n", error);
    } else {
        printf("no error is found\n");
    }
}

size_t compute_buffer_length(MPI_Datatype pdt, int count)
{
    MPI_Aint extent, lb, true_extent, true_lb;
    size_t length;

    MPI_Type_get_extent(pdt, &lb, &extent);
    MPI_Type_get_true_extent(pdt, &true_lb, &true_extent); (void)true_lb;
    length = true_lb + true_extent + (count - 1) * extent;

    return  length;
}

int main(int argc, char** argv) {
    int my_rank =0; 
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    
    printf("rank %d, pid %d\n", my_rank, getpid());
    //sleep(10);

    opterr = 0;
    int c;
    int j, k;
    uint8_t char_val;
    size_t length;
#if defined (DDT_TEST)
    double *buffer_cuda = NULL;
    double *buffer_host = NULL;
#else 
    char *buffer_cuda = NULL;
    char *buffer_host = NULL;
    char *buffer_bcast = NULL;
#endif
    int root = 0;
    MPI_Datatype root_type;
    size_t root_size;
    double t1, t2;
    cudaError_t err;
    
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
    


    CUcontext cuda_ctx[6];
    /*
    cuInit(0);
    for (j = 0; j < world_size; j++) {
        if (CUDA_SUCCESS != cuCtxCreate(&cuda_ctx[j], 0, j)) {
            assert(0);
        }
    }    */
/*
    for (k = 0; k < world_size; k++) {
        cudaSetDevice(k);
        for (j = 0; j < world_size; j++) {
            if (j != k) {
   //             err = cudaDeviceEnablePeerAccess(j, 0);
                if (err != cudaSuccess  && err != cudaErrorPeerAccessAlreadyEnabled) {
                    printf("peer access error\n");
                    exit(0);
                }
            }
        }
    }*/


    cudaSetDevice(my_rank % 4 + 0);

    int new_rank;
    int new_size;
    MPI_Comm new_comm;
    int my_new_rank;
    int node_id = my_rank / 4;
    int rank_id = my_rank % 4;
    my_new_rank = node_id + rank_id*3;
    MPI_Comm_split(MPI_COMM_WORLD, 0, my_new_rank, &new_comm);
    MPI_Comm_rank(new_comm, &new_rank);
    MPI_Comm_size(new_comm, &new_size);

    my_rank = new_rank;
//    cudaSetDevice(my_rank % 2 + 0);


#if defined (DDT_TEST)
    create_vector(length, length, 2*length, &root_type);
    root_size = compute_buffer_length(root_type, 1);
    cudaMalloc((void **)&buffer_cuda, sizeof(double)*length*length*8);
    cudaMallocHost((void **)&buffer_host, sizeof(double)*length*length*8);
 
    MPI_Bcast(buffer_cuda, 2, root_type, root, MPI_COMM_WORLD);
    cudaMemset(buffer_cuda, 0, sizeof(double)*length*length*4);   
    cudaDeviceSynchronize();
    if (my_rank == 0) {
        fill_vectors(buffer_host, length, length, 2*length);
        cudaMemcpy(buffer_cuda, buffer_host, root_size, cudaMemcpyHostToDevice);
    } 
#else
    cudaMalloc((void **)&buffer_cuda, sizeof(char)*length);
    cudaMallocHost((void **)&buffer_host, sizeof(char)*length);
    //buffer_host = malloc(sizeof(char)*length);
    if (CPU_TEST) {
        buffer_bcast = buffer_host;
    } else {
        buffer_bcast = buffer_cuda;
    }
    MPI_Bcast(buffer_bcast, length, MPI_CHAR, root, new_comm);
    MPI_Bcast(buffer_bcast, length, MPI_CHAR, root, new_comm);
    //MPI_Bcast(buffer_bcast, length, MPI_CHAR, root, MPI_COMM_WORLD);
    //MPI_Barrier(MPI_COMM_WORLD);
    cudaMemset(buffer_cuda, 0, sizeof(char)*length);
    cudaDeviceSynchronize();
    if (my_rank == root) {
        for (j = 0; j < length; j++) {
            buffer_host[j] = 97 + j%25;
        }
        cudaMemcpy(buffer_cuda, buffer_host, sizeof(char)*length, cudaMemcpyHostToDevice);
    }
#endif
    
    if (CPU_TEST) {
        buffer_bcast = buffer_host;
    } else {
        buffer_bcast = buffer_cuda;
    }
    
    MPI_Barrier(new_comm);
    if (my_rank == root) {
        t1 = MPI_Wtime();  
    }
#if defined (DDT_TEST)
    MPI_Bcast(buffer_cuda, 2, root_type, root, MPI_COMM_WORLD);
#else
    int r;
    // for (r = 0; r < world_size; r++) {
    //     root =  r;
    //     if (my_rank == root) {
    //         for (j = 0; j < length; j++) {
    //             buffer_host[j] = 97 + j%25;
    //         }
    //         cudaMemcpy(buffer_cuda, buffer_host, sizeof(char)*length, cudaMemcpyHostToDevice);
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    for (j = 0; j < 10; j++) {
        MPI_Bcast(buffer_bcast, length, MPI_CHAR, root, new_comm); 
        MPI_Barrier(new_comm);
    }

#endif
    if (my_rank == root) {
        t2 = MPI_Wtime();
        printf("root send&recv time %fs, BW %f GB/s\n", (t2-t1)/10, length*sizeof(char)/1.0E9/(t2-t1)*10);
    }
#if defined (DDT_TEST)
    if (my_rank != 0) {
        cudaMemcpy(buffer_host, buffer_cuda, root_size, cudaMemcpyDeviceToHost);
        verify_vectors(buffer_host, length, length, 2*length); 
    }
#else
    if (my_rank != root) {
        if (!CPU_TEST) {
            cudaMemcpy(buffer_host, buffer_cuda, sizeof(char)*length, cudaMemcpyDeviceToHost);
        }
        for (j = 0; j < length; j++) {
            if (buffer_host[j] != (97 + j%25)) {
                printf("error find , val %c\n", buffer_host[j]);
                assert(0);
                goto cleanup;
            }
        }
        printf("no error is found\n");
    }
#endif
    // }
cleanup:
    if (buffer_cuda != NULL) cudaFree(buffer_cuda);
   // free(buffer_cuda);
   // if (buffer_host != NULL) cudaFreeHost(buffer_host);
    // Finalize the MPI environment.
    MPI_Finalize();
}
