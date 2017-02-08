#include <mpi.h>
#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <stdint.h>

//#define DDT_TEST

int double_equals(double a, double b, double epsilon)
{
    return abs(a - b) < epsilon;
}

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
#if defined (DDT_TEST)
    double *buffer_cuda = NULL;
    double *buffer_cuda_global = NULL;
    double *buffer_host = NULL;
#else 
    double *buffer_cuda = NULL;
    double *buffer_cuda_global = NULL;
    double *buffer_host = NULL;
#endif
    int root = 0;
    MPI_Datatype root_type;
    size_t root_size;
    double t1, t2;
    cudaError_t err;
    double sum = 0.0;
    
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
    /*       
    for (k = 0; k < world_size; k++) {
        cudaSetDevice(k);
        for (j = 0; j < world_size; j++) {
            if (j != k) {
                err = cudaDeviceEnablePeerAccess(j, 0);
                if (err != cudaSuccess  && err != cudaErrorPeerAccessAlreadyEnabled) {
                    printf("peer access error\n");
           //         exit(0);
                }
            }
        }
    }*/
           
    cudaSetDevice(my_rank % 4);
    

    printf("rank %d, pid %d\n", my_rank, getpid());
    //sleep(10);

#if defined (DDT_TEST)
    create_vector(length, length, 2*length, &root_type);
    root_size = compute_buffer_length(root_type, 1);
    cudaMalloc((void **)&buffer_cuda, sizeof(double)*length*length*4);
    cudaMallocHost((void **)&buffer_host, sizeof(double)*length*length*4);
 
    MPI_Bcast(buffer_cuda, 1, root_type, root, MPI_COMM_WORLD);
    cudaMemset(buffer_cuda, 0, sizeof(double)*length*length*4);   
    if (my_rank == 0) {
        fill_vectors(buffer_host, length, length, 2*length);
        cudaMemcpy(buffer_cuda, buffer_host, root_size, cudaMemcpyHostToDevice);
    } 
#else
    cudaMalloc((void **)&buffer_cuda, sizeof(double)*length);
    cudaMalloc((void **)&buffer_cuda_global, sizeof(double)*length);
    cudaMallocHost((void **)&buffer_host, sizeof(double)*length);
//    MPI_Reduce(buffer_cuda, buffer_cuda_global, length, MPI_CHAR, MPI_SUM, root, MPI_COMM_WORLD);
    MPI_Bcast(buffer_cuda_global, length, MPI_DOUBLE, root, MPI_COMM_WORLD);
    MPI_Reduce(buffer_cuda, buffer_cuda_global, length, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
    cudaMemset(buffer_cuda_global, 0, sizeof(double)*length);
    cudaDeviceSynchronize();
    for (j = 0; j < length; j++) {
        buffer_host[j] = (double)my_rank + 0.001;
    }
    cudaMemcpy(buffer_cuda, buffer_host, sizeof(double)*length, cudaMemcpyHostToDevice);
#endif
    
    MPI_Barrier(MPI_COMM_WORLD); 
    if (my_rank == root) {
        t1 = MPI_Wtime();  
    }
#if defined (DDT_TEST)
    MPI_Bcast(buffer_cuda, 1, root_type, root, MPI_COMM_WORLD);
#else
    int r;
    /* for (r = 0; r < world_size; r++) {
         root = r;
         for (j = 0; j < length; j++) {
             buffer_host[j] = (double)my_rank;
         }
         cudaMemcpy(buffer_cuda, buffer_host, sizeof(double)*length, cudaMemcpyHostToDevice);
         //cudaMemset(buffer_cuda_global, (double)my_rank, sizeof(double)*length);
         cudaMemcpy(buffer_cuda_global, buffer_host, sizeof(double)*length, cudaMemcpyHostToDevice);
         cudaDeviceSynchronize();
         MPI_Barrier(MPI_COMM_WORLD);*/
    for (j = 0; j < 10; j++) {
       /*if (my_rank == root) {
            MPI_Reduce(MPI_IN_PLACE, buffer_cuda_global, length, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
        } else {
            MPI_Reduce(buffer_cuda, buffer_cuda, length, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
        } */
        MPI_Reduce(buffer_cuda, buffer_cuda_global, length, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
#endif
    if (my_rank == root) {
        t2 = MPI_Wtime();
        printf("root send&recv time %fs, BW %f GB/s\n", (t2-t1)/10, length*sizeof(double)/1.0E9/(t2-t1)*10);
    }
#if defined (DDT_TEST)
    if (my_rank != 0) {
        cudaMemcpy(buffer_host, buffer_cuda, root_size, cudaMemcpyDeviceToHost);
        verify_vectors(buffer_host, length, length, 2*length); 
    }
#else
    if (my_rank == root) {
        sum = 0.0;
        for (j = 0; j < world_size; j++) {
            sum += (double)j;
        }
        sum += 0.001 * (double)world_size;
        cudaMemcpy(buffer_host, buffer_cuda_global, sizeof(double)*length, cudaMemcpyDeviceToHost);
        for (j = 0; j < length; j++) {
            if (double_equals(buffer_host[j], sum, 0.00000001) == 0) {
                printf("root %d, error find , val %f, sum %f\n", root, buffer_host[j], sum);
                assert(0);
                goto cleanup;
            }
        }
        printf("root %d, no error is found\n", root);
    }
#endif
   // }
cleanup:    
    if (buffer_cuda != NULL) cudaFree(buffer_cuda);
    if (buffer_cuda_global != NULL) cudaFree(buffer_cuda_global);
    if (buffer_host != NULL) cudaFreeHost(buffer_host);
    // Finalize the MPI environment.
    MPI_Finalize();
}
