#include "mpi.h"

#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <unistd.h>
#include <cuda_runtime.h>

#define CUDA_TEST
//#define MPI_ASYNC


void create_vector(int count, int blocklength, int stride, MPI_Datatype *vector)
{
    int i, ierr;


    ierr = MPI_Type_vector(count, blocklength, stride,
                            MPI_DOUBLE,
                            vector);
    if (ierr != MPI_SUCCESS) { 
        printf("MPI_Type_indexed() returned %d", ierr);
    }
    ierr = MPI_Type_commit (vector);
    if (ierr != MPI_SUCCESS) { 
        printf("MPI_Type_commit() returned %d", ierr);
    }
}

static size_t compute_buffer_length(MPI_Datatype pdt, int count)
{
    MPI_Aint extent, lb, true_extent, true_lb;
    size_t length;

    MPI_Type_get_extent(pdt, &lb, &extent);
    MPI_Type_get_true_extent(pdt, &true_lb, &true_extent); (void)true_lb;
    length = true_lb + true_extent + (count - 1) * extent;

    return  length;
}

static void fill_vectors(double* vp, int count, int blocklength, int stride)
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

static void verify_vectors(double *vp, int count, int blocklength, int stride)
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
    // printf("vector received:\n");
    // for (i = 0; i < (itera-1)*gap+contig; i++) {
    //     printf("%1.f ", vp[i]);
    // }
    if (error != 0) {
        printf("%d error is found\n", error);
    } else {
        printf("no error is found\n");
    }
}

void parse_argv(int argc, char **argv, int *count, int *blocklength, int *stride, int *iter)
{
    opterr = 0;
    int c;
    
    while ((c = getopt (argc, argv, "c:b:s:i:")) != -1) {
        switch (c) {
            case 'c':
                *count = atoi(optarg);
                break;
            case 'b':
                *blocklength = atoi(optarg);
                break;
            case 's':
                *stride = atoi(optarg);
                break;
            case 'i':
                *iter = atoi(optarg);
                break;
            case '?':
                if (optopt == 'c' || optopt == 'i' || optopt == 'b' || optopt == 's')
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
    
    MPI_Datatype test_type;
    
    int ierr, tag = 0;
    
    int rank, size;
    
    int count, stride, blocklength, iterations, i;
    
    double *buffer_pingpong, *buffer_cuda, *buffer_host;
    
    int root, dest;
    
    double t1, t2;

    blocklength = 0;
    iterations = 0;
    
    parse_argv(argc, argv, &count, &blocklength, &stride, &iterations);

    if (count == 0) {
        count = 4000;
    }
    if (blocklength == 0) {
        blocklength = 256;
    }
    if (stride == 0) {
        stride = 384;
    }
    if (iterations == 0) {
        iterations = 0;
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
    sleep(15);
    
    /* upper triangular matrix */
    create_vector(count, blocklength, stride, &test_type);
    
    size_t vector_size = compute_buffer_length(test_type, 1);
    
    cudaMallocHost((void **)&buffer_host, vector_size);
#if defined (CUDA_TEST)    
    cudaMalloc((void **)&buffer_cuda, vector_size);
#endif
    
    if (rank == root) {
        /* send */
        memset(buffer_host, 0, vector_size);
        fill_vectors(buffer_host, count, blocklength, stride);
#if defined (CUDA_TEST)
        cudaMemset(buffer_cuda, 0, vector_size);
        buffer_pingpong = buffer_cuda;
        cudaMemcpy(buffer_cuda, buffer_host, vector_size, cudaMemcpyHostToDevice);
#else
        buffer_pingpong = buffer_host;
#endif
        /* warm up */
        printf("rank 0 SEND!!!!!!!!!!!!!!!!!\n");
#if defined (MPI_ASYNC) 
        ierr = MPI_Isend(buffer_pingpong, 1, test_type, dest, tag, MPI_COMM_WORLD, &request);
#else
        ierr = MPI_Send(buffer_pingpong, 1, test_type, dest, tag, MPI_COMM_WORLD);
#endif
        if (ierr != MPI_SUCCESS) {
            printf("MPI_Send() returned %d", ierr);
        }
#if defined (MPI_ASYNC)
            MPI_Wait(&request, &status);
#endif
        cudaMemset(buffer_cuda, 0, vector_size);
        printf("rank 0 RECEIVE!!!!!!!!!!!!!!!!!\n");
#if defined (MPI_ASYNC)
        ierr = MPI_Irecv(buffer_pingpong, 1, test_type, dest, tag, MPI_COMM_WORLD, &request);
#else
        ierr = MPI_Recv(buffer_pingpong, 1, test_type, dest, tag, MPI_COMM_WORLD, &status);
#endif
        if (ierr != MPI_SUCCESS) {
            printf("MPI_Recv() returned %d", ierr);
        }
#if defined (MPI_ASYNC)
        MPI_Wait(&request, &status);
#endif

        t1 = MPI_Wtime();
        for (i = 0; i < iterations; i++) {
            printf("rank 0 SEND!!!!!!!!!!!!!!!!!\n");
#if defined (MPI_ASYNC)
            ierr = MPI_Isend(buffer_pingpong, 1, test_type, dest, tag, MPI_COMM_WORLD, &request);
#else
            ierr = MPI_Send(buffer_pingpong, 1, test_type, dest, tag, MPI_COMM_WORLD);
#endif
            if (ierr != MPI_SUCCESS) {
                printf("MPI_Send() returned %d", ierr);
            }

#if defined (MPI_ASYNC)
            MPI_Wait(&request, &status);
#endif

            printf("rank 0 RECEIVE!!!!!!!!!!!!!!!!!\n");
            cudaMemset(buffer_cuda, 0, vector_size);
#if defined (MPI_ASYNC)
            ierr = MPI_Irecv(buffer_pingpong, 1, test_type, dest, tag, MPI_COMM_WORLD, &request);
#else
            ierr = MPI_Recv(buffer_pingpong, 1, test_type, dest, tag, MPI_COMM_WORLD, &status);
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
        /* verify result */
#if defined (CUDA_TEST)
        cudaMemcpy(buffer_host, buffer_cuda, vector_size, cudaMemcpyDeviceToHost);
#endif
        verify_vectors(buffer_host, count, blocklength, stride);    
        
    }
    
    if (rank == dest) {
        /* receive */
        memset(buffer_host, 0, vector_size);
#if defined (CUDA_TEST)
        cudaMemset(buffer_cuda, 0, vector_size);
        buffer_pingpong = buffer_cuda;
#else
        buffer_pingpong = buffer_host;
#endif
        for (i = 0; i < iterations+1; i++) {
            printf("rank 1 RECEIVE!!!!!!!!!!!!!!!!!\n");
#if defined (MPI_ASYNC)
            ierr = MPI_Irecv(buffer_pingpong, 1, test_type, root, tag, MPI_COMM_WORLD, &request);
#else
            ierr = MPI_Recv(buffer_pingpong, 1, test_type, root, tag, MPI_COMM_WORLD, &status);
#endif
            if (ierr != MPI_SUCCESS) {
                printf("MPI_Recv() returned %d", ierr);
            }
#if defined (MPI_ASYNC)
            MPI_Wait(&request, &status);
#endif
//            cudaMemset(buffer_cuda, 0, sizeof(double)*length*length);
            printf("rank 1 SEND!!!!!!!!!!!!!!!!!\n");
#if defined (MPI_ASYNC)
            ierr = MPI_Isend(buffer_pingpong, 1, test_type, root, tag, MPI_COMM_WORLD, &request);
#else
            ierr = MPI_Send(buffer_pingpong, 1, test_type, root, tag, MPI_COMM_WORLD);
#endif
            if (ierr != MPI_SUCCESS) {
                printf("MPI_Send() returned %d", ierr);
            }
#if defined (MPI_ASYNC)
            MPI_Wait(&request, &status);
#endif
            cudaMemset(buffer_cuda, 0, vector_size);
        }
    }
    
    /* free newly created datatype */
    ierr = MPI_Type_free(&test_type);
    if (ierr != MPI_SUCCESS) {
        printf("MPI_Type_free() returned %d", ierr);
    }
    cudaFree(buffer_cuda);
    cudaFreeHost(buffer_host);
    MPI_Finalize();
    return 0;
}
