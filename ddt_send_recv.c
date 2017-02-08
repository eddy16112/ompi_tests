#include "mpi.h"

#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <cuda_runtime.h>

#define CUDA_TEST
#define MPI_ASYNC

static int *displs, *blklens;

void upper_matrix(int mat_size, MPI_Datatype *upper)
{
    int i, ierr;

    displs = (int*)malloc( sizeof(int) * mat_size );
    blklens = (int*)malloc( sizeof(int) * mat_size );

    for( i = 0; i < mat_size; i++ ) {
        displs[i] = i * mat_size + i;
        blklens[i] = mat_size - i;
    }

    ierr = MPI_Type_indexed(mat_size, blklens, displs,
                            MPI_DOUBLE,
                            upper);
    if (ierr != MPI_SUCCESS) { 
        printf("MPI_Type_indexed() returned %d", ierr);
    }
    ierr = MPI_Type_commit (upper);
    if (ierr != MPI_SUCCESS) { 
        printf("MPI_Type_commit() returned %d", ierr);
    }
}

static void fill_upper_matrix(double *mat, int mat_size)
{
    int i, j, start, end;
    for (i = 0; i < mat_size; i++) {
        start = displs[i];
        end = start + blklens[i];
        for (j = start; j < end; j++) {
            mat[j] = 0.0+i;
        }
    }

    // printf("matrix generate\n");
    // for (i = 0; i < mat_size; i++) {
    //     for (j = 0; j < mat_size; j++) {
    //         printf(" %1.f ", mat[i*mat_size+j]);
    //     }
    //     printf("\n");
    // }
}

static void verify_mat_result(double *mat, int mat_size)
{
    int i, j, error = 0;
    int start, end;

    for (i = 0; i < mat_size; i++) {
        start = displs[i];
        end = start + blklens[i];
        for (j = start; j < end; j++) {
            if (mat[j] != (0.0+i)) {
                error ++;
            }
        }
    }

    // printf("matrix received\n");
    // for (i = 0; i < msize; i++) {
    //     for (j = 0; j < msize; j++) {
    //         printf(" %1.f ", mat[i*msize+j]);
    //     }
    //     printf("\n");
    // }

    if (error != 0) {
        printf("error is found %d\n", error);
        assert(0);
    } else {
        printf("no error is found\n");
    }
    
  //  free( displs );
  //  free( blklens );
}

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
    MPI_Request request[10];
    MPI_Status req_status[10];
#endif
    
    MPI_Datatype test_type;
    
    int ierr, tag = 0;
    
    int rank, size;
    
    int length, iterations, i, j;
    
    double *buffer_pingpong[10], *buffer_cuda[10], *buffer_host;

    char *buffer_cuda_contig;
    
    int root, dest;
    
    double t1, t2;
    
    int nb_send;

    length = 0;
    iterations = 0;
    nb_send = 6;
    
    parse_argv(argc, argv, &length, &iterations);

    if (length == 0) {
        length = 4000;
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
    printf("rank %d, pid %d\n", rank, getpid());
    sleep(10);
    if (rank == 0) {
        cudaSetDevice(0);
    } else {
        cudaSetDevice(1);
    }
    
    /* upper triangular matrix */
    upper_matrix(length, &test_type);
    
    cudaMallocHost((void **)&buffer_host, sizeof(double)*length*length);
#if defined (CUDA_TEST)
    cudaMalloc((void **)&buffer_cuda_contig, sizeof(double)*length*length*nb_send);
    for (i = 0; i < nb_send; i++) { 
          buffer_cuda[i] = buffer_cuda_contig + i * sizeof(double)*length*length;  
          printf("buffer i %d, %p\n", i, buffer_cuda[i]); 
    //    cudaMalloc((void **)&buffer_cuda[i], sizeof(double)*length*length);
    }
#endif
    
    if (rank == root) {
        /* send */
        memset(buffer_host, 0, sizeof(double)*length*length);
        fill_upper_matrix(buffer_host, length);
#if defined (CUDA_TEST)
        for (i = 0; i < nb_send; i++) {   
            cudaMemset(buffer_cuda[i], 0, sizeof(double)*length*length);
            buffer_pingpong[i] = buffer_cuda[i];
            cudaMemcpy(buffer_cuda[i], buffer_host, sizeof(double)*length*length, cudaMemcpyHostToDevice);
        }
#else
        buffer_pingpong[0] = buffer_host;
#endif
        /* warm up */
        printf("WARM UP rank 0 SEND!!!!!!!!!!!!!!!!!\n");
#if defined (MPI_ASYNC) 
        for (i = 0; i < nb_send; i++) {  
            printf("send buffer %p\n", buffer_pingpong[i]); 
            ierr = MPI_Isend(buffer_pingpong[i], 1, test_type, dest, i, MPI_COMM_WORLD, &request[i]);
        }
#else
        ierr = MPI_Send(buffer_pingpong[0], 1, test_type, dest, tag, MPI_COMM_WORLD);
//        ierr = MPI_Send(buffer_pingpong, 1, test_type, dest, tag, MPI_COMM_WORLD);
#endif
        if (ierr != MPI_SUCCESS) {
            printf("MPI_Send() returned %d", ierr);
        }
#if defined (MPI_ASYNC)
        MPI_Waitall(nb_send, request, req_status);
	    MPI_Barrier(MPI_COMM_WORLD);
#endif
        for (i = 0; i < nb_send; i++) {   
            cudaMemset(buffer_cuda, 0, sizeof(double)*length*length);
        }
        cudaDeviceSynchronize();
        printf("WARMUP rank 0 RECEIVE!!!!!!!!!!!!!!!!!\n");
#if defined (MPI_ASYNC)
        for (i = 0; i < nb_send; i++) {   
            printf("recv buffer %p\n", buffer_pingpong[i]);
            ierr = MPI_Irecv(buffer_pingpong[i], 1, test_type, dest, i, MPI_COMM_WORLD, &request[i]);
        }
#else
        ierr = MPI_Recv(buffer_pingpong[0], 1, test_type, dest, tag, MPI_COMM_WORLD, &status);
 //       ierr = MPI_Recv(buffer_pingpong, 1, test_type, dest, tag, MPI_COMM_WORLD, &status);
#endif
        if (ierr != MPI_SUCCESS) {
            printf("MPI_Recv() returned %d", ierr);
        }
#if defined (MPI_ASYNC)
        MPI_Waitall(nb_send, request, req_status);
	MPI_Barrier(MPI_COMM_WORLD);
#endif

        t1 = MPI_Wtime();
        for (i = 0; i < iterations; i++) {
            printf("rank 0 SEND!!!!!!!!!!!!!!!!!\n");
#if defined (MPI_ASYNC)
            for (j = 0; j < nb_send; j++) {   
                printf("send buffer %p\n", buffer_pingpong[j]);
                ierr = MPI_Isend(buffer_pingpong[j], 1, test_type, dest, j, MPI_COMM_WORLD, &request[j]);
            }
#else
            ierr = MPI_Send(buffer_pingpong[0], 1, test_type, dest, tag, MPI_COMM_WORLD);
 //           ierr = MPI_Send(buffer_pingpong, 1, test_type, dest, tag, MPI_COMM_WORLD);
#endif
            if (ierr != MPI_SUCCESS) {
                printf("MPI_Send() returned %d", ierr);
            }

#if defined (MPI_ASYNC)
            MPI_Waitall(nb_send, request, req_status);
	        MPI_Barrier(MPI_COMM_WORLD);
#endif

            printf("rank 0 RECEIVE!!!!!!!!!!!!!!!!!\n");
    //        cudaMemset(buffer_cuda, 0, sizeof(double)*length*length);
#if defined (MPI_ASYNC)
            for (j = 0; j < nb_send; j++) {   
                printf("recv buffer %p\n", buffer_pingpong[j]);
                ierr = MPI_Irecv(buffer_pingpong[j], 1, test_type, dest, j, MPI_COMM_WORLD, &request[j]);
            }
#else
            ierr = MPI_Recv(buffer_pingpong[0], 1, test_type, dest, tag, MPI_COMM_WORLD, &status);
 //           ierr = MPI_Recv(buffer_pingpong, 1, test_type, dest, tag, MPI_COMM_WORLD, &status);
#endif
            if (ierr != MPI_SUCCESS) {
                printf("MPI_Recv() returned %d", ierr);
            }
#if defined (MPI_ASYNC)
            MPI_Waitall(nb_send, request, req_status);
	        MPI_Barrier(MPI_COMM_WORLD);
#endif
        }
        t2 = MPI_Wtime();
        printf("root send&recv time %f\n", (t2-t1)*1e6/iterations);
        /* verify result */
#if defined (CUDA_TEST)
        for (j = 0; j < nb_send; j++) {   
            cudaMemcpy(buffer_host, buffer_cuda[j], sizeof(double)*length*length, cudaMemcpyDeviceToHost);
            verify_mat_result(buffer_host, length);
        }
#else
        verify_mat_result(buffer_host, length);
#endif      
        
    }
    
    if (rank == dest) {
        /* receive */
        memset(buffer_host, 0, sizeof(double)*length*length);
#if defined (CUDA_TEST)
        for (i = 0; i < nb_send; i++) {   
            cudaMemset(buffer_cuda[i], 0, sizeof(double)*length*length);
            buffer_pingpong[i] = buffer_cuda[i];
        }

#else
        buffer_pingpong[0] = buffer_host;
#endif
        cudaDeviceSynchronize();
        for (i = 0; i < iterations+1; i++) {
            printf("rank 1 RECEIVE!!!!!!!!!!!!!!!!!\n");
#if defined (MPI_ASYNC)
            for (j = 0; j < nb_send; j++) {   
                ierr = MPI_Irecv(buffer_pingpong[j], 1, test_type, root, j, MPI_COMM_WORLD, &request[j]);
            }
#else
            ierr = MPI_Recv(buffer_pingpong[0], 1, test_type, root, tag, MPI_COMM_WORLD, &status);
  //          ierr = MPI_Recv(buffer_pingpong, 1, test_type, root, tag, MPI_COMM_WORLD, &status);
#endif
            if (ierr != MPI_SUCCESS) {
                printf("MPI_Recv() returned %d", ierr);
            }
#if defined (MPI_ASYNC)
            MPI_Waitall(nb_send, request, req_status);
	        MPI_Barrier(MPI_COMM_WORLD);
#endif
//            cudaMemset(buffer_cuda, 0, sizeof(double)*length*length);
            printf("rank 1 SEND!!!!!!!!!!!!!!!!!\n");
#if defined (MPI_ASYNC)
            for (j = 0; j < nb_send; j++) {   
                ierr = MPI_Isend(buffer_pingpong[j], 1, test_type, root, j, MPI_COMM_WORLD, &request[j]);
            }
#else
            ierr = MPI_Send(buffer_pingpong[0], 1, test_type, root, tag, MPI_COMM_WORLD);
 //           ierr = MPI_Send(buffer_pingpong, 1, test_type, root, tag, MPI_COMM_WORLD);
#endif
            if (ierr != MPI_SUCCESS) {
                printf("MPI_Send() returned %d", ierr);
            }
#if defined (MPI_ASYNC)
            MPI_Waitall(nb_send, request, req_status);
	        MPI_Barrier(MPI_COMM_WORLD);
#endif
      //      cudaMemset(buffer_cuda, 0, sizeof(double)*length*length);
        }
    }
    
    /* free newly created datatype */
    ierr = MPI_Type_free(&test_type);
    if (ierr != MPI_SUCCESS) {
        printf("MPI_Type_free() returned %d", ierr);
    }
   /* for (j = 0; j < nb_send; j++) {   
        cudaFree(buffer_cuda[j]);
    }*/
    cudaFree(buffer_cuda_contig);
    cudaFreeHost(buffer_host);
    MPI_Finalize();
    return 0;
}
