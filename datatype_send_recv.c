#include "mpi.h"

#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <unistd.h>
#include <cuda_runtime.h>


#define DDT_INDEX_LOW   1
#define DDT_INDEX_UP    2
#define DDT_VEC         3
#define DDT_CONT        4

#define CUDA_TEST
//#define MPI_ASYNC

static int *displs, *blklens;
int iterations;

size_t compute_buffer_length(MPI_Datatype pdt, int count)
{
    MPI_Aint extent, lb, true_extent, true_lb;
    size_t length;

    MPI_Type_get_extent(pdt, &lb, &extent);
    MPI_Type_get_true_extent(pdt, &true_lb, &true_extent); (void)true_lb;
    length = true_lb + true_extent + (count - 1) * extent;

    return  length;
}

void lower_matrix(int mat_size, MPI_Datatype *lower)
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
                            lower);
    if (ierr != MPI_SUCCESS) { 
        printf("MPI_Type_indexed() returned %d", ierr);
    }
    ierr = MPI_Type_commit (lower);
    if (ierr != MPI_SUCCESS) { 
        printf("MPI_Type_commit() returned %d", ierr);
    }
}

void upper_matrix(int mat_size, MPI_Datatype *upper)
{
    int i, ierr;

    displs = (int*)malloc( sizeof(int) * mat_size );
    blklens = (int*)malloc( sizeof(int) * mat_size );

    for( i = 0; i < mat_size; i++ ) {
        displs[i] = i * mat_size;
        blklens[i] = i+1;
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

static void fill_lower_matrix(double *mat, int mat_size)
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

static void verify_lower_mat_result(double *mat, int mat_size)
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
    } else {
        printf("no error is found\n");
    }
    
    free( displs );
    free( blklens );
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

void create_contiguous(int count, MPI_Datatype *cont)
{
    int i, ierr;

    ierr = MPI_Type_contiguous(count, MPI_DOUBLE, cont);
    if (ierr != MPI_SUCCESS) { 
        printf("MPI_Type_contiguous() returned %d", ierr);
    }
    ierr = MPI_Type_commit (cont);
    if (ierr != MPI_SUCCESS) { 
        printf("MPI_Type_commit() returned %d", ierr);
    }
}

void fill_contiguous(double* vp, int count)
{
    int i;
    for (i = 0; i < count; i++) {
        vp[i] = i;
    }
}

void verify_contiguous(double *vp, int count)
{
    int i;
    int error = 0;
    for (i = 0; i < count; i++) {
        if (vp[i] != i) {
            error ++;
        } 
    }
    if (error != 0) {
        printf("%d error is found\n", error);
    } else {
        printf("no error is found\n");
    }
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

void ping_pong(MPI_Datatype *test_type, size_t ddt_size, void *buffer_host, void *buffer_cuda, int dest)
{
    int ierr;
    int tag = 0;
    double t1, t2;
    void *buffer_pingpong;
    MPI_Status status;
    int i;
    
#if defined (CUDA_TEST)
    cudaMemset(buffer_cuda, 0, ddt_size);
    buffer_pingpong = buffer_cuda;
    cudaMemcpy(buffer_cuda, buffer_host, ddt_size, cudaMemcpyHostToDevice);
#else
    buffer_pingpong = buffer_host;
#endif
    /* warm up */
    printf("WARM UP rank 0 SEND!!!!!!!!!!!!!!!!!\n");
    ierr = MPI_Send(buffer_pingpong, 1, *test_type, dest, tag, MPI_COMM_WORLD);

    if (ierr != MPI_SUCCESS) {
        printf("MPI_Send() returned %d", ierr);
    }
        
    cudaMemset(buffer_cuda, 0, ddt_size);
    printf("WARMUP rank 0 RECEIVE!!!!!!!!!!!!!!!!!\n");

    ierr = MPI_Recv(buffer_pingpong, 1, *test_type, dest, tag, MPI_COMM_WORLD, &status);

    if (ierr != MPI_SUCCESS) {
        printf("MPI_Recv() returned %d", ierr);
    }
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    t1 = MPI_Wtime();
    for (i = 0; i < iterations; i++) {
        printf("rank 0 SEND!!!!!!!!!!!!!!!!!\n");
        ierr = MPI_Send(buffer_pingpong, 1, *test_type, dest, tag, MPI_COMM_WORLD);

        if (ierr != MPI_SUCCESS) {
            printf("MPI_Send() returned %d", ierr);
        }

        printf("rank 0 RECEIVE!!!!!!!!!!!!!!!!!\n");
        cudaMemset(buffer_cuda, 0, ddt_size);

        ierr = MPI_Recv(buffer_pingpong, 1, *test_type, dest, tag, MPI_COMM_WORLD, &status);
        
        if (ierr != MPI_SUCCESS) {
            printf("MPI_Recv() returned %d", ierr);
        }
    }
    t2 = MPI_Wtime();
    printf("root send&recv time %f\n", (t2-t1)*1e6/iterations);
    /* pop out to verify result */
#if defined (CUDA_TEST)
    cudaMemcpy(buffer_host, buffer_cuda, ddt_size, cudaMemcpyDeviceToHost);
#endif
    
}

void pong_ping(MPI_Datatype *test_type, size_t ddt_size, void *buffer_host, void *buffer_cuda, int src)
{
    int ierr;
    int tag = 0;
    void *buffer_pingpong;
    MPI_Status status;
    int i;
    
#if defined (CUDA_TEST)
    cudaMemset(buffer_cuda, 0, ddt_size);
    buffer_pingpong = buffer_cuda;
#else
    buffer_pingpong = buffer_host;
#endif
    printf("rank 1 RECEIVE!!!!!!!!!!!!!!!!!\n");
    ierr = MPI_Recv(buffer_pingpong, 1, *test_type, src, tag, MPI_COMM_WORLD, &status);
    if (ierr != MPI_SUCCESS) {
        printf("MPI_Recv() returned %d", ierr);
    }

    printf("rank 1 SEND!!!!!!!!!!!!!!!!!\n");
    ierr = MPI_Send(buffer_pingpong, 1, *test_type, src, tag, MPI_COMM_WORLD);

    if (ierr != MPI_SUCCESS) {
        printf("MPI_Send() returned %d", ierr);
    }

    cudaMemset(buffer_cuda, 0, ddt_size);
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    
    for (i = 0; i < iterations; i++) {
        printf("rank 1 RECEIVE!!!!!!!!!!!!!!!!!\n");
        ierr = MPI_Recv(buffer_pingpong, 1, *test_type, src, tag, MPI_COMM_WORLD, &status);
        if (ierr != MPI_SUCCESS) {
            printf("MPI_Recv() returned %d", ierr);
        }

        printf("rank 1 SEND!!!!!!!!!!!!!!!!!\n");
        ierr = MPI_Send(buffer_pingpong, 1, *test_type, src, tag, MPI_COMM_WORLD);

        if (ierr != MPI_SUCCESS) {
            printf("MPI_Send() returned %d", ierr);
        }

        cudaMemset(buffer_cuda, 0, ddt_size);
    }
}

int main(int argc, char **argv)
{
    MPI_Comm comm;  
    
    MPI_Status status;
    
    MPI_Datatype root_type, dest_type;
    
    int ierr;
    
    int rank, size;
    
    int length, i;
    
    double *buffer_pingpong, *buffer_cuda, *buffer_host;
    
    int root, dest;
    
    size_t root_size, dest_size;
    
    int root_ddt, dest_ddt;

    length = 0;
    iterations = 0;
    
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
        cudaSetDevice(0);
    }
    
   root_ddt = DDT_INDEX_LOW;
   dest_ddt = DDT_INDEX_UP;
    
//    root_ddt = DDT_VEC;
//    dest_ddt = DDT_VEC;
    
//    root_ddt = DDT_CONT;
//    dest_ddt = DDT_CONT;
    
   /* lower triangular matrix */
   if (root_ddt == DDT_INDEX_LOW) {
       lower_matrix(length, &root_type);
       root_size = sizeof(double)*length*length;
   } else if (root_ddt == DDT_VEC) {
       create_vector(length, 384, 512, &root_type);
       root_size = compute_buffer_length(root_type, 1);
   } else if (root_ddt == DDT_CONT) {
       create_contiguous(length, &root_type);
       root_size = sizeof(double)*length;
   }
    
    // sender
    if (rank == root) {

        cudaMallocHost((void **)&buffer_host, root_size);
#if defined (CUDA_TEST)    
        cudaMalloc((void **)&buffer_cuda, root_size);
#endif
        memset(buffer_host, 0, root_size);
        
        if (root_ddt == DDT_INDEX_LOW) {
            fill_lower_matrix(buffer_host, length);
        } else if (root_ddt == DDT_VEC) {
            fill_vectors(buffer_host, length, 384, 512);
        } else if (root_ddt == DDT_CONT) {
            fill_contiguous(buffer_host, length);
        }
        
        ping_pong(&root_type, root_size, buffer_host, buffer_cuda, dest);

        if (root_ddt == DDT_INDEX_LOW) {
            verify_lower_mat_result(buffer_host, length);   
        } else if (root_ddt == DDT_VEC) {
            verify_vectors(buffer_host, length, 384, 512); 
        } else if (root_ddt == DDT_CONT) {
            verify_contiguous(buffer_host, length);
        }  
        
    }
    
    if (dest_ddt == DDT_INDEX_LOW) {
        /* lower triangular matrix */
        lower_matrix(length, &dest_type);
        dest_size = sizeof(double)*length*length;
    } else if (dest_ddt == DDT_INDEX_UP){
        /* upper triangular matrix */
        upper_matrix(length, &dest_type);
        dest_size = sizeof(double)*length*length;
    } else if (dest_ddt == DDT_VEC) {
        create_vector(length, 384, 512, &dest_type);
        dest_size = compute_buffer_length(dest_type, 1);
    } else if (dest_ddt == DDT_CONT) {
        create_contiguous(length, &dest_type);
        dest_size = sizeof(double)*length;
    }
    
    // receiver
    if (rank == dest) {
        
        cudaMallocHost((void **)&buffer_host, dest_size);
    #if defined (CUDA_TEST)    
        cudaMalloc((void **)&buffer_cuda, dest_size);
    #endif
        /* receive */
        memset(buffer_host, 0, dest_size);
        pong_ping(&dest_type, dest_size, buffer_host, buffer_cuda, root);

    }
    
    /* free newly created datatype */
    ierr = MPI_Type_free(&root_type);
    ierr = MPI_Type_free(&dest_type);
    if (ierr != MPI_SUCCESS) {
        printf("MPI_Type_free() returned %d", ierr);
    }
    cudaFree(buffer_cuda);
    cudaFreeHost(buffer_host);
    MPI_Finalize();
    return 0;
}