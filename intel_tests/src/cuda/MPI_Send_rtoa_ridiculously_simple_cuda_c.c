#include "mpitest_cfg.h"
#include "mpitest.h"
#include "cudatest.h"

/**
 * Simple test that has root rank send messages to non-root.
 * Default error handler used so no need to check MPI errors.
 */

static int fail;            /*  counts total number of failures                   */
static int i, idx;          /*  utility loop index variables                      */
static int length;          /*  The length of the current buffer                  */
static int len_cnt;         /*  loop counter for message length loop              */
static int max_length;      /*  maximum buffer length specified in config. file   */
static int root;            /*  the root of the current broadcast                 */
static int np;              /*  The number of processors in current communicator  */
static int value;

static void *recv_buffer;
static void *send_buffer;          /* message buffer                                   */
static CUdeviceptr gpu_recv_buffer;
static CUdeviceptr gpu_send_buffer;
static size_t gpu_buffer_size;
#define NUMMSGLENGTHS 2
static int msglengths[NUMMSGLENGTHS] = {65000, 65536};
#define MAX_ERRS 20
static int errs = 0;
static int myrank;
#define MAXMSGLENGTH 65536 /* In bytes */

int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &np);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (CUDATEST_UNSUPPORTED_DEVICES == CUDATEST_init(0)) {
        /* There must be some old devices.  Skip the test. */
        printf("Skipping test.  There are some unsupported devices.\n");
        MPI_Finalize();
        return 0;
    }

    if (myrank == 0) {
        printf("rank=%d: Starting MPI_Send_rtoa_super_simple_cuda_c: Root sends TO All test\n", myrank);
    }

    fail = 0; /* global error counter */

	/* Do the loop 2 times */

	/* Convert the number of bytes in the maximum length
	 * message into the number of elements of the current type.
	 * Note that if I change the datatype, then it can be more general 
	 * MPI_Type_get_extent(datatypes[dtype_cnt], &lb, &size); */
	max_length = MAXMSGLENGTH / sizeof(int);

	/* Allocate send and receive Buffers */
	recv_buffer = malloc((max_length + 1) * sizeof(int));
	send_buffer = malloc((max_length + 1) * sizeof(int));
	gpu_buffer_size = (max_length + 1) * sizeof(int);
	EXIT_ON_ERROR(cuMemAlloc(&gpu_recv_buffer, gpu_buffer_size));
	EXIT_ON_ERROR(cuMemAlloc(&gpu_send_buffer, gpu_buffer_size));


	/* Use the same buffers, but send different length messages from
	 * the buffers.  */

	for (len_cnt = 0; len_cnt < NUMMSGLENGTHS; len_cnt++) {
		/* Sending MPI_INTs so each datatype is 4 bytes */
		length = msglengths[len_cnt] / 4;

		for (root = 0; root < np; root++) {
			errs = 0;
			value = root + len_cnt * 10; /* Make each loop a different value */

			if (myrank == 0) {
				printf("rank=%d: length=%d commsize=%d commtype=MPI_COMM_WORLD data_type=MPI_INT root=%d\n",
					   myrank, length, np, root);
			}

			if (myrank != root) {
				/* Initialize the receive buffer to -1 */
				for (idx = 0; idx < length + 1; idx++) {
					((int *)recv_buffer)[idx] = -1;
				}

				/* Initialize entire GPU recv buffer to match host recv buffer */
				EXIT_ON_ERROR(cuMemcpy(gpu_recv_buffer, (CUdeviceptr)recv_buffer, 
									   gpu_buffer_size));

				MPI_Recv((void *)gpu_recv_buffer, length, MPI_INT, root, 42, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				/* Move entire GPU recv buffer back into host recv buffer for checking */
				EXIT_ON_ERROR(cuMemcpy((CUdeviceptr)recv_buffer, gpu_recv_buffer,
									   gpu_buffer_size));

				for (idx = 0; idx < length; idx++) {
					if (((int *)recv_buffer)[idx] != value) {
						errs++;
						if (errs < MAX_ERRS) {
							printf("Error in recv_buffer[%d]: exp=%d, act=%d\n",
								   idx, value, ((int *)recv_buffer)[idx]);
						} else if (MAX_ERRS == errs) {
							printf("Too many errors.  Output stopped.\n");
						}
					}
				}
				if (errs) {
					fail++;
				}
				printf("rank=%d: %d errors in buffer length=%d commsize=%d data_type=MPI_INT root=%d\n",
					   myrank, errs, length, np, root);
			
			}

			/* Send from each root node to all other nodes   */
			if (myrank == root) {
				for (idx = 0; idx < length + 1; idx++) {
					((int *)send_buffer)[idx] = value;
				}

#if 0
				/* Doing this a second time allows the test to pass. */
				for (idx = 0; idx < length + 1; idx++) {
					((int *)send_buffer)[idx] = myrank;
				}
#endif
				/* Initialize entire GPU send buffer to match host send buffer */
				EXIT_ON_ERROR(cuMemcpy(gpu_send_buffer, (CUdeviceptr)send_buffer,
									   gpu_buffer_size));

				for (i = 0; i < np; i++) {
					if (i == root) {
						/* Do not send to self */
						continue;
					}

					MPI_Send((void *)gpu_send_buffer, length, MPI_INT, i, 42, MPI_COMM_WORLD);
				}
			}
		
			MPI_Barrier(MPI_COMM_WORLD);

		} /* Loop over Roots  */

		
		MPI_Barrier(MPI_COMM_WORLD);

	} /* Loop over Roots  */

	free(send_buffer);
	free(recv_buffer);
	EXIT_ON_ERROR(cuMemFree(gpu_send_buffer));
	EXIT_ON_ERROR(cuMemFree(gpu_recv_buffer));

	MPI_Barrier(MPI_COMM_WORLD);

	
	if (fail) {
		printf("rank=%d: Test FAILED\n", myrank);
	} else {
		printf("rank=%d: Test PASSED\n", myrank);
	}

    MPI_Finalize();
	CUDATEST_finalize();
    return fail;

}
