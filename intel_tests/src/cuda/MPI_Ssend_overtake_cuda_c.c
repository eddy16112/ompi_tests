/*-----------------------------------------------------------------------------
MESSAGE PASSING INTERFACE TEST CASE SUITE

Copyright - 1996 Intel Corporation

Intel Corporation hereby grants a non-exclusive license under Intel's
copyright to copy, modify and distribute this software for any purpose
and without fee, provided that the above copyright notice and the following
paragraphs appear on all copies.

Intel Corporation makes no representation that the test cases comprising
this suite are correct or are an accurate representation of any standard.

IN NO EVENT SHALL INTEL HAVE ANY LIABILITY FOR ANY DIRECT, INDIRECT OR
SPECULATIVE DAMAGES, (INCLUDING WITHOUT LIMITING THE FOREGOING, CONSEQUENTIAL,
INCIDENTAL AND SPECIAL DAMAGES) INCLUDING, BUT NOT LIMITED TO INFRINGEMENT,
LOSS OF USE, BUSINESS INTERRUPTIONS, AND LOSS OF PROFITS, IRRESPECTIVE OF
WHETHER INTEL HAS ADVANCE NOTICE OF THE POSSIBILITY OF ANY SUCH DAMAGES.

INTEL CORPORATION SPECIFICALLY DISCLAIMS ANY WARRANTIES INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NON-INFRINGEMENT.  THE SOFTWARE PROVIDED HEREUNDER
IS ON AN "AS IS" BASIS AND INTEL CORPORATION HAS NO OBLIGATION TO PROVIDE
MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS OR MODIFICATIONS.
-----------------------------------------------------------------------------*/
/******************************************************************************
		  Test for MPI_Ssend() 

This test verifies that the basic blocking MPI_Sesnd operation does not
return before a matching Receive is posted.

This test uses the first 2 ranks in MPI_COMM_WORLD, the first sending a
message with MPI_Ssend() then MPI_Send, the second waits a while for the 
second message.  If it does not arrive, the test passes.
******************************************************************************/

#include "mpitest_cfg.h"
#include "mpitest.h"
#include "cudatest.h"

#define ITERATIONS 1000000

int main(int argc, char *argv[])
{

    int
     loop_cnt,                  /* counts total number of times through loop     */
     fail,                      /* counts total number of failures               */
     ierr,                      /* return value from MPI calls                   */
     size,                      /* return size from MPI_Error_string             */
     i, flag;

    CUdeviceptr gpu_buff1;      /* buffer for first message                       */
    CUdeviceptr gpu_buff2;      /* buffer for second message                      */

    char
     info_buf[256],             /* buffer for passing mesages to MPITEST         */
     testname[64];              /* the name of this test                         */

    MPI_Status recv1_stat, recv2_stat;

    MPI_Request request;
    /*-----------------------------  MPI_Init  ------------------------------*/
    ierr = MPI_Init(&argc, &argv);
    if (ierr != MPI_SUCCESS) {
        sprintf(info_buf, "Non-zero return code (%d) from MPI_Init()",
                ierr);
        MPITEST_message(MPITEST_FATAL, info_buf);
    }


    sprintf(testname, "MPI_Ssend_overtake");
    if (CUDATEST_UNSUPPORTED_DEVICES == CUDATEST_init(0)) {
        /* There must be some old devices.  Skip the test. */
        printf("Skipping test.  There are some unsupported devices.\n");
        MPI_Finalize();
        return 0;
    }

    /*-----------------------------  MPITEST_init  --------------------------*/
    MPITEST_init(argc, argv);
    if (MPITEST_me == 0) {
        sprintf(info_buf, "Starting %s test", testname);
        MPITEST_message(MPITEST_INFO0, info_buf);
    }

    if (MPITEST_nump < 2) {
        sprintf(info_buf, "At least 2 ranks required to run this test");
        MPITEST_message(MPITEST_FATAL, info_buf);
    }

    /* set the global error counter */
    fail = 0;
    loop_cnt = 0;
    EXIT_ON_ERROR(cuMemAlloc(&gpu_buff1, 3));
    EXIT_ON_ERROR(cuMemAlloc(&gpu_buff2, 4));

    if (MPITEST_me < 2) {

        if (MPITEST_me == 0) {  /* sender */
            loop_cnt = 1;
            MPI_Barrier(MPI_COMM_WORLD);

            ierr = MPI_Ssend((void *)gpu_buff1, 3, MPI_CHAR, 1, 1, MPI_COMM_WORLD);
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Ssend", ierr);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail = 1;
            }
            /* Error Test  */
            ierr = MPI_Send((void *)gpu_buff2, 4, MPI_CHAR, 1, 2, MPI_COMM_WORLD);
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Send", ierr);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail = 1;
            }
            /* Error Test  */
        } /* sender */
        else {                  /* receiver */
            loop_cnt = 1;
            MPI_Barrier(MPI_COMM_WORLD);

            ierr = MPI_Irecv((void *)gpu_buff2, 4, MPI_CHAR, 0, 2, MPI_COMM_WORLD,
                             &request);
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Recv", ierr);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail = 1;
            }
            /* Error Test  */
            for (i = 1; i < ITERATIONS; i++) {
                ierr = MPI_Test(&request, &flag, &recv2_stat);
                if (flag == 1) {
                    sprintf(info_buf,
                            "MPI_Ssend() FAILED; returned prematurely before matching MPI_Recv()");
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    fail = 1;
                    i = ITERATIONS;
                }
            }

            ierr = MPI_Recv((void *)gpu_buff1, 4, MPI_CHAR, 0, 1, MPI_COMM_WORLD,
                            &recv1_stat);
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Recv", ierr);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail = 1;
            }

        }                       /* receiver */

    } else {                    /* rank >= 2 need to match Barrier above */
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /* report overall results  */

    MPITEST_report(loop_cnt - fail, fail, 0, testname);

    MPI_Finalize();
    /* Cannot free until after finalize - still may be in flight */
    EXIT_ON_ERROR(cuMemFree(gpu_buff1));
    EXIT_ON_ERROR(cuMemFree(gpu_buff2));
    CUDATEST_finalize();

    /* 77 is a special return code for a skipped test. So we don't
     * want to return it */
    if(77 == fail) {
        fail++;
    }
    return fail;

}                               /* main() */
