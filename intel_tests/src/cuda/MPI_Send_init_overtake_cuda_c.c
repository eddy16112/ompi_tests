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
		  Test for Persistent MPI_Send_init()

This test verifies that the  MPI_Send_init/MPI_Start  operation with
MPI_Recv_init/MPI_Start  delivers messages in the order they were sent.

This test uses the first 2 ranks in MPI_COMM_WORLD, first sending a large
message then a number of short ones, and ensures that they are received
in the proper order by verifying the data received.
******************************************************************************/

#include "mpitest_cfg.h"
#include "mpitest.h"
#include "cudatest.h"

#define LONGLENGTH 32*1024
#define NUMSHORT   20

int main(int argc, char *argv[])
{

    int
     loop_cnt,                  /* counts total number of times through loop        */
     fail,                      /* counts total number of failures                  */
     ierr,                      /* return value from MPI calls                      */
     size,                      /* return size from MPI_Error_string                */
     i, errors;

    char shortb[NUMSHORT];      /* buffer for short messages                */
    char buffer[LONGLENGTH];    /* send/recv buffer for long message        */
    CUdeviceptr gpu_shortb;
    CUdeviceptr gpu_buffer;
    CUdeviceptr gpuptr;

    char
     info_buf[256],             /* buffer for passing mesages to MPITEST    */
     testname[64];              /* the name of this test                    */

    struct dataTemplate
     value;                     /* dataTemplate for initializing buffers    */

    MPI_Status recv_stat, send_stat;    /* MPI  status structure                    */

    MPI_Request recv_request,   /* MPI request structure                    */
        send_request;           /* MPI request structure                    */
    int len_recv;

    /*-----------------------------  MPI_Init  -----------------------------*/
    ierr = MPI_Init(&argc, &argv);
    if (ierr != MPI_SUCCESS) {
        sprintf(info_buf, "Non-zero return code (%d) from MPI_Init()",
                ierr);
        MPITEST_message(MPITEST_FATAL, info_buf);
    }


    sprintf(testname, "MPI_Send_init_overtake");
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
    loop_cnt = 0;
    fail = 0;

    EXIT_ON_ERROR(cuMemAlloc(&gpu_shortb, NUMSHORT));
    EXIT_ON_ERROR(cuMemAlloc(&gpu_buffer, LONGLENGTH));

    if (MPITEST_me < 2) {

        if (MPITEST_me == 0) {  /* sender */
            MPITEST_dataTemplate_init(&value, 0);
            MPITEST_init_buffer_inc(MPITEST_char, LONGLENGTH, value,
                                    &buffer);
            EXIT_ON_ERROR(cuMemcpy(gpu_buffer, (CUdeviceptr)buffer, LONGLENGTH));
            MPITEST_dataTemplate_init(&value, 1);
            MPITEST_init_buffer_inc(MPITEST_char, NUMSHORT, value,
                                    &shortb);
            EXIT_ON_ERROR(cuMemcpy(gpu_shortb, (CUdeviceptr)shortb, NUMSHORT));

            MPI_Barrier(MPI_COMM_WORLD);

            ierr = MPI_Send_init((void *)gpu_buffer, LONGLENGTH, MPI_CHAR, 1, 1,
                                 MPI_COMM_WORLD, &send_request);
            if (ierr != MPI_SUCCESS) {
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Send_init",
                        ierr);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail++;
            }
            /* Send_init Error Test  */
            ierr = MPI_Start(&send_request);
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Start for send",
                        ierr);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail++;
            }


            ierr = MPI_Wait(&send_request, &send_stat);

            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Wait on send",
                        ierr);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail++;
            }



            /* Wait Error Test  */
 /*-------- Call MPI_Request_free -------*/
            ierr = MPI_Request_free(&send_request);
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Request_free for Send",
                        ierr);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail++;
            }


            for (i = 0; i < NUMSHORT; i++) {
                gpuptr = (CUdeviceptr)((char *)gpu_shortb + i);
                ierr = MPI_Send_init((void *)gpuptr, 1, MPI_CHAR, 1, 1,
                                     MPI_COMM_WORLD, &send_request);
                if (ierr != MPI_SUCCESS) {
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    sprintf(info_buf,
                            "Non-zero return code (%d) from MPI_Isend",
                            ierr);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    MPI_Error_string(ierr, &info_buf[0], &size);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    fail++;
                }

                /* Isend Error Test  */
                ierr = MPI_Start(&send_request);
                if (ierr != MPI_SUCCESS) {
                    sprintf(info_buf,
                            "Non-zero return code (%d) from MPI_Start for send",
                            ierr);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    MPI_Error_string(ierr, &info_buf[0], &size);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    fail++;
                }

                ierr = MPI_Wait(&send_request, &send_stat);

                if (ierr != MPI_SUCCESS) {
                    sprintf(info_buf,
                            "Non-zero return code (%d) from MPI_Wait on send",
                            ierr);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    MPI_Error_string(ierr, &info_buf[0], &size);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    fail++;
                }

                /* Wait Error Test  */
                ierr = MPI_Request_free(&send_request);
                if (ierr != MPI_SUCCESS) {
                    sprintf(info_buf,
                            "Non-zero return code (%d) from MPI_Request_free for send",
                            ierr);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    MPI_Error_string(ierr, &info_buf[0], &size);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    fail++;
                }

            }
            loop_cnt = 1 + NUMSHORT;

        } /* sender */
        else {                  /* receiver */
            MPITEST_dataTemplate_init(&value, -1);
            MPITEST_init_buffer(MPITEST_char, LONGLENGTH, value, &buffer);
            MPITEST_init_buffer(MPITEST_char, NUMSHORT, value, &shortb);
            EXIT_ON_ERROR(cuMemcpy(gpu_buffer, (CUdeviceptr)buffer, LONGLENGTH));
            EXIT_ON_ERROR(cuMemcpy(gpu_shortb, (CUdeviceptr)shortb, NUMSHORT));

            MPI_Barrier(MPI_COMM_WORLD);

            ierr =
                MPI_Recv_init((void *)gpu_buffer, LONGLENGTH, MPI_CHAR, 0, 1,
                              MPI_COMM_WORLD, &recv_request);
            if (ierr != MPI_SUCCESS) {
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Recv_init",
                        ierr);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail++;
            }

            /* Recv_init  Error Test  */
            ierr = MPI_Start(&recv_request);
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Start for receive",
                        ierr);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail++;
            }

            ierr = MPI_Wait(&recv_request, &recv_stat);
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Wait on Irecv",
                        ierr);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail++;
            }
            /* Wait Error Test  */
            ierr = MPI_Request_free(&recv_request);
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Request_free for receive",
                        ierr);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail++;
            }

            for (i = 0; i < NUMSHORT; i++) {
                len_recv=NUMSHORT-i;
                gpuptr = (CUdeviceptr)((char *)gpu_shortb + i);
                ierr =
                    MPI_Recv_init((void *)gpuptr, len_recv, MPI_CHAR, 0, 1,
                                  MPI_COMM_WORLD, &recv_request);
                if (ierr != MPI_SUCCESS) {
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    sprintf(info_buf,
                            "Non-zero return code (%d) from MPI_Recv_init",
                            ierr);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    MPI_Error_string(ierr, &info_buf[0], &size);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    fail++;
                }

                /* Irecv Error Test  */
                ierr = MPI_Start(&recv_request);
                if (ierr != MPI_SUCCESS) {
                    sprintf(info_buf,
                            "Non-zero return code (%d) from MPI_Start for receive",
                            ierr);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    MPI_Error_string(ierr, &info_buf[0], &size);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    fail++;
                }

                ierr = MPI_Wait(&recv_request, &recv_stat);
                if (ierr != MPI_SUCCESS) {
                    sprintf(info_buf,
                            "Non-zero return code (%d) from MPI_Wait on receive",
                            ierr);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    MPI_Error_string(ierr, &info_buf[0], &size);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    fail++;
                }
                /* Wait on Irecv Error Test  */
                ierr = MPI_Request_free(&recv_request);
                if (ierr != MPI_SUCCESS) {
                    sprintf(info_buf,
                            "Non-zero return code (%d) from MPI_Request_free for receive",
                            ierr);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    MPI_Error_string(ierr, &info_buf[0], &size);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    fail++;
                }
            }
            /*
             * Check Received data
             */

            loop_cnt = 1 + NUMSHORT;
            EXIT_ON_ERROR(cuMemcpy((CUdeviceptr)buffer, gpu_buffer, LONGLENGTH))
            MPITEST_dataTemplate_init(&value, 0);
            errors =
                MPITEST_buffer_errors_inc(MPITEST_char, LONGLENGTH, value,
                                          &buffer);
            if (errors) {
                sprintf(info_buf, "%d errors found in first (long) buffer",
                        errors);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail++;
            }
            EXIT_ON_ERROR(cuMemcpy((CUdeviceptr)shortb, gpu_shortb, NUMSHORT));
            MPITEST_dataTemplate_init(&value, 1);
            errors =
                MPITEST_buffer_errors_inc(MPITEST_char, NUMSHORT, value,
                                          &shortb);
            if (errors) {
                sprintf(info_buf, "%d short messages received incorrectly",
                        errors);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail = fail + errors;
            }


        }                       /* receiver */

    } else {                    /* rank >= 2 need to match Barrier above */
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /* report overall results  */

    MPITEST_report(loop_cnt - fail, fail, 0, testname);

    EXIT_ON_ERROR(cuMemFree(gpu_buffer));
    EXIT_ON_ERROR(cuMemFree(gpu_shortb));
    MPI_Finalize();

    /* 77 is a special return code for a skipped test. So we don't
     * want to return it */
    if(77 == fail) {
        fail++;
    }
	CUDATEST_finalize();
    return fail;

}                               /* main() */
