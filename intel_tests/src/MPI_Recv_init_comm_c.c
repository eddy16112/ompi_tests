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
*                          Functional test for MPI_Recv_init
*
*  This test verifies MPI_Recv_init sorts based on the communicator
*  specified.  4 communicators are created, and an assortment of messages
*  are sent from rank 0 to rank 1, using the first 3 communicators.
*
*  Rank 1 starts a receive on comm 3 (which should not be satisfied until
*  the end), then the receives specified above.  Messages should be
*  received in the order sent for each communicator, not in the order
*  the receives were posted (which is different).
******************************************************************************/

#define  PROG_NAME   "MPI_Recv_init_comm"
#define  PROG_DESC   " "

#define  NUMMESG     20         /* Number of messages to send                 */
#define  NUMELM      10         /* # of elements to send and receive                 */

#include "mpitest_cfg.h"
#include "mpitest.h"

int main(int argc, char *argv[])
{
    int
     flag,                      /* flag return from  MPI_Test calls                  */
     i, j,                      /* general index variable                            */
     cnt_len,                   /* received length returned by MPI_Get_Count         */
     dest,                      /* Destination of Isend message                      */
     fail,                      /* counts total number # of failures                 */
     error,                     /* errors from one MPI call                          */
     ierr,                      /* return value from MPI calls                       */
     errorclass,                /* error class of ierr                               */
     size,                      /* length of error message                           */
     send_loop_cnt,             /* count tries on send Test loop                     */
     recv_loop_cnt,             /* count tries on receive Test loop                  */
     total_cnt;                 /* total test count                                  */

    char
     testname[128],             /* the name of this test                             */
     info_buf[256];             /* for sprintf                                       */

    struct dataTemplate
     value;                     /* dataTemplate for initializing buffers             */

    MPI_Request send_request[4 * NUMMESG], recv_request[4 * NUMMESG + 1];       /* MPI request structure             */

    MPI_Status send_stat,       /* Send Source/Tag information                       */
        recv_stat;              /* Recv Source/Tag information                       */

    MPI_Comm comm[4];           /* A number of communicators to sort over            */
    int
     recv_buffer[4 * NUMMESG + 1][NUMELM],      /* input to Irecv            */
     send_buffer[4 * NUMMESG][NUMELM];  /* input to Isend            */

    char
     bsend_buff[NUMMESG * (8 * NUMELM + MPI_BSEND_OVERHEAD + 100)];
    void
     *buf_ptr;

    /*-----------------------  MPI Initialization  --------------------------*/

    /* Initialize the MPI environment and test environment. */

    ierr = MPI_Init(&argc, &argv);
    if (ierr != MPI_SUCCESS) {
        sprintf(info_buf, "Non-zero return code (%d) from MPI_Init()",
                ierr);
        MPITEST_message(MPITEST_FATAL, info_buf);
    }

    sprintf(testname, "%s:  %s", PROG_NAME, PROG_DESC);

    /*--------------------  MPITEST Initialization  -------------------------*/

    MPITEST_init(argc, argv);

    if (MPITEST_me == 0) {
        sprintf(info_buf, "Starting:  %s ", testname);
        MPITEST_message(MPITEST_INFO0, info_buf);
    }

    total_cnt = 0;
    fail = 0;

    /* For this simple wait test we need only two nodes  */

    if (MPITEST_nump < 2) {
        if (MPITEST_me == 0) {
            sprintf(info_buf,
                    "At least 2 ranks required to run this test");
            MPITEST_message(MPITEST_FATAL, info_buf);
        }
    }

    /* Create some communicators.  Use the first 3 to send some messages.
       Leave the last idle until the end
     */
    comm[0] = MPI_COMM_WORLD;
    for (i = 1; i < 4; i++) {
        ierr = MPI_Comm_dup(MPI_COMM_WORLD, &comm[i]);
        if (ierr != MPI_SUCCESS) {
            sprintf(info_buf,
                    "Non-zero return code (%d) from MPI_Comm_dup", ierr);
            MPITEST_message(MPITEST_NONFATAL, info_buf);
            MPI_Error_string(ierr, &info_buf[0], &size);
            MPITEST_message(MPITEST_FATAL, info_buf);
            fail++;
        }
    }

    /*-------------------------------  Sends --------------------------------*/
    if (MPITEST_me == 0) {
        ierr = MPI_Buffer_attach(bsend_buff,
                                 NUMMESG * (8 * NUMELM +
                                            MPI_BSEND_OVERHEAD + 100));
        if (ierr != MPI_SUCCESS) {
            sprintf(info_buf,
                    "Non-zero return code (%d) from MPI_Buffer_attach",
                    ierr);
            MPITEST_message(MPITEST_NONFATAL, info_buf);
            MPI_Error_string(ierr, &info_buf[0], &size);
            MPITEST_message(MPITEST_FATAL, info_buf);
            fail++;
        }

        /* Initialize Send buffers */

        for (i = 0; i < 4 * NUMMESG; i++) {
            MPITEST_dataTemplate_init(&value, i);
            MPITEST_init_buffer(MPITEST_int, NUMELM, value,
                                &send_buffer[i]);

        }
        MPI_Barrier(MPI_COMM_WORLD);

        j = -1;
        for (i = 0; i < NUMMESG; i++) {
            j++;
            if (j == 3)
                j = 0;
            ierr = MPI_Send_init(send_buffer[4 * i],
                                 NUMELM,
                                 MPI_INT,
                                 1, 1, comm[j], &send_request[4 * i]);

            total_cnt++;
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Send_init",
                        ierr);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail++;
            }
            /* Isend  Error Test  */
            j++;
            if (j == 3)
                j = 0;
            ierr = MPI_Bsend_init(send_buffer[4 * i + 1],
                                  NUMELM,
                                  MPI_INT,
                                  1, 1, comm[j], &send_request[4 * i + 1]);

            total_cnt++;
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Bsend_init",
                        ierr);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail++;
            }
            /* Ibsend  Error Test  */
            j++;
            if (j == 3)
                j = 0;
            ierr = MPI_Rsend_init(send_buffer[4 * i + 2],
                                  NUMELM,
                                  MPI_INT,
                                  1, 1, comm[j], &send_request[4 * i + 2]);

            total_cnt++;
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Rsend_init",
                        ierr);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail++;
            }
            /* Irsend  Error Test  */
            j++;
            if (j == 3)
                j = 0;
            ierr = MPI_Ssend_init(send_buffer[4 * i + 3],
                                  NUMELM,
                                  MPI_INT,
                                  1, 1, comm[j], &send_request[4 * i + 3]);

            total_cnt++;
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Ssend_init",
                        ierr);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail++;
            }
            /* Ssend_init  Error Test  */
        }                       /* Send Loop */

        /*---------------------  Startall sends  ---------------------------*/


        ierr = MPI_Startall(4 * NUMMESG, send_request);

        if (ierr != MPI_SUCCESS) {
            sprintf(info_buf,
                    "Non-zero return code (%d) from MPI_Startall on send",
                    ierr);
            MPITEST_message(MPITEST_NONFATAL, info_buf);
            MPI_Error_string(ierr, &info_buf[0], &size);
            MPITEST_message(MPITEST_NONFATAL, info_buf);
            fail++;
        }

        /*----------------  Test for all messages to send ---------------*/

        for (i = 0; i < 4 * NUMMESG; i++) {
            flag = FALSE;
            ierr = FALSE;
            send_loop_cnt = 0;
            while (!flag && !ierr) {
                ierr = MPI_Test(&send_request[i], &flag, &send_stat);
                send_loop_cnt++;
            }
            total_cnt++;
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code ierr/flag (%d/%d) from MPI_Test on Sends, after %d/%d  calls/loops to MPI_Test",
                        ierr, flag, send_loop_cnt, i);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail++;
            }                   /* End of Error Test  */
        }                       /* End of Test on sends */

        for (i = 0; i < 4 * NUMMESG; i++) {
            if (send_request[i] != MPI_REQUEST_NULL)
                MPI_Request_free(&send_request[i]);
        }

        /* Wait before sending the final message on the last comm.
         */
        MPI_Barrier(comm[3]);
        ierr = MPI_Send(send_buffer[0], NUMELM, MPI_INT, 1, 1, comm[3]);

        total_cnt++;
        if (ierr != MPI_SUCCESS) {
            sprintf(info_buf, "Non-zero return code (%d) from MPI_Send",
                    ierr);
            MPITEST_message(MPITEST_NONFATAL, info_buf);
            MPI_Error_string(ierr, &info_buf[0], &size);
            MPITEST_message(MPITEST_NONFATAL, info_buf);
            fail++;
        }

        /* send  Error Test  */
        ierr = MPI_Buffer_detach(&buf_ptr, &size);
        if (ierr != MPI_SUCCESS) {
            sprintf(info_buf,
                    "Non-zero return code (%d) from MPI_Buffer_detach",
                    ierr);
            MPITEST_message(MPITEST_NONFATAL, info_buf);
            MPI_Error_string(ierr, &info_buf[0], &size);
            MPITEST_message(MPITEST_FATAL, info_buf);
            fail++;
        }

    }

    /* End of sends from node 0 */
 /*--------------------------------- Receives  ---------------------------*/
    if (MPITEST_me == 1) {
        /* Initialize Receive buffers - start with the message to be received
           last on comm3, then do the rest
         */

        ierr = MPI_Recv_init(recv_buffer[4 * NUMMESG],
                             NUMELM,
                             MPI_INT,
                             0, 1, comm[3], &recv_request[4 * NUMMESG]);

        total_cnt++;
        if (ierr != MPI_SUCCESS) {
            sprintf(info_buf,
                    "Non-zero return code (%d) from MPI_Recv_init", ierr);
            MPITEST_message(MPITEST_NONFATAL, info_buf);
            MPI_Error_string(ierr, &info_buf[0], &size);
            MPITEST_message(MPITEST_NONFATAL, info_buf);
            fail++;
        }
        /* Recv_init Error Test  */
        ierr = MPI_Start(&recv_request[4 * NUMMESG]);
        if (ierr != MPI_SUCCESS) {
            sprintf(info_buf,
                    "Non-zero return code (%d) from MPI_Start on receive",
                    ierr);
            MPITEST_message(MPITEST_NONFATAL, info_buf);
            MPI_Error_string(ierr, &info_buf[0], &size);
            MPITEST_message(MPITEST_NONFATAL, info_buf);
            fail++;
        }

        j = -1;
        for (i = 0; i < 4 * NUMMESG; i++) {
            j++;
            if (j == 3)
                j = 0;
            MPITEST_dataTemplate_init(&value, -1);
            MPITEST_init_buffer(MPITEST_int, NUMELM, value,
                                &recv_buffer[i]);

            ierr = MPI_Recv_init(recv_buffer[i],
                                 NUMELM,
                                 MPI_INT, 0, 1, comm[j], &recv_request[i]);

            total_cnt++;
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Recv_initfor message %d",
                        ierr, i);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail++;
            }
            /* Recv_init Error Test  */
        }                       /* Receive Loop  */

        /*---------------------  Startall receives  --------------------*/

        ierr = MPI_Startall(4 * NUMMESG, recv_request);

        if (ierr != MPI_SUCCESS) {
            sprintf(info_buf,
                    "Non-zero return code (%d) from MPI_Startall on receive",
                    ierr);
            MPITEST_message(MPITEST_NONFATAL, info_buf);
            MPI_Error_string(ierr, &info_buf[0], &size);
            MPITEST_message(MPITEST_NONFATAL, info_buf);
            fail++;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        /*---------------------  Receive  Test  ----------------------*/

        for (i = 0; i < 4 * NUMMESG; i++) {
            flag = FALSE;
            ierr = FALSE;
            recv_loop_cnt = 0;

            while (!flag && !ierr) {
                ierr = MPI_Test(&recv_request[i], &flag, &recv_stat);
                recv_loop_cnt++;
            }

            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code ierr/flag (%d/%d) from MPI_Test on receives, after %d calls to MPI_Test on Receive",
                        ierr, flag, recv_loop_cnt);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail++;

            }


            /* End of Test error checking */
            /* Set up the dataTemplate for checking the recv'd buffer. */
            MPITEST_dataTemplate_init(&value, i);

            error =
                MPITEST_buffer_errors(MPITEST_int, NUMELM, value,
                                      recv_buffer[i]);
            if (error) {
                sprintf(info_buf,
                        "Unexpected value in buffer %d, actual =  %d    expected = %d",
                        i, recv_buffer[i][0], i);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail++;
            }

            /* Call the MPI_Get_Count function, and compare value with NUMELM */

            cnt_len = -1;

            ierr = MPI_Get_count(&recv_stat, MPI_INT, &cnt_len);
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Get_count",
                        ierr);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail++;
            }
            /*
             * Print non-fatal error if Received length not equal to send
             * length
             */

            error = NUMELM - cnt_len;

            if (error) {
                sprintf(info_buf,
                        "send/receive lengths differ - Sender(length)=%d,  Receiver(index/length)=%d/%d",
                        NUMELM, i, cnt_len);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail++;
            }
            /*
             * Print non-fatal error if tag is not correct.
             */
            total_cnt++;
            if (recv_stat.MPI_TAG != 1) {
                sprintf(info_buf, "Unexpected tag value=%d, expected=%d",
                        recv_stat.MPI_TAG, i);
                MPITEST_message(MPITEST_NONFATAL, info_buf);

                fail++;
            }

        }                       /* End of for loop: test received records */

        for (i = 0; i < 4 * NUMMESG; i++) {
            if (recv_request[i] != MPI_REQUEST_NULL)
                MPI_Request_free(&recv_request[i]);
        }

        /* Wait here, then receive for the first recv posted on comm3 */
        MPI_Barrier(comm[3]);
        MPI_Wait(&recv_request[4 * NUMMESG], &recv_stat);
        MPI_Request_free(&recv_request[4 * NUMMESG]);

    }
    /* End of node 1 receives  */
    if (MPITEST_me >= 2) {      /* rank >= 2 need to match Barrier above */
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Barrier(comm[3]);
    }

    for (j = 1; j < 4; j++) {
        ierr = MPI_Comm_free(&comm[j]);
        if (ierr != MPI_SUCCESS) {
            sprintf(info_buf,
                    "Non-zero return code (%d) from MPI_Comm_free", ierr);
            MPITEST_message(MPITEST_NONFATAL, info_buf);
            MPI_Error_string(ierr, &info_buf[0], &size);
            MPITEST_message(MPITEST_FATAL, info_buf);
            fail++;
        }
    }

    /* report overall results  */

    MPITEST_report(total_cnt - fail, fail, 0, testname);

    MPI_Finalize();
    /* 77 is a special return code for a skipped test. So we don't
     * want to return it */
    if(77 == fail) {
        fail++;
    }
    return fail;
}                               /* main() */
