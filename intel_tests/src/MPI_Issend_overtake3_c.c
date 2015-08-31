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
		  Test for MPI_Issend() 

This is another test to make sure that MPI_Issend() works properly.

In this test, 2 ranks post an MPI_Issend to each other, then do an MPI_Recv.
This should make progress.

******************************************************************************/

#include "mpitest_cfg.h"
#include "mpitest.h"

int main(int argc, char *argv[])
{

    int
     loop_cnt,                  /* counts total number of times through loop     */
     fail,                      /* counts total number of failures               */
     ierr,                      /* return value from MPI calls                   */
     size,                      /* return size from MPI_Error_string             */
     i, flag;

    char buf1[3];               /* buffer for first message                       */
    char buf2[3];               /* buffer for second message                      */

    char
     info_buf[256],             /* buffer for passing mesages to MPITEST         */
     testname[64];              /* the name of this test                         */

    MPI_Status send_stat, recv_stat;

    MPI_Request request;
    /*-----------------------------  MPI_Init  ------------------------------*/
    ierr = MPI_Init(&argc, &argv);
    if (ierr != MPI_SUCCESS) {
        sprintf(info_buf, "Non-zero return code (%d) from MPI_Init()",
                ierr);
        MPITEST_message(MPITEST_FATAL, info_buf);
    }


    sprintf(testname, "MPI_Issend_overtake3");

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

    if (MPITEST_me < 2) {

        loop_cnt = 1;
        buf1[0] = buf1[1] = buf1[2] = MPITEST_me;
        buf2[0] = buf2[1] = buf2[2] = -1;

        ierr = MPI_Issend(&buf1,
                          3,
                          MPI_CHAR,
                          1 - MPITEST_me, 1, MPI_COMM_WORLD, &request);

        if (ierr != MPI_SUCCESS) {
            sprintf(info_buf, "Non-zero return code (%d) from MPI_Issend",
                    ierr);
            MPITEST_message(MPITEST_NONFATAL, info_buf);
            MPI_Error_string(ierr, &info_buf[0], &size);
            MPITEST_message(MPITEST_FATAL, info_buf);
            fail = 1;
        }

        /* Issend Error Test  */
        ierr = MPI_Recv(&buf2, 3, MPI_CHAR, 1 - MPITEST_me, 1,
                        MPI_COMM_WORLD, &recv_stat);
        if (ierr != MPI_SUCCESS) {
            sprintf(info_buf, "Non-zero return code (%d) from MPI_Recv",
                    ierr);
            MPITEST_message(MPITEST_NONFATAL, info_buf);
            MPI_Error_string(ierr, &info_buf[0], &size);
            MPITEST_message(MPITEST_FATAL, info_buf);
            fail = 1;
        }

        if ((buf2[0] != 1 - MPITEST_me) ||
            (buf2[1] != 1 - MPITEST_me) || (buf2[2] != 1 - MPITEST_me)) {
            sprintf(info_buf,
                    "Received data = %d,%d,%d, expected %d,%d,%d", buf1[0],
                    buf1[1], buf1[2], buf2[0], buf2[1], buf2[2]);
            MPITEST_message(MPITEST_NONFATAL, info_buf);
            fail = 1;
        }

        ierr = MPI_Wait(&request, &send_stat);

        if (ierr != MPI_SUCCESS) {
            sprintf(info_buf,
                    "Non-zero return code (%d) from MPI_Wait on Issend",
                    ierr);
            MPITEST_message(MPITEST_NONFATAL, info_buf);
            MPI_Error_string(ierr, &info_buf[0], &size);
            MPITEST_message(MPITEST_NONFATAL, info_buf);
            fail = 1;
        }
        /* Wait on Issend Error Test  */
    }

    /* report overall results  */

    MPITEST_report(loop_cnt - fail, fail, 0, testname);

    MPI_Finalize();

    /* 77 is a special return code for a skipped test. So we don't
     * want to return it */
    if(77 == fail) {
        fail++;
    }
    return fail;

}                               /* main() */
