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
		  Test for MPI_Win_get_name()

This test calls MPI_Win_get_name() on each rank and prints it.
******************************************************************************/

#include "mpitest_cfg.h"
#include "mpitest.h"

int main(int argc, char *argv[])
{
    int
     fail,                      /* Counts number of test failures  */
     loop_cnt,                  /* Counts number of tests executed */
     ierr,                      /* Return value from MPI calls     */
     len,                       /* Length of String                */
     i, size;

    char
     name[MPI_MAX_OBJECT_NAME], info_buf[256],       /* buffer for passing mesages to MPITEST         */
     testname[64];              /* the name of this test                         */
    char window_name[] = "Test window name";
    MPI_Win win;

    /*-----------------------------  MPI_Init  ------------------------------*/
    ierr = MPI_Init(&argc, &argv);
    if (ierr != MPI_SUCCESS) {
        sprintf(info_buf, "Non-zero return code (%d) from MPI_Init()",
                ierr);
        MPITEST_message(MPITEST_FATAL, info_buf);
    }


    sprintf(testname, "MPI_Win_get_name");

    /*-----------------------------  MPITEST_init  --------------------------*/
    MPITEST_init(argc, argv);
    if (MPITEST_me == 0) {
        sprintf(info_buf, "Starting %s test", testname);
        MPITEST_message(MPITEST_INFO0, info_buf);
    }

    /* set the global error counter */
    fail = 0;
    loop_cnt = 1;

    /* There are no pre-defined windows, so make one */
    ierr = MPI_Win_create(info_buf, sizeof(info_buf), 1,
                          MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    for (i = 0; i < MPITEST_nump; i++) {
        if (i == MPITEST_me) {
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Win_create",
                        ierr);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail++;
            } /* Error Test  */
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /* Set a name on the window */
    ++loop_cnt;
    ierr = MPI_Win_set_name(win, window_name);
    for (i = 0; i < MPITEST_nump; i++) {
        if (i == MPITEST_me) {
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Win_set_name",
                        ierr);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail++;
            } /* Error Test  */
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /* Now get the name and check it */
    ++loop_cnt;
    ierr = MPI_Win_get_name(win, name, &len);
    for (i = 0; i < MPITEST_nump; i++) {
        if (i == MPITEST_me) {
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Win_get_name",
                        ierr);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail++;
            } /* Error Test  */
            else if (0 != strcmp(name, window_name)) {
                sprintf(info_buf,
                        "Got wrong window name back (\"%s\", expected \"%s\")",
                        name, window_name);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                fail++;
            } /* Error Test */
            else {
                if ((len > MPI_MAX_OBJECT_NAME)
                    || (len != strlen(name))) {
                    fail++;
                    sprintf(info_buf,
                            "Returned length=%d, MPI_MAX_OBJECT_NAME=%d",
                            len, MPI_MAX_OBJECT_NAME);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    MPITEST_message(MPITEST_INFO0, name);
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /* report overall results  */

    MPITEST_report(loop_cnt - fail, fail, 0, testname);
    MPI_Win_free(&win);

    MPI_Finalize();
    return fail;
}
