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
                          Test for MPI_Type_size()

All rank call MPI_Type_size() with MPI_LB and MPI_UB.  The resulting size
will be verified (should be 0).

This test may be run in any communicator with a minimum of 1 group members.

The MPITEST environment provides looping over communicator size.
The properties of the loops are encoded in configuration
arrays in the file mpitest_cfg.h .

MPI Calls dependencies for this test:
  MPI_Init(), MPI_Finalize(), MPI_Comm_test_inter(), MPI_Error_string(),
  MPI_Type_size(), 
  [MPI_Get_count(), MPI_Allreduce(), MPI_Comm_rank(), MPI_Comm_size()]

Test history:
   1  07/08/96     simont       Original version

******************************************************************************/

#include "mpitest_cfg.h"
#include "mpitest.h"

/* Minimum displacement */
#define MPITEST_MIN_DISPL     100

int main(int argc, char *argv[])
{
    int
     test_nump,                 /*  The number of processors in current communicator  */
     comm_index,                /*  the array index of the current comm               */
     comm_type,                 /*  the index of the current communicator type        */
     comm_count,                /*  loop counter for communicator loop                */
     fail,                      /*  counts total number of failures                   */
     size,                      /*  return size from MPI_Error_string                 */
     loop_cnt,                  /*  counts total number of loops through test         */
     ierr;                      /*  return value from MPI calls                       */

    signed char
     info_buf[256],             /*  buffer for passing mesages to MPITEST             */
     testname[128];             /*  the name of this test                             */

    MPI_Comm comm;              /*  MPI communicator                                  */

    int
     type_size;                 /*  size returned from MPI_Type_size                  */

    int inter_flag;

    ierr = MPI_Init(&argc, &argv);
    if (ierr != MPI_SUCCESS) {
        sprintf(info_buf, "MPI_Init() returned %d", ierr);
        MPITEST_message(MPITEST_FATAL, info_buf);
    }

    sprintf(testname, "MPI_Type_size_MPI_LB_UB");

    MPITEST_init(argc, argv);

    if (MPITEST_me == 0) {
        sprintf(info_buf, "Starting %s test", testname);
        MPITEST_message(MPITEST_INFO0, info_buf);
    }

    /* set the global error counter */
    fail = 0;
    loop_cnt = 0;

    for (comm_count = 0; comm_count < MPITEST_num_comm_sizes();
         comm_count++) {
        comm_index = MPITEST_get_comm_index(comm_count);
        comm_type = MPITEST_get_comm_type(comm_count);

        test_nump = MPITEST_get_communicator(comm_type, comm_index, &comm);

        if (comm != MPI_COMM_NULL) {
            if (test_nump < 1) {
                /* Skipping communicator with comm size < 1 */
                MPITEST_free_communicator(comm_type, &comm);
                sprintf(info_buf,
                        "Skipping communicator with comm_size < 1 (commtype: %d) for this test",
                        comm_type);
                MPITEST_message(MPITEST_INFO1, info_buf);
                continue;
            }

            ierr = MPI_Comm_test_inter(comm, &inter_flag);
            if (ierr != MPI_SUCCESS) {
                fail++;
                sprintf(info_buf, "MPI_Comm_test_inter() returned %d",
                        ierr);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_FATAL, info_buf);
            }

            if (inter_flag) {
                /* Ignore inter-communicator for test */
                sprintf(info_buf,
                        "Skipping inter communicator (commtype: %d) for this test",
                        comm_type);
                MPITEST_message(MPITEST_INFO1, info_buf);
                continue;
            }

            loop_cnt++;

            /* MPI_LB */
            sprintf(info_buf, "Calling MPI_Type_size() with MPI_LB");
            MPITEST_message(MPITEST_INFO1, info_buf);

            ierr = MPI_Type_size(MPI_LB, &type_size);
            if (ierr != MPI_SUCCESS) {
                fail++;
                sprintf(info_buf, "MPI_Type_size() returned %d", ierr);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_FATAL, info_buf);
            }

            if (type_size != 0) {
                fail++;
                sprintf(info_buf,
                        "MPI_Type_size() w/ MPI_LB returned unexpected %d Expected: %d, Actual: %d",
                        0, type_size);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
            }

            /* MPI_UB */
            sprintf(info_buf, "Calling MPI_Type_size() with MPI_UB");
            MPITEST_message(MPITEST_INFO1, info_buf);

            ierr = MPI_Type_size(MPI_UB, &type_size);
            if (ierr != MPI_SUCCESS) {
                fail++;
                sprintf(info_buf, "MPI_Type_size() returned %d", ierr);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_FATAL, info_buf);
            }

            if (type_size != 0) {
                fail++;
                sprintf(info_buf,
                        "MPI_Type_size() w/ MPI_UB returned unexpected %d Expected: %d, Actual: %d",
                        0, type_size);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
            }
        }

        MPITEST_free_communicator(comm_type, &comm);
    }

    /* report overall results  */
    MPITEST_report(loop_cnt - fail, fail, 0, testname);

    ierr = MPI_Finalize();
    if (ierr != MPI_SUCCESS) {
        fail++;
        sprintf(info_buf, "MPI_Finalize() returned %d, FAILED", ierr);
        MPITEST_message(MPITEST_NONFATAL, info_buf);
        MPI_Error_string(ierr, &info_buf[0], &size);
        MPITEST_message(MPITEST_FATAL, info_buf);
    }

    /* 77 is a special return code for a skipped test. So we don't
     * want to return it */
    if(77 == fail) {
        fail++;
    }
    return fail;
}                               /* main() */
