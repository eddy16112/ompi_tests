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
			  Test for MPI_Group_difference

This test verifies that MPI_Group_difference works correctly with
itself, MPI_EMPTY_GROUP and the group of COMM_WORLD:

(group, group) => empty
(empty, group) => empty
(group, empty) => group
(group, world) => empty

Other cases are tested elsewhere.  This test is repeated for all configured
communicators.

Test history:
   1  08/28/96     gt       Original version

******************************************************************************/

#include "mpitest_cfg.h"
#include "mpitest.h"


int main(int argc, char *argv[])
{
    int
     fail,                      /* Counts number of test failures  */
     loop_cnt,                  /* Counts number of tests executed */
     verify,                    /* Counts number of tests to verify */
     ierr,                      /* Return value from MPI calls     */
     test_nump,                 /* number of ranks in current comm */
     comm_index,                /* array index of current comm     */
     comm_type,                 /* index of current comm type      */
     comm_count,                /* number of communicators to test */
     type_count,                /* loop counter for data type loop */
     result, size;

    MPI_Comm comm;              /* Communicator under test         */

    MPI_Group group, group2, groupworld;

    char
     info_buf[256],             /* buffer for passing mesages to MPITEST  */
     testname[64];              /* the name of this test           */

    /*-----------------------------  MPI_Init  ------------------------------*/
    ierr = MPI_Init(&argc, &argv);
    if (ierr != MPI_SUCCESS) {
        sprintf(info_buf, "Non-zero return code (%d) from MPI_Init()",
                ierr);
        MPITEST_message(MPITEST_FATAL, info_buf);
    }


    sprintf(testname, "MPI_Group_difference");

    /*-----------------------------  MPITEST_init  --------------------------*/
    MPITEST_init(argc, argv);
    if (MPITEST_me == 0) {
        sprintf(info_buf, "Starting %s test", testname);
        MPITEST_message(MPITEST_INFO0, info_buf);
    }

    /* set the global error counter */
    fail = 0;
    verify = 0;
    loop_cnt = 0;

    /* Loop through the configured communicators */
    for (comm_count = 0; comm_count < MPITEST_num_comm_sizes();
         comm_count++) {
        comm_index = MPITEST_get_comm_index(comm_count);
        comm_type = MPITEST_get_comm_type(comm_count);

        test_nump = MPITEST_get_communicator(comm_type, comm_index, &comm);

        /* Only test if this node is part of the current communicator */
        if (MPITEST_current_rank != MPI_UNDEFINED) {

            /* Get a group for the communicator */
            ierr = MPI_Comm_group(comm, &group);
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Comm_group (comm_index %d)",
                        ierr, comm_index);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_FATAL, info_buf);
                fail++;
            }                   /* Error Test  */
            loop_cnt++;




            /* difference with itself */
            ierr = MPI_Group_difference(group, group, &group2);
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Group_difference(group, group)  (comm_index %d)",
                        ierr, comm_index);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_FATAL, info_buf);
                fail++;
            }                   /* Error Test  */
            loop_cnt++;

            if (group2 != MPI_GROUP_EMPTY) {
                fail++;
                sprintf(info_buf,
                        "MPI_Group_difference(group, group) did not return MPI_GROUP_EMPTY (comm_index %d)",
                        comm_index);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
            }
            loop_cnt++;




            /* difference with GROUP_EMPTY */
            ierr = MPI_Group_difference(MPI_GROUP_EMPTY, group, &group2);
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Group_difference(EMPTY, group)  (comm_index %d)",
                        ierr, comm_index);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_FATAL, info_buf);
                fail++;
            }                   /* Error Test  */
            loop_cnt++;

            if (group2 != MPI_GROUP_EMPTY) {
                fail++;
                sprintf(info_buf,
                        "MPI_Group_difference(EMPTY, group) did not return MPI_GROUP_EMPTY (comm_index %d)",
                        comm_index);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
            }
            loop_cnt++;



            /* reverse difference with GROUP_EMPTY */
            ierr = MPI_Group_difference(group, MPI_GROUP_EMPTY, &group2);
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Group_difference(group, EMPTY)  (comm_index %d)",
                        ierr, comm_index);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_FATAL, info_buf);
                fail++;
            }                   /* Error Test  */
            loop_cnt++;

            ierr = MPI_Group_compare(group, group2, &result);
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Group_compare(group, EMPTY)  (comm_index %d)",
                        ierr, comm_index);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_FATAL, info_buf);
                fail++;
            }                   /* Error Test  */
            loop_cnt++;
            if (result != MPI_IDENT) {
                fail++;
                sprintf(info_buf,
                        "MPI_Group_difference(group, EMPTY) did not return identical group (comm_index %d)",
                        comm_index);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
            }

            ierr = MPI_Group_free(&group2);
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Group_free(group, EMPTY)  (comm_index %d)",
                        ierr, comm_index);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_FATAL, info_buf);
                fail++;
            }                   /* Error Test  */
            loop_cnt++;



            /* Get a group for the MPI_COMM_WORLD */
            ierr = MPI_Comm_group(comm, &groupworld);
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Comm_group(world) (comm_index %d)",
                        ierr, comm_index);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_FATAL, info_buf);
                fail++;
            }                   /* Error Test  */
            loop_cnt++;

            /* difference with groupworld */
            ierr = MPI_Group_difference(group, groupworld, &group2);
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Group_difference(group, world)  (comm_index %d)",
                        ierr, comm_index);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_FATAL, info_buf);
                fail++;
            }                   /* Error Test  */
            loop_cnt++;

            if (group2 != MPI_GROUP_EMPTY) {
                fail++;
                sprintf(info_buf,
                        "MPI_Group_difference(group, world) did not return MPI_GROUP_EMPTY (comm_index %d)",
                        comm_index);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
            }
            loop_cnt++;



            /* Free the groups */
            ierr = MPI_Group_free(&groupworld);
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Group_free(groupworld) (comm_index %d)",
                        ierr, comm_index);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_FATAL, info_buf);
                fail++;
            }                   /* Error Test  */
            loop_cnt++;


            ierr = MPI_Group_free(&group);
            if (ierr != MPI_SUCCESS) {
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Group_free(group) (comm_index %d)",
                        ierr, comm_index);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_FATAL, info_buf);
                fail++;
            }                   /* Error Test  */
            loop_cnt++;



            MPITEST_free_communicator(comm_type, &comm);

        }
        /* Node is in this communicator */
    }                           /* Communicator loop */


    /* report overall results  */

    MPITEST_report(loop_cnt - fail, fail, verify, testname);

    MPI_Finalize();

    /* 77 is a special return code for a skipped test. So we don't 
     * want to return it */
    if(77 == fail) {
        fail++;
    }
    return fail;

}                               /* main() */
