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
                          Test for MPI_Type_lb()

All ranks will first do a MPI_Type_struct() of all types in the type config
array (in mpitest_cfg.h) with the minimum displacement (the first type) being the
length value obtained from the message length array (in mpitest_cfg.h).  All ranks
then use call MPI_Type_lb() using the newly created datatype and verify that
the returned displacement is equal to length.

This test may be run in any communicator with a minimum of 1 group members,
with any data type, and with any message length.

The MPITEST environment provides looping over communicator size,
message length.  The properties of the loops are encoded in configuration
arrays in the file mpitest_cfg.h .

MPI Calls dependencies for this test:
  MPI_Init(), MPI_Finalize(), MPI_Type_lb(),
  MPI_Comm_test_inter(), MPI_Error_string(),
  MPI_Type_struct(), MPI_Type_commit(),
  MPI_Type_size(), MPI_Type_free(),
  [MPI_Get_count(), MPI_Allreduce(), MPI_Comm_rank(), MPI_Comm_size()]

Test history:
   1  07/08/96     simont       Original version

******************************************************************************/

#include "mpitest_cfg.h"
#include "mpitest.h"

int main(int argc, char *argv[])
{
    int
     test_type,                 /*  the index of the current buffer type              */
     length_count,              /*  loop counter for length loop                      */
     length,                    /*  The length of the current buffer                  */
     test_nump,                 /*  The number of processors in current communicator  */
     comm_index,                /*  the array index of the current comm               */
     comm_type,                 /*  the index of the current communicator type        */
     type_count,                /*  loop counter for data type loop                   */
     comm_count,                /*  loop counter for communicator loop                */
     fail,                      /*  counts total number of failures                   */
     size,                      /*  return size from MPI_Error_string                 */
     loop_cnt,                  /*  counts total number of loops through test         */
     ierr;                      /*  return value from MPI calls                       */

    signed char
     info_buf[256],             /*  buffer for passing mesages to MPITEST             */
     testname[128];             /*  the name of this test                             */

    MPI_Comm comm;              /*  MPI communicator                                  */

    MPI_Datatype newtype, *types;

    int *blklens, num_types, *type_sizes;

    MPI_Aint *displs, displ;

    int inter_flag;

    ierr = MPI_Init(&argc, &argv);
    if (ierr != MPI_SUCCESS) {
        sprintf(info_buf, "MPI_Init() returned %d", ierr);
        MPITEST_message(MPITEST_FATAL, info_buf);
    }

    /* Open MPI: See https://svn.open-mpi.org/trac/ompi/ticket/3641 --
     * this test was ruled incorrect.  If someone wants to fix it,
     * great.  But for now, we're just going to mark it as
     * "skipped". */
    MPI_Finalize();
    exit(77);

    sprintf(testname, "MPI_Type_lb_pos_displ");

    MPITEST_init(argc, argv);

    if (MPITEST_me == 0) {
        sprintf(info_buf, "Starting %s test", testname);
        MPITEST_message(MPITEST_INFO0, info_buf);
    }

    /* set the global error counter */
    fail = 0;
    loop_cnt = 0;

    num_types = MPITEST_num_datatypes();

    if (num_types == 0)
        MPITEST_message(MPITEST_FATAL, "No basic data types configured");

    /* Set up various arrays */
    types = (MPI_Datatype *) calloc(num_types, sizeof(MPI_Datatype));
    if (!types) {
        sprintf(info_buf, "Cannot allocate enough memory for types array");
        MPITEST_message(MPITEST_FATAL, info_buf);
    }

    blklens = (int *) calloc(num_types, sizeof(int));
    if (!blklens) {
        sprintf(info_buf,
                "Cannot allocate enough memory for blklens array");
        MPITEST_message(MPITEST_FATAL, info_buf);
    }

    type_sizes = (int *) calloc(num_types, sizeof(int));
    if (!type_sizes) {
        sprintf(info_buf,
                "Cannot allocate enough memory for type_sizes array");
        MPITEST_message(MPITEST_FATAL, info_buf);
    }

    displs = (MPI_Aint *) calloc(num_types, sizeof(MPI_Aint));
    if (!displs) {
        sprintf(info_buf,
                "Cannot allocate enough memory for displs array");
        MPITEST_message(MPITEST_FATAL, info_buf);
    }

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

            for (length_count = 0;
                 length_count < MPITEST_num_message_lengths();
                 length_count++) {
                length = MPITEST_get_message_length(length_count);

                for (type_count = 0; type_count < num_types; type_count++) {
                    test_type = MPITEST_get_datatype(type_count);
                    types[type_count] = MPITEST_mpi_datatypes[test_type];

                    if (type_count == 0) {
                        displs[type_count] = length;
                    } else {
                        displs[type_count] =
                            displs[type_count - 1] +
                            type_sizes[type_count - 1];
                    }

                    blklens[type_count] = 1;

                    ierr =
                        MPI_Type_size(types[type_count],
                                      &(type_sizes[type_count]));
                    if (ierr != MPI_SUCCESS) {
                        sprintf(info_buf, "MPI_Type_size() returned %d",
                                ierr);
                        MPITEST_message(MPITEST_NONFATAL, info_buf);
                        MPI_Error_string(ierr, &info_buf[0], &size);
                        MPITEST_message(MPITEST_FATAL, info_buf);
                    }
                }

                ierr =
                    MPI_Type_struct(num_types, blklens, displs, types,
                                    &newtype);
                if (ierr != MPI_SUCCESS) {
                    sprintf(info_buf, "MPI_Type_struct() returned %d",
                            ierr);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    MPI_Error_string(ierr, &info_buf[0], &size);
                    MPITEST_message(MPITEST_FATAL, info_buf);
                }

                /* Committing newly created datatype */
                ierr = MPI_Type_commit(&newtype);
                if (ierr != MPI_SUCCESS) {
                    fail++;
                    sprintf(info_buf, "MPI_Type_commit() returned %d",
                            ierr);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    MPI_Error_string(ierr, &info_buf[0], &size);
                    MPITEST_message(MPITEST_FATAL, info_buf);
                }

                if (MPITEST_current_rank == 0) {
                    sprintf(info_buf,
                            "(%d, %d) length %d commsize %d commtype %d",
                            length_count, comm_count, length, test_nump,
                            comm_type);
                    MPITEST_message(MPITEST_INFO1, info_buf);

                    for (type_count = 0; type_count < num_types;
                         type_count++) {
                        sprintf(info_buf,
                                "blklens[%d] = %d, displs[%d] = %d, "
                                "types[%d] = %d, type_sizes[%d] = %d",
                                type_count, blklens[type_count],
                                type_count, displs[type_count], type_count,
                                types[type_count], type_count,
                                type_sizes[type_count]);
                        MPITEST_message(MPITEST_INFO2, info_buf);
                    }
                }

                loop_cnt++;

                /* Test MPI_Type_lb() for this datatype */
                ierr = MPI_Type_lb(newtype, &displ);
                if (ierr != MPI_SUCCESS) {
                    fail++;
                    sprintf(info_buf, "MPI_Type_lb() returned %d", ierr);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    MPI_Error_string(ierr, &info_buf[0], &size);
                    MPITEST_message(MPITEST_FATAL, info_buf);
                } else if (displ != length) {
                    fail++;
                    sprintf(info_buf,
                            "MPI_Type_lb() returned unexpected displacement  Expected: %d, Actual: %d",
                            length, displ);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                }

                /* Free newly created datatype */
                ierr = MPI_Type_free(&newtype);
                if (ierr != MPI_SUCCESS) {
                    fail++;
                    sprintf(info_buf, "MPI_Type_free() returned %d", ierr);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    MPI_Error_string(ierr, &info_buf[0], &size);
                    MPITEST_message(MPITEST_FATAL, info_buf);
                }
#ifdef MPITEST_SYNC
                ierr = MPI_Barrier(comm);
                if (ierr != MPI_SUCCESS) {
                    sprintf(info_buf, "MPI_Barrier() returned %d", ierr);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    MPI_Error_string(ierr, &info_buf[0], &size);
                    MPITEST_message(MPITEST_FATAL, info_buf);
                }
#endif
            }
        }

        MPITEST_free_communicator(comm_type, &comm);
    }
    free(types);
    free(blklens);
    free(displs);

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
