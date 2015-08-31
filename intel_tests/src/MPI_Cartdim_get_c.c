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

                          MPI_Cartdim_get

        Reference:  MPI Standard,  Chapter 6:  Process Topologies
                                   Section 6.5.4

 Uses only intra-communicators,  all inter-communicators are skipped
 Must have a minimum of MINNODE nodes in the communicator
 The test program is limited to testing not more than MAXNDIMS dimensions


******************************************************************************/

#define  MPITEST_MAXNDIMS    4  /* Maximum dimensions to generate            */
#define  MPITEST_MINNODE     6  /* Minimum number of nodes required for test */


#include "mpitest_cfg.h"
#include "mpitest.h"

int main(int argc, char *argv[])
{
    int
     i,                         /* General loop indices                              */
     comm_count,                /* loop counter for communicator loop                */
     comm_index,                /* the array index of the current comm               */
     comm_type,                 /* the index of the current communicator type        */
     error,                     /* errors from one MPI call                          */
     fail = 0,                  /* counts total number of failures                   */
     ierr,                      /* return value from MPI calls                       */
     inter_comm,                /* Intercommunicator flag, true if
                                 * intercommunicator */
     ndims,                     /* Number of dimensions to generate
                                 * coordinates for  */
     nnodes,                    /* node count for Dims_create                        */
     reorder,                   /* rank reorder flag                                 */
     size,                      /* return size from MPI_Error_string                 */
     test_nump,                 /* The number of processors in current
                                 * communicator  */
     cart_dims,                 /* dimensions returned by Cartdim_get                */
     cart_nump,                 /* number of ranks in Cartesian communicator         */
     rank_cart,                 /* rank in the Cartesian communicator                */
     world_nump,                /* number of processes in MPI_COMM_WORLD             */
     rank_world,                /* node rank in MPI_COMM_WORLD                       */
     test_val,                  /* communication test value                          */
     loop_cnt;                  /* test event count                                  */

    int
     periods[MPITEST_MAXNDIMS], /* periodic assignment flag vector   */
     dim_size[MPITEST_MAXNDIMS];        /* Cartesian dimension sizes         */

    char
     info_buf[256],             /* buffer for passing mesages to MPITEST             */
     testname[64];              /* the name of this test                             */

    MPI_Comm comm,              /* MPITEST communicator                              */
        comm_cart;              /* Cartesian communicator                            */

   /*------------------------------  MPI_Init  -----------------------------*/
    ierr = MPI_Init(&argc, &argv);
    if (ierr != MPI_SUCCESS) {
        sprintf(info_buf, "Non-zero return code (%d) from MPI_Init()",
                ierr);
        MPITEST_message(MPITEST_FATAL, info_buf);
    }

    sprintf(testname, "MPI_Cartdim_get:");

   /*-----------------------------  MPITEST_init  --------------------------*/
    MPITEST_init(argc, argv);

    if (MPITEST_me == 0) {
        sprintf(info_buf, "Starting %s test", testname);
        MPITEST_message(MPITEST_INFO0, info_buf);
    }

    /* Can't run if not at least  MPITEST_MINNODE  nodes  */

    MPI_Comm_size(MPI_COMM_WORLD, &world_nump);
    if (world_nump < MPITEST_MINNODE) {
        fail++;
        sprintf(info_buf,
                "WARNING --  nodes = %3d   Need %3d nodes to run test",
                world_nump, MPITEST_MINNODE);
        MPITEST_message(MPITEST_SKIP, info_buf);
    }

    /* set the global error counter */
    fail = 0;
    loop_cnt = 0;
    test_val = -1;
   /*----------------------  Loop over Communicators  ----------------------*/

    for (comm_count = 0; comm_count < MPITEST_num_comm_sizes();
         comm_count++) {
        comm_index = MPITEST_get_comm_index(comm_count);
        comm_type = MPITEST_get_comm_type(comm_count);
        test_nump = MPITEST_get_communicator(comm_type, comm_index, &comm);

        if (MPITEST_current_rank != MPI_UNDEFINED) {
            /* ...  inter_comm is true if  inter-communicator ... */

            ierr = MPI_Comm_test_inter(comm, &inter_comm);
            if (ierr != MPI_SUCCESS) {
                fail++;
                sprintf(info_buf,
                        "Non-zero return code (%d) from MPI_Comm_test_inter",
                        ierr);
                MPITEST_message(MPITEST_NONFATAL, info_buf);
                MPI_Error_string(ierr, &info_buf[0], &size);
                MPITEST_message(MPITEST_FATAL, info_buf);
            }

            if (!inter_comm && test_nump >= MPITEST_MINNODE) {

                /* Decide how many dimensions to generate  */

                ndims = MPITEST_MAXNDIMS;

                while (test_nump % ndims != 0)
                    ndims--;

                for (i = 0; i < ndims; i++) {
                    dim_size[i] = 0;
                }

            /*-------------------  MPI_Dims_create  ---------------------*/

                nnodes = test_nump;

                ierr = MPI_Dims_create(nnodes, ndims, dim_size);

                if (ierr != MPI_SUCCESS) {
                    fail++;
                    sprintf(info_buf, "MPI_Dims_create error (%d)", ierr);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    MPI_Error_string(ierr, &info_buf[0], &size);
                    MPITEST_message(MPITEST_FATAL, info_buf);
                }

                for (i = 0; i < ndims; i++) {
                    periods[i] = TRUE;
                }

            /*---------------------  MPI_Cart_create  -------------------*/
                reorder = TRUE;

                ierr = MPI_Cart_create(comm,
                                       ndims,
                                       dim_size,
                                       periods, reorder, &comm_cart);

                if (ierr != MPI_SUCCESS) {
                    sprintf(info_buf,
                            "Non-zero return code (%d) from MPI_Cart_create",
                            ierr);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    MPI_Error_string(ierr, &info_buf[0], &size);
                    MPITEST_message(MPITEST_FATAL, info_buf);
                    fail++;
                }
                /* Error Test  */
                MPI_Comm_rank(comm_cart, &rank_cart);

                if (rank_cart == 0)
                    test_val = 1;

                /* Broacast from node zero of Cartesian communicator */

                ierr = MPI_Bcast(&test_val, 1, MPI_INT, 0, comm_cart);
                if (ierr != MPI_SUCCESS) {
                    sprintf(info_buf,
                            "Non-zero return code (%d) from MPI_Bcast",
                            ierr);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    MPI_Error_string(ierr, &info_buf[0], &size);
                    MPITEST_message(MPITEST_FATAL, info_buf);
                    fail++;
                }


                /* Error Test  */
                /* nodes in communicator check for broadcast value */
                if (test_val != 1) {
                    sprintf(info_buf,
                            "Error in broadcast  value:  expected 1   actual = %d",
                            test_val);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    fail++;
                }

            /*--------------------------- MPI_Cartdim_get   ----------------------------*/

                loop_cnt++;

                ierr = MPI_Cartdim_get(comm_cart, &cart_dims);

                if (ierr != MPI_SUCCESS) {
                    fail++;
                    sprintf(info_buf, "MPI_Cartdim_get error (%d)", ierr);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    MPI_Error_string(ierr, &info_buf[0], &size);
                    MPITEST_message(MPITEST_FATAL, info_buf);
                }

                if (cart_dims != ndims) {
                    sprintf(info_buf,
                            "Error in Cartdim dimensions:  Actual = (%d)   Expected (%d)",
                            cart_dims, ndims);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                }

                MPI_Comm_free(&comm_cart);

            }
            /* end of Cartesian communicator code  */
            else {
                /* intra-communicator  or insufficient-nodes */

                if (MPITEST_current_rank == 0) {
                    if (inter_comm) {
                        sprintf(info_buf, "Skipped inter-communicator");
                        MPITEST_message(MPITEST_INFO1, info_buf);
                    } else {
                        sprintf(info_buf,
                                "Skipping:  Communicator smaller than minimum  %d/%d",
                                test_nump, MPITEST_MINNODE);
                        MPITEST_message(MPITEST_INFO1, info_buf);
                    }
                }

            }


        }
        /* node defined for this MPITEST communicator  */
#ifdef MPITEST_SYNC

        ierr = MPI_Barrier(MPI_COMM_WORLD);
        if (ierr != MPI_SUCCESS) {
            sprintf(info_buf, "Non-zero return code (%d) from MPI_Barrier",
                    ierr);
            MPITEST_message(MPITEST_NONFATAL, info_buf);
            MPI_Error_string(ierr, &info_buf[0], &size);
            MPITEST_message(MPITEST_FATAL, info_buf);
            fail++;
        }
#endif

        MPITEST_free_communicator(comm_type, &comm);

    }                           /* Loop over Communicators  */

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
