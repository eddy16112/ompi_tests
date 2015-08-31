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
                          MPI_Graph_map

 Reference:  MPI Standard,  Chapter 6:  Process Topologies
                            Section 6.5.7:  Low-level topology functions

 Uses only intra-communicators,  all inter-communicators are skipped
 Must have a minimum of MINNODE nodes in the communicator
 The test program is limited to testing not more than MAXNDIMS dimensions


******************************************************************************/

#define  MPITEST_MAXNDIMS    4  /* Maximum dimensions to generate            */
#define  MPITEST_MINNODE     6  /* Minimum number of nodes required for test */
#define  MPITEST_MAXN        8  /* maximum number of neighbors for each node */


#include "mpitest_cfg.h"
#include "mpitest.h"

int main(int argc, char *argv[])
{
    int
     i, j, k,                   /* General loop indices                              */
     comm_count,                /* loop counter for communicator loop                */
     comm_index,                /* the array index of the current comm               */
     comm_type,                 /* the index of the current communicator type        */
     error,                     /* errors from one MPI call                          */
     fail,                      /* counts total number of failures                   */
     ierr,                      /* return value from MPI calls                       */
     inter_comm,                /* Intercommunicator flag, true if intercommunicator */
     ndims,                     /* Number of dimensions to generate coordinates for  */
     nnodes,                    /* node count for Dims_create                        */
     reorder,                   /* rank reorder flag                                 */
     size,                      /* return size from MPI_Error_string                 */
     test_nump,                 /* The number of processors in current communicator  */
     graph_nump,                /* number of ranks in graph communicator             */
     rank_graph,                /* rank in the graph communicator                    */
     world_nump,                /* number of processes in MPI_COMM_WORLD             */
     test_val,                  /* communication test value                          */
     newrank,                   /* rank returned by MPI_Graph_map                    */
     loop_cnt;                  /* test event count                                  */

    int
     index[MPITEST_MAX_RANKS],  /* Cumulative neighbor count    */
     edges[MPITEST_MAX_RANKS * MPITEST_MAXN];   /* Node connections list        */

    int
     ranks[2],                  /* old and new ranks for a node      */
     all_ranks[MPITEST_MAX_RANKS][2];   /* old and new ranks for all nodes   */

    char
     info_buf[256],             /* buffer for passing mesages to MPITEST             */
     testname[64];              /* the name of this test                             */

    MPI_Comm comm,              /* MPITEST communicator                              */
        comm_graph;             /* graph communicator                                */

    /*------------------------------  MPI_Init  -----------------------------*/
    ierr = MPI_Init(&argc, &argv);
    if (ierr != MPI_SUCCESS) {
        sprintf(info_buf, "Non-zero return code (%d) from MPI_Init()",
                ierr);
        MPITEST_message(MPITEST_FATAL, info_buf);
    }

    sprintf(testname, "MPI_Graph_map");

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
         comm_count++)
    {
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

/*----------------------  Set up index and edge arrays  ---------------------*/

                /* assume a single string of nodes with wraparound  */

                /*  Set up edges for node 0  */
                edges[0] = test_nump - 1;
                edges[1] = 1;

                j = 2;
                for (i = 1; i < test_nump - 1; i++) {
                    edges[j] = i - 1;
                    edges[j + 1] = i + 1;
                    j = j + 2;
                }

                /* Set up edges for last node  */

                edges[2 * test_nump - 2] = test_nump - 2;
                edges[2 * test_nump - 1] = 0;

                for (i = 0; i < test_nump; i++) {
                    index[i] = (i + 1) * 2;
                }


/*-----------------------------  MPI_Graph_create  --------------------------*/

                nnodes = test_nump;
                reorder = TRUE;

                ierr = MPI_Graph_create(comm,
                                        nnodes,
                                        index,
                                        edges, reorder, &comm_graph);

                if (ierr != MPI_SUCCESS) {
                    fail++;
                    sprintf(info_buf, "MPI_Graph_create error (%d) ",
                            ierr);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    MPI_Error_string(ierr, &info_buf[0], &size);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                }


                MPI_Comm_size(comm_graph, &graph_nump);
                MPI_Comm_rank(comm_graph, &rank_graph);

                if (rank_graph == 0)
                    test_val = 1;

                /* Broacast from node zero of Graph communicator */

                ierr = MPI_Bcast(&test_val, 1, MPI_INT, 0, comm_graph);
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


/*------------------------- MPI_Graph_map   ----------------------------*/


                loop_cnt++;

                ierr = MPI_Graph_map(comm_graph,
                                     nnodes, index, edges, &newrank);

                if (ierr != MPI_SUCCESS) {
                    fail++;
                    sprintf(info_buf, "MPI_Graph_map error (%d)", ierr);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    MPI_Error_string(ierr, &info_buf[0], &size);
                    MPITEST_message(MPITEST_FATAL, info_buf);
                }


                /* Send graph rank and newrank to graph node 0  */
                ranks[0] = rank_graph;
                ranks[1] = newrank;

                ierr =
                    MPI_Gather(ranks, 2, MPI_INT, all_ranks, 2, MPI_INT, 0,
                               comm_graph);

                if (ierr != MPI_SUCCESS) {
                    fail++;
                    sprintf(info_buf, "MPI_Gather error (%d)", ierr);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    MPI_Error_string(ierr, &info_buf[0], &size);
                    MPITEST_message(MPITEST_FATAL, info_buf);
                }
                /* Now, graph node 0 can compare all the ranks  */
                if (rank_graph == 0) {
                    /* Make sure there are graph_nump distinct ranks in the reorder  */
                    for (j = 0; j < graph_nump; j++) {
                        k = 0;
                        while (j != all_ranks[k][1] && k < graph_nump)
                            k++;
                        if (k >= graph_nump) {

                            sprintf(info_buf,
                                    "ERROR:  Missing rank in new mapping  loc = %d  Old/New = (%d/%d) ",
                                    j, all_ranks[j][0], all_ranks[j][1]);
                            MPITEST_message(MPITEST_FATAL, info_buf);
                        }

                    }

                }

                /* End of rank_graph == 0 code  */
                MPI_Comm_free(&comm_graph);

            } /* End of code for graph communicator  */
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
        /* node defined for this communicator  */
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
