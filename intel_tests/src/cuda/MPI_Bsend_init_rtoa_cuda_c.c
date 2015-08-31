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

       Functional test for MPI persistent I/O  Requests
                      with MPI_Bsend_init():  Root send to all Others
                           MPI_Start to send/receive requests
                           MPI_Test  to check completion
                           MPI_Request_free to free send/recv requests
                      and  MPI_Recv_init():  Any source/Any tag


MPI_Standard:  Section 3.9  Persistent communication requests
               also Sections:  3.7.3, 3.7.5, 3.8

USING MPI:     Section 5.2.2

Persistent communication requests may optimize throughput for repeated calls
to a communication function with the same arguments.  The persistent
communication calls replace the communiction request with two calls, a
communication initialization which is called once for each set of parameters
and an MPI_Start, or MPI_Startall to initiate each communication.
The initialization process does not create a full channel from sender to
receiver, only from the controller to the process. Persistent communications
offers four different initialization calls, matching the four types of
nonblocking I/O, i.e. MPI_Send_init, MPI_Bsend_init, MPI_Rsend_init,
MPI_Ssend_init.

For an INTRA-communicator, the program selects each node, in turn,
of the communicator to be the root.  the ROOT sends to all of the nodes,
including itself

For an INTER-communicator, there are two groups of nodes; each group being
termed a sub-communicator.  The nodes in each sub-communicator are numbered
0 to (n-1) for an n-node sub-communicator.  So,the MPITEST_current_rank will
return duplicate node numbers for nodes (0 to k-1) where k is the number of
nodes in the smaller communicator.

The program cycles through the nodes in one sub-communicator, sending from each
selected node to  all of the nodes in the other sub-communicator.  Then
the program reverses the send and receive role of the two sub-communicators,
and repeats the process.
	
This test initializes the send buffer with the root's rank in the
communicator (or an appropriate value for the non-integer types.)
Once the receiving nodes have completed their message receive, they check to
make sure the current root's rank is in the received buffer.

This test may be run in any communicator, with any data type, and with
any non-negative message length.

The MPITEST environment provides looping over communicator size and
type, message length, and data type.  The properties of the loops are
encoded in configuration arrays in the file mpitest_cfg.h .  See the
MPITEST README for further details.

******************************************************************************/

#include "mpitest_cfg.h"
#include "mpitest.h"
#include "cudatest.h"


int main(int argc, char *argv[])
{

    int
     flag,                      /* Test call flag                                    */
     buffsize,                  /* Size of malloc buffer for Ibsend                  */
     berr,                      /* error flag for buffer ATTACH and DETACH           */
     byte_length,               /* The length of the current buffer in bytes         */
     cnt_len,                   /* received length returned by MPI_Get_Count         */
     comm_count,                /* loop counter for communicator loop                */
     comm_index,                /* the array index of the current comm               */
     comm_type,                 /* the index of the current communicator type        */
     error,                     /* errors from one MPI call                          */
     fail,                      /* counts total number of failures                   */
     loop_fail,                 /* counts number of failures in loop                 */
     grp_lup,                   /* index for comm groups (1 to ntimes)               */
     i,                         /* utility loop index variables                      */
     ierr,                      /* return value from MPI calls                       */
     inter_comm,                /* flag, true, if intercommunicator                  */
     left_recv_size,            /* Left Intercommunictor receive size                */
     left_send_size,            /* Left Intercommunicator send size                  */
     length,                    /* The length of the current buffer                  */
     length_count,              /* loop counter for message length loop              */
     loop_cnt,                  /* counts total number of loops through test         */
     max_byte_length,           /* maximum buffer length in bytes                    */
     max_length,                /* max buffer length specified in config. file       */
     ntimes,                    /* # of communicator groups INTRA=1, INTER=2         */
     print_node_flag,           /* node 0 of INTRA, node 0 of left comm of INTER     */
     receivers,                 /* INTER-comm #receivers for this send/rec choice    */
     recv_group,                /* INTER-comm test receiving group(0 or 1)           */
     recv_size,                 /* INTER  receive size MPI_Comm_remote_size          */
     right_recv_size,           /* INTER: absolute count of right receive nodes      */
     right_send_size,           /* INTER: count of right sending nodes               */
     root,                      /* the root of the current broadcast                 */
     send_group,                /* INTER-comm test sending group #(0 or 1)           */
     send_size,                 /* INTER: relative send size from MPI_Comm_size      */
     send_to,                   /* Number of nodes  receiving current message        */
     senders,                   /* INTER-comm #senders for this send/rec choice      */
     size,                      /* return size from MPI_Error_string                 */
     test_nump,                 /* The number of nodes in current communicator       */
     test_type,                 /* the index of the current buffer type              */
     type_count;                /* loop counter for data type loop                   */


    struct dataTemplate
     value;                     /* dataTemplate for initializing buffers             */

    void *recv_buffer;
    void *send_buffer;          /* message buffer                                    */
    CUdeviceptr gpu_recv_buffer;
    CUdeviceptr gpu_send_buffer;
    size_t gpu_recv_buffer_size;
    size_t gpu_send_buffer_size;

    char
     info_buf[256],             /* buffer for passing mesages to MPITEST             */
     testname[128];             /* the name of this test                             */

    MPI_Comm comm,              /* MPI communicator                                  */
        barrier_comm;           /* If intercommunicator, merged comm for Barrier     */


    MPI_Status send_stat, recv_stat;    /* MPI  status structure                             */

    MPI_Request recv_request,   /* MPI request structure                             */
        send_request;           /* MPI request structure                             */


    char *buff;
    int bsize;

    /*------------------------------  MPI_Init  -----------------------------*/
    cnt_len = -1;

    ierr = MPI_Init(&argc, &argv);
    if (ierr != MPI_SUCCESS) {
        sprintf(info_buf, "Non-zero return code (%d) from MPI_Init()",
                ierr);
        MPITEST_message(MPITEST_FATAL, info_buf);
    }

    sprintf(testname,
            "MPI_Bsend_init:  Persistent root buffered send to all others");
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

    /* set the global error counter */
    fail = 0;
    loop_cnt = 0;

    max_byte_length = MPITEST_get_max_message_length();



    /*--------------------- Allocate buffer for Ibsend ----------------------*/

    buffsize =
        MPITEST_nump * (MPI_BSEND_OVERHEAD + max_byte_length + 1000);
    buff = malloc(buffsize * sizeof(char));
    if (!buff) {
        sprintf(info_buf, "Ibsend malloc request failed");
        MPITEST_message(MPITEST_FATAL, info_buf);
    }
    berr = MPI_Buffer_attach(buff, buffsize);
    if (berr != MPI_SUCCESS) {
        sprintf(info_buf,
                "Non-zero return code (%d) from MPI_Buffer_attach", berr);
        MPITEST_message(MPITEST_FATAL, info_buf);
    }

    /* Detach, and get buffer pointer, and size for future attaches */

    berr = MPI_Buffer_detach(&buff, &bsize);

    if (berr != MPI_SUCCESS) {
        sprintf(info_buf,
                "Non-zero return code (%d) from MPI_Buffer_detach", berr);
        MPITEST_message(MPITEST_NONFATAL, info_buf);
        MPI_Error_string(berr, &info_buf[0], &size);
        MPITEST_message(MPITEST_NONFATAL, info_buf);
        fail++;

    }


    /* Buffer detach error Test  */
 /*----------------------  Loop over Communicators  ----------------------*/
    for (comm_count = 0; comm_count < MPITEST_num_comm_sizes();
         comm_count++) {
        comm_index = MPITEST_get_comm_index(comm_count);
        comm_type = MPITEST_get_comm_type(comm_count);
        /*
         * Reset a bunch of variables that will be set when we get our
         * communicator. These are mainly variables for determining
         * sub-communicator, left or right, when we get an
         * INTER-communicator.
         */

        left_send_size = -1;
        left_recv_size = -1;
        right_send_size = -1;
        right_recv_size = -1;
        send_size = -1;
        recv_size = -1;

        inter_comm = -1;
        print_node_flag = 0;

        test_nump = MPITEST_get_communicator(comm_type, comm_index, &comm);

        /* Skip everything if not a member of this communicator */
        if (MPITEST_current_rank != MPI_UNDEFINED) {
            /*
             * Test if INTER-communicator:  inter_comm is true if this is an
             * inter-communicator
             */

            MPI_Comm_test_inter(comm, &inter_comm);
            /*
             * Variables, send_size, and recv_size  both default to the total
             * nodes in the communicator for an INTRA-communicator. For an
             * INTER-communicator: MPI_Comm_size returns the size of the
             * communicator for all of the nodes in each respective
             * communicator MPI_Comm_remote_size returns the size of the
             * opposite communicator for all nodes in each respective
             * communicator
             * 
             * Set the print_node_flag: This is the flag that tells which node
             * gets to print.  For INTRA-communicators it is simply node 0.
             * For INTER-communication it is node zero of the left group.
             */

            if (inter_comm) {   /* Do this stuff for INTER-communicators */

                if (MPITEST_inter == 0) {       /* Do for left group of INTER-communicator */
                    MPI_Comm_size(comm, &send_size);
                    MPI_Comm_remote_size(comm, &recv_size);

                    left_send_size = send_size;
                    left_recv_size = recv_size;
                    right_send_size = recv_size;
                    right_recv_size = send_size;

                    if (MPITEST_current_rank == 0) {    /* Do for node zero of left bank of
                                                         * INTER-communicator */
                        print_node_flag = 1;
                    }
                    /* End of INTER-communicator left group node zero */
                    ierr = MPI_Intercomm_merge(comm, FALSE, &barrier_comm);
                    if (ierr != MPI_SUCCESS) {
                        sprintf(info_buf,
                                "Non-zero return code (%d) from MPI_Intercomm_merge",
                                ierr);
                        MPITEST_message(MPITEST_NONFATAL, info_buf);
                        MPI_Error_string(ierr, &info_buf[0], &size);
                        MPITEST_message(MPITEST_FATAL, info_buf);
                    }

                }
                /* End of      INTER-communicator left group */
                if (MPITEST_inter == 1) {       /* Do this for right group of an INTER-communicator  */
                    MPI_Comm_size(comm, &send_size);
                    MPI_Comm_remote_size(comm, &recv_size);

                    left_send_size = recv_size;
                    left_recv_size = send_size;
                    right_send_size = send_size;
                    right_recv_size = recv_size;

                    ierr = MPI_Intercomm_merge(comm, FALSE, &barrier_comm);
                    if (ierr != MPI_SUCCESS) {
                        sprintf(info_buf,
                                "Non-zero return code (%d) from MPI_Intercomm_merge",
                                ierr);
                        MPITEST_message(MPITEST_NONFATAL, info_buf);
                        MPI_Error_string(ierr, &info_buf[0], &size);
                        MPITEST_message(MPITEST_FATAL, info_buf);
                    }
                }               /* End of    INTER_communicator right group */
            }

            /* End of        INTER_communicator  */
            /* Do stuff for INTRA-communicator  */
            if (!inter_comm) {
                if (MPITEST_current_rank == 0)
                    print_node_flag = 1;
                barrier_comm = comm;
            }

            /*----------------  Loop over Data Types  -----------------------*/

            for (type_count = 0; type_count < MPITEST_num_datatypes();
                 type_count++) {
                test_type = MPITEST_get_datatype(type_count);

                /* convert the number of bytes in the maximum length message */
                /* into the number of elements of the current type */

                max_length =
                    MPITEST_byte_to_element(test_type, max_byte_length);

                /* Allocate send and receive Buffers */
                MPITEST_get_buffer(test_type, max_length, &recv_buffer);
                MPITEST_get_buffer(test_type, max_length, &send_buffer);
                CUDATEST_get_gpu_buffer(test_type, max_length, &gpu_recv_buffer, &gpu_recv_buffer_size);
                CUDATEST_get_gpu_buffer(test_type, max_length, &gpu_send_buffer, &gpu_send_buffer_size);

                /*-------------  Loop over Message Lengths  -----------------*/

                for (length_count = 0;
                     length_count < MPITEST_num_message_lengths();
                     length_count++) {
                    byte_length = MPITEST_get_message_length(length_count);

                    length =
                        MPITEST_byte_to_element(test_type, byte_length);

                    /*--------------------  Loop over roots  ----------------*/
                    /*
                     * For an intra-communicator we loop over the roots, and
                     * each root sends to all of the nodes of the
                     * communicator. For an inter-communicator we have two
                     * sets of nodes, a left and right sub-communicator, and
                     * we want to loop through all combinations. When a node
                     * is selected in a sub-communicator, it sends to all
                     * nodes in the other communicator.
                     */

                    i = -1;
                    /*
                     * Extra loop for INTER-communicators to have each group
                     * send and receive
                     */
                    ntimes = 1;
                    if (inter_comm)
                        ntimes = 2;
                    /*
                     * grp_lup = 1  send from left  sub-communicator to right
                     * sub-communicator
                     */
                    /*
                     * grp_lup = 2  send from right sub-communicator to left
                     * sub-communicator
                     */

                    for (grp_lup = 1; grp_lup <= ntimes; grp_lup++) {
                        send_group = grp_lup - 1;
                        recv_group = 2 - grp_lup;
                        /*
                         * Set up the number of senders and  receivers
                         * depending on sub-group of INTER-communicator
                         */

                        senders = test_nump;
                        receivers = test_nump - 1;
                        if (inter_comm) {
                            receivers = left_recv_size;
                            senders = left_send_size;
                        }
                        if (inter_comm && grp_lup == 2) {
                            receivers = right_recv_size;
                            senders = right_send_size;
                        }

                        for (root = 0; root < senders; root++) {
                            /* print an informational message */
                            sprintf(info_buf,
                                    "(%d,%d,%d) length %d commsize %d commtype %d data_type %d root %d",
                                    length_count, comm_count, type_count,
                                    length, test_nump, comm_type,
                                    test_type, root);
                            if (print_node_flag)
                                MPITEST_message(MPITEST_INFO1, info_buf);

                            /*-------------------------------------------------
			                                   Send
			     INTRA and INTER:  Root node in send group sends to all nodes of receive group
			    -------------------------------------------------*/

                            /* Send from each root node to all  nodes   */

                            if ((!inter_comm
                                 && MPITEST_current_rank == root)
                                || (inter_comm
                                    && MPITEST_inter == send_group
                                    && MPITEST_current_rank == root))
                            {
                                /*
                                 * Reattach the  send buffer for this root,
                                 * and detach it before switching to new root
                                 */

                                berr = MPI_Buffer_attach(buff, buffsize);
                                if (berr != MPI_SUCCESS) {
                                    sprintf(info_buf,
                                            "Non-zero return code (%d) from MPI_Buffer_attach",
                                            berr);
                                    MPITEST_message(MPITEST_FATAL,
                                                    info_buf);
                                }
                                /*
                                 * Set up the dataTemplate for initializing
                                 * send buffer
                                 */

                                MPITEST_dataTemplate_init(&value,
                                                          MPITEST_current_rank);

                                /* Initialize send buffer */
                                MPITEST_init_buffer(test_type, length + 1,
                                                    value, send_buffer);
                                /* Initialize entire GPU send buffer to match host send buffer */
                                EXIT_ON_ERROR(cuMemcpy(gpu_send_buffer, (CUdeviceptr)send_buffer,
                                                        gpu_send_buffer_size));

                                send_to = test_nump;
                                if (inter_comm)
                                    send_to = receivers;
                                for (i = 0; i < send_to; i++) {
                                    loop_cnt++;
                                    loop_fail = 0;

                                    ierr = MPI_Bsend_init((void *)gpu_send_buffer,
                                                          length,
                                                          MPITEST_mpi_datatypes
                                                          [test_type], i,
                                                          root, comm,
                                                          &send_request);

                                    if (ierr != MPI_SUCCESS) {
                                        sprintf(info_buf,
                                                "Non-zero return code (%d) from MPI_Bsend_init",
                                                ierr);
                                        MPITEST_message(MPITEST_NONFATAL,
                                                        info_buf);
                                        MPI_Error_string(ierr,
                                                         &info_buf[0],
                                                         &size);
                                        MPITEST_message(MPITEST_NONFATAL,
                                                        info_buf);
                                        loop_fail++;
                                    }

                                    /* Ibsend Error Test  */
                                    ierr = MPI_Start(&send_request);

                                    if (ierr != MPI_SUCCESS) {
                                        sprintf(info_buf,
                                                "Non-zero return code (%d) from MPI_Start for Bsend",
                                                ierr);
                                        MPITEST_message(MPITEST_NONFATAL,
                                                        info_buf);
                                        MPI_Error_string(ierr,
                                                         &info_buf[0],
                                                         &size);
                                        MPITEST_message(MPITEST_NONFATAL,
                                                        info_buf);
                                        loop_fail++;
                                    }

                                    flag = FALSE;
                                    while (!flag) {
                                        ierr =
                                            MPI_Test(&send_request, &flag,
                                                     &send_stat);
                                    }
                                    if (ierr != MPI_SUCCESS) {
                                        sprintf(info_buf,
                                                "Non-zero return code (%d) from MPI_Test on Start for Bsend_init",
                                                ierr);
                                        MPITEST_message(MPITEST_NONFATAL,
                                                        info_buf);
                                        MPI_Error_string(ierr,
                                                         &info_buf[0],
                                                         &size);
                                        MPITEST_message(MPITEST_NONFATAL,
                                                        info_buf);
                                        loop_fail++;
                                    }



                                    /* Test Ibsend Error Test  */
 /*--------- Call MPI_Request_free -------*/
                                    ierr = MPI_Request_free(&send_request);
                                    if (ierr != MPI_SUCCESS) {
                                        sprintf(info_buf,
                                                "Non-zero return code (%d) from MPI_Request_free after send",
                                                ierr);
                                        MPITEST_message(MPITEST_NONFATAL,
                                                        info_buf);
                                        MPI_Error_string(ierr,
                                                         &info_buf[0],
                                                         &size);
                                        MPITEST_message(MPITEST_NONFATAL,
                                                        info_buf);
                                        fail++;
                                    }

                                    if (send_request != MPI_REQUEST_NULL) {
                                        sprintf(info_buf,
                                                "request not equal to  MPI_REQUEST_NULL after calling MPI_Request_free");
                                        MPITEST_message(MPITEST_NONFATAL,
                                                        info_buf);
                                        MPI_Error_string(ierr,
                                                         &info_buf[0],
                                                         &size);
                                        MPITEST_message(MPITEST_NONFATAL,
                                                        info_buf);
                                        fail++;
                                    }

                                    /* Make sure nothing got scrambled */
                                    EXIT_ON_ERROR(cuMemcpy((CUdeviceptr)send_buffer, gpu_send_buffer,
                                                            gpu_send_buffer_size));

                                    error =
                                        MPITEST_buffer_errors(test_type,
                                                              length,
                                                              value,
                                                              send_buffer);
                                    if (error) {
                                        sprintf(info_buf,
                                                "%d errors in send buffer (%d,%d,%d) len %d commsize %d commtype %d data_type %d root %d",
                                                error, length_count,
                                                comm_count, type_count,
                                                length, test_nump,
                                                comm_type, test_type,
                                                root);
                                        MPITEST_message(MPITEST_NONFATAL,
                                                        info_buf);
                                        loop_fail++;
                                    } else {
                                        sprintf(info_buf,
                                                "%d errors found in send buffer",
                                                error);
                                        MPITEST_message(MPITEST_INFO2,
                                                        info_buf);
                                    }

                                    if (loop_fail != 0)
                                        fail++;
                                }       /* End of for loop:  Send from
                                         * current root to all nodes  */

                            }


                            /* End of test for sending node */
 /*------------------------------------------------
					       POST  RECEIVES
			
			    INTRA and INTER:  Each node of send group gets to
			    be root and send to all nodes in receive group
			    -------------------------------------------------*/
                            if ((!inter_comm) ||
                                (inter_comm
                                 && MPITEST_inter == recv_group))
                            {
                                /* Initialize the receive buffer to -1 */

                                MPITEST_dataTemplate_init(&value, -1);
                                MPITEST_init_buffer(test_type, length + 1,
                                                    value, recv_buffer);
                                loop_cnt++;
                                loop_fail = 0;

                                /* Initialize entire GPU recv buffer to match host recv buffer */
                                EXIT_ON_ERROR(cuMemcpy(gpu_recv_buffer, (CUdeviceptr)recv_buffer, 
                                                        gpu_recv_buffer_size));


                                ierr = MPI_Recv_init((void *)gpu_recv_buffer,
                                                     length,
                                                     MPITEST_mpi_datatypes
                                                     [test_type], root,
                                                     MPI_ANY_TAG, comm,
                                                     &recv_request);

                                if (ierr != MPI_SUCCESS) {
                                    sprintf(info_buf,
                                            "Non-zero return code (%d) from MPI_Recv_init",
                                            ierr);
                                    MPITEST_message(MPITEST_NONFATAL,
                                                    info_buf);
                                    MPI_Error_string(ierr, &info_buf[0],
                                                     &size);
                                    MPITEST_message(MPITEST_NONFATAL,
                                                    info_buf);
                                    loop_fail++;
                                }

                                /* Irecv  Error Test  */
                                ierr = MPI_Start(&recv_request);

                                if (ierr != MPI_SUCCESS) {
                                    sprintf(info_buf,
                                            "Non-zero return code (%d) from MPI_Start for Recv",
                                            ierr);
                                    MPITEST_message(MPITEST_NONFATAL,
                                                    info_buf);
                                    MPI_Error_string(ierr, &info_buf[0],
                                                     &size);
                                    MPITEST_message(MPITEST_NONFATAL,
                                                    info_buf);
                                    loop_fail++;
                                }

                                ierr = MPI_Wait(&recv_request, &recv_stat);


                                if (ierr != MPI_SUCCESS) {
                                    sprintf(info_buf,
                                            "Non-zero return code (%d) from MPI_Wait on  Receive",
                                            ierr);
                                    MPITEST_message(MPITEST_NONFATAL,
                                                    info_buf);
                                    MPI_Error_string(ierr, &info_buf[0],
                                                     &size);
                                    MPITEST_message(MPITEST_NONFATAL,
                                                    info_buf);
                                    fail++;
                                }



                                /* End of Wait Error Test  */
 /*------- Call MPI_Request_free ------------*/
                                ierr = MPI_Request_free(&recv_request);
                                if (ierr != MPI_SUCCESS) {
                                    sprintf(info_buf,
                                            "Non-zero return code (%d) from MPI_Request_free after receive",
                                            ierr);
                                    MPITEST_message(MPITEST_NONFATAL,
                                                    info_buf);
                                    MPI_Error_string(ierr, &info_buf[0],
                                                     &size);
                                    MPITEST_message(MPITEST_NONFATAL,
                                                    info_buf);
                                    fail++;
                                }

                                if (recv_request != MPI_REQUEST_NULL) {
                                    sprintf(info_buf,
                                            "request not equal to  MPI_REQUEST_NULL after calling MPI_Request_free");
                                    MPITEST_message(MPITEST_NONFATAL,
                                                    info_buf);
                                    MPI_Error_string(ierr, &info_buf[0],
                                                     &size);
                                    MPITEST_message(MPITEST_NONFATAL,
                                                    info_buf);
                                    fail++;
                                }

                                /*
                                 * Set up the dataTemplate for checking the
                                 * recv'd buffer.  Note that the sending node
                                 * rank will be sent.
                                 */

                                /* Move entire GPU recv buffer back into host recv buffer for checking */
                                EXIT_ON_ERROR(cuMemcpy((CUdeviceptr)recv_buffer, gpu_recv_buffer,
                                                        gpu_recv_buffer_size));

                                MPITEST_dataTemplate_init(&value,
                                                          recv_stat.
                                                          MPI_SOURCE);


                                error = MPITEST_buffer_errors(test_type,
                                                              length,
                                                              value,
                                                              recv_buffer);

                                /* check for receive buffer overflow */
                                MPITEST_dataTemplate_init(&value, -1);


                                error +=
                                    MPITEST_buffer_errors_ov(test_type,
                                                             length, value,
                                                             recv_buffer);

                                if (error) {
                                    sprintf(info_buf,
                                            "%d errors in buffer (%d,%d,%d) len %d commsize %d commtype %d data_type %d root %d",
                                            error, length_count,
                                            comm_count, type_count, length,
                                            test_nump, comm_type,
                                            test_type, root);
                                    loop_fail++;
                                    MPITEST_message(MPITEST_NONFATAL,
                                                    info_buf);
                                } else {
                                    sprintf(info_buf,
                                            "%d errors found in buffer",
                                            error);
                                    MPITEST_message(MPITEST_INFO2,
                                                    info_buf);
                                }
                                /*
                                 * Call the MPI_Get_Count function, and
                                 * compare value with length
                                 */
                                cnt_len = -1;

                                ierr = MPI_Get_count(&recv_stat,
                                                     MPITEST_mpi_datatypes
                                                     [test_type],
                                                     &cnt_len);
                                if (ierr != MPI_SUCCESS) {
                                    sprintf(info_buf,
                                            "Non-zero return code (%d) from MPI_Get_count",
                                            ierr);
                                    MPITEST_message(MPITEST_NONFATAL,
                                                    info_buf);
                                    MPI_Error_string(ierr, &info_buf[0],
                                                     &size);
                                    MPITEST_message(MPITEST_NONFATAL,
                                                    info_buf);
                                    loop_fail++;
                                }

                                /*
                                 * Print non-fatal error if Received lenth
                                 * not equal to send length
                                 */
                                error = length - cnt_len;

                                if (error) {
                                    sprintf(info_buf,
                                            "Send/Receive lengths differ - Sender(node/length)=%d/%d,  Receiver(node/length)=%d/%d",
                                            recv_stat.MPI_SOURCE, length,
                                            MPITEST_current_rank, cnt_len);
                                    MPITEST_message(MPITEST_NONFATAL,
                                                    info_buf);
                                    loop_fail++;
                                }

                                if (loop_fail != 0)
                                    fail++;
                            }
                            /* End of for loop:  Receives from INTRA root
                               * sender or INTER send group root */
                            if ((!inter_comm
                                 && MPITEST_current_rank == root)
                                || (inter_comm
                                    && MPITEST_inter == send_group
                                    && MPITEST_current_rank == root))

                                /* If sender, detach Ibsend buffer  */

                                berr = MPI_Buffer_detach(&buff, &bsize);

                            if (berr != MPI_SUCCESS) {
                                sprintf(info_buf,
                                        "Non-zero return code (%d) from MPI_Buffer_detach",
                                        berr);
                                MPITEST_message(MPITEST_NONFATAL,
                                                info_buf);
                                MPI_Error_string(berr, &info_buf[0],
                                                 &size);
                                MPITEST_message(MPITEST_NONFATAL,
                                                info_buf);
                                fail++;
                            }
                            /* Buffer detach error Test  */
                            ierr = MPI_Barrier(barrier_comm);
                            if (ierr != MPI_SUCCESS) {
                                sprintf(info_buf,
                                        "Non-zero return code (%d) from MPI_Barrier",
                                        ierr);
                                MPITEST_message(MPITEST_NONFATAL,
                                                info_buf);
                                MPI_Error_string(ierr, &info_buf[0],
                                                 &size);
                                MPITEST_message(MPITEST_NONFATAL,
                                                info_buf);
                                fail++;
                            }

                        }       /* Loop over Roots  */

                    }           /* End of extra loop for INTER-communicator  */

                }               /* Loop over Message Lengths  */

                free(send_buffer);
                free(recv_buffer);
				EXIT_ON_ERROR(cuMemFree(gpu_send_buffer));
				EXIT_ON_ERROR(cuMemFree(gpu_recv_buffer));

            }                   /* Loop over Data Types  */

            if (inter_comm) {
                ierr = MPI_Comm_free(&barrier_comm);
                if (ierr != MPI_SUCCESS) {
                    sprintf(info_buf,
                            "Non-zero return code (%d) from MPI_Comm_free",
                            ierr);
                    MPITEST_message(MPITEST_NONFATAL, info_buf);
                    MPI_Error_string(ierr, &info_buf[0], &size);
                    MPITEST_message(MPITEST_FATAL, info_buf);
                }
            }

        }
        /* node rank not defined for this communicator */
        MPITEST_free_communicator(comm_type, &comm);


    }                           /* Loop over Communicators  */

    free(buff);

    /* report overall results  */

    MPITEST_report(loop_cnt - fail, fail, 0, testname);

    MPI_Finalize();

    /* 77 is a special return code for a skipped test. So we don't 
     * want to return it */
    if(77 == fail) {
        fail++;
    }
	CUDATEST_finalize();
    return fail;

}                               /* main() */
