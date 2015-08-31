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
                       Test for MPI_Reduce_scatter()

This code tests the MPI_Reduce_scatter() function.
MPI_Reduce_scatter() performs a binary, associative operation in such
a way that data from each process is operand exactly once.  The result
is scattered to all processes.  Process "i" receives recv_counts[i] elements
of data.

The operations to be looped over are stored in the array MPITEST_default_ops[].
This array must be initialized at runtime, after the call to MPI_Init().

There are several auxiliary functions needed for Reduce tests, and these
are in the module reduce.c, with its associated header file reduce.h .

Revision History:
  1               gm   Original version.
  2     2/1/96    gt   Reduced size of send buffer to stay within bounds,
                       and modified data sent/expected.
******************************************************************************/

#include "mpitest_cfg.h"
#include "mpitest.h"
#include "reduce.h"

extern struct MPITEST_op MPITEST_reduce_ops[];


int main(int argc, char *argv[])
{
    int
     test_type,                 /* the index of the current buffer type              */
     length,                    /* The length of the current buffer                  */
     byte_length,               /* The length of the current buffer in bytes         */
     test_nump,                 /* The number of processors in current
                                 * communicator  */
     comm_index,                /* the array index of the current comm               */
     comm_type,                 /* the index of the current communicator type        */
     type_count,                /* loop counter for data type loop                   */
     length_count,              /* loop counter for message length loop              */
     comm_count,                /* loop counter for communicator loop                */
     error,                     /* errors from one MPI call                          */
     fail,                      /* counts total number of failures                   */
     size,                      /* return size from MPI_Error_string                 */
     loop_cnt,                  /* counts total number of loops through test         */
     ierr,                      /* return value from MPI calls                       */
     max_length,                /* maximum buffer length specified in config.
                                 * file   */
     max_byte_length,           /* maximum buffer length in bytes                    */
     num_ops,                   /* total number of predefined MPI ops                 */
     op_count, i, j;

    struct dataTemplate
     value;                     /* dataTemplate for initializing buffers             */
    struct dataTemplate
    *values;                    /* Array of dataTemplates for verbose init           */

    void *send_buffer,          /* message buffer                                     */
    *recv_buffer;

    int
    *counts;                    /* counts[i] is the number of elements recv'd
                                 * at proc. i */


    char
     info_buf[256],             /* buffer for passing mesages to MPITEST             */
     testname[128];             /* the name of this test                             */



    MPI_Comm comm;              /* MPI communicator                                  */

    MPI_Op MPITEST_default_ops[OP_ARRAY_SIZE];

    int inter_flag;

    ierr = MPI_Init(&argc, &argv);
    if (ierr != MPI_SUCCESS) {
        sprintf(info_buf, "MPI_Init() returned %d", ierr);
        MPITEST_message(MPITEST_FATAL, info_buf);
    }


    sprintf(testname, "MPI_Reduce_scatter()");

    num_ops = set_default_ops(MPITEST_default_ops);

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
                /* Ignore inter-communicator for collective functional tests */
                MPITEST_free_communicator(comm_type, &comm);
                sprintf(info_buf,
                        "Skipping intercommunicator (commtype: %d) for this test",
                        comm_type);
                MPITEST_message(MPITEST_INFO1, info_buf);

                continue;
            }
            counts = (int *) malloc(test_nump * sizeof(int));
            if (!counts)
                MPITEST_message(MPITEST_NONFATAL,
                                "Could not allocate counts array");

            for (type_count = 0; type_count < MPITEST_num_datatypes();
                 type_count++) {
                test_type = MPITEST_get_datatype(type_count);

                /* find the maximum sized buffer we will use */
                max_byte_length = MPITEST_get_max_message_length();

                /*
                 * convert the number of bytes to the number of elements of the
                 * current type
                 */
                max_length =
                    MPITEST_byte_to_element(test_type, max_byte_length);

                if (max_length < MPITEST_nump)
                    max_length = MPITEST_nump;  /* rev2 */
                max_length = max_length / MPITEST_nump; /* rev2 */


                /* then allocate the buffer */
                MPITEST_get_buffer(test_type, MPITEST_nump * max_length,
                                   &send_buffer);
                MPITEST_get_buffer(test_type, max_length, &recv_buffer);

                for (length_count = 0;
                     length_count < MPITEST_num_message_lengths();
                     length_count++) {
                    byte_length = MPITEST_get_message_length(length_count);
                    length =
                        MPITEST_byte_to_element(test_type, byte_length);

                    if (length > max_length)
                        length = max_length;    /* rev2 */

                    /* set up the counts array */
                    for (i = 0; i < test_nump; i++) {
                        counts[i] = length;
                    }

                    /* loop over binary operations */
                    for (op_count = 0; op_count < num_ops; op_count++) {
                        if (MPITEST_current_rank == 0) {
                            sprintf(info_buf,
                                    "length %d commsize %d commtype %d data_type %d operation %d",
                                    length, test_nump, comm_type,
                                    test_type, op_count);
                            MPITEST_message(MPITEST_INFO1, info_buf);
                        }

                        /* Set up the dataTemplate for initializing send buffer */
                        if (MPITEST_default_ops[op_count] == MPI_PROD) {
                            j = (MPITEST_current_rank + 1) / 2;
                            j = j * 2 - MPITEST_current_rank + 1;
                            MPITEST_dataTemplate_init(&value, j);
                        } else {
                            MPITEST_dataTemplate_init(&value,
                                                      MPITEST_current_rank
                                                      + 1);
                        }


                        /* Initialize send buffer */
                        MPITEST_init_buffer(test_type, test_nump * length,
                                            value, send_buffer);

                        /* Set up the dataTemplate for initializing recv buffer */
                        MPITEST_dataTemplate_init(&value, -1);

                        /* Initialize recv buffer */
                        MPITEST_init_buffer(test_type, length + 1,
                                            value, recv_buffer);


                        if (has_op(op_count, test_type) && (length != 0)) {
                            loop_cnt++;
                            ierr =
                                MPI_Reduce_scatter(send_buffer,
                                                   recv_buffer, counts,
                                                   MPITEST_mpi_datatypes
                                                   [test_type],
                                                   MPITEST_default_ops
                                                   [op_count], comm);

                            if (ierr != MPI_SUCCESS) {
                                sprintf(info_buf,
                                        "MPI_Reduce_scatter() returned %d",
                                        ierr);
                                MPITEST_message(MPITEST_NONFATAL,
                                                info_buf);
                                MPI_Error_string(ierr, &info_buf[0],
                                                 &size);
                                MPITEST_message(MPITEST_FATAL, info_buf);
                                fail++;
                            }

                            /* generate the correct answer */
                            get_reduce_answer(op_count, test_nump, &value);

                            /* error test */
                            error = 0;
                            error =
                                MPITEST_buffer_errors(test_type, length,
                                                      value, recv_buffer);

                            /* check for recv_buffer overflow */
                            MPITEST_dataTemplate_init(&value, -1);
                            error += MPITEST_buffer_errors_ov(test_type,
                                                              length,
                                                              value,
                                                              recv_buffer);

                            if (error) {
                                if (ierr == MPI_SUCCESS)
                                    fail++;
                                sprintf(info_buf,
                                        "%d errors found in buffer, len %d commsize %d commtype %d data_type %d op %d",
                                        error, length, test_nump,
                                        comm_type, test_type, op_count);
                                MPITEST_message(MPITEST_NONFATAL,
                                                info_buf);
                            } else {
                                sprintf(info_buf,
                                        "%d errors found in buffer",
                                        error);
                                MPITEST_message(MPITEST_INFO2, info_buf);
                            }

                        }


                    }

                }               /**** for (op_count=0;...) *******/

                free(send_buffer);
                free(recv_buffer);

            }                   /**** for (length=0;...) *******/

            free(counts);

        }
                                    /**** for (type=0;...) *******/
        MPITEST_free_communicator(comm_type, &comm);
    }                           /**** for (comm=0;...) *******/
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
#include "mpitest_def.h"
#include "mpitest.h"
#include "reduce.h"


/*************************************************************************
  Special functions and definitions for the reduce family of functions.
*************************************************************************/


int set_default_ops(MPI_Op * op_array)
/**********************************************************************
Sets up the default operation array.  Returns the number of default 
operations.
***********************************************************************/
{
    op_array[0] = MPI_MAX;
    op_array[1] = MPI_MIN;
    op_array[2] = MPI_SUM;
    op_array[3] = MPI_PROD;
    op_array[4] = MPI_LAND;
    op_array[5] = MPI_LOR;
    op_array[6] = MPI_LXOR;
    op_array[7] = MPI_BAND;
    op_array[8] = MPI_BOR;
    op_array[9] = MPI_BXOR;

    return 10;


}

int has_op(int op, int test_type)
/***********************************************************************
Determines whether a particular operation may be applied to a particular 
data type, as specified in section 4.9.2 of the MPI Standard.
*************************************************************************/
{
    switch (test_type) {

    case MPITEST_int:
    case MPITEST_short_int:
    case MPITEST_long:
    case MPITEST_unsigned_short:
    case MPITEST_unsigned:
    case MPITEST_unsigned_long:
        /* all pre-defined operations apply to integer types */
        return 1;
        break;


    case MPITEST_char:
    case MPITEST_unsigned_char:
    case MPITEST_byte:
        /* none of the pre-defined operations apply to the character types */
        return 0;
        break;


    case MPITEST_float:
    case MPITEST_double:
#if MPITEST_long_double_def
    case MPITEST_long_double:
#endif
        /* only the first four operations (min, max, sum, product) apply
           to float and double */
        if ((op == 0) || (op == 1) || (op == 2) || (op == 3))
            return 1;
        else
            return 0;

    default:
        return 0;
    }
}



long apply_int_op(int op_index, long x1, long x2)
/***************************************************************************
Applies a binary operator to the two integers x1 and x2, returning the 
result.  The binary operation is determined by the integer op_index.  The 
mapping of op_index to operation is determined by the array 
MPITEST_default_ops[], which is set at runtime in the main test code.
**************************************************************************/
{
    long value = 0;
    switch (op_index) {
    case 0:                    /* maximum */
        value = (x1 > x2 ? x1 : x2);
        break;
    case 1:                    /* minimum */
        value = (x1 < x2 ? x1 : x2);
        break;
    case 2:                    /* sum */
        value = x1 + x2;
        break;
    case 3:                    /* product */
        value = x1 * x2;
        break;
    case 4:                    /* logical and */
        value = x1 && x2;
        break;
    case 5:                    /* logical or */
        value = x1 || x2;
        break;
    case 6:                    /* logical xor */
        value = (x1 || x2) && !(x1 && x2);
        break;
    case 7:                    /* bitwise and */
        value = x1 & x2;
        break;
    case 8:                    /* bitwise or */
        value = x1 | x2;
        break;
    case 9:                    /* bitwise xor */
        value = x1 ^ x2;
        break;
    }
    return value;
}


#if MPITEST_longlong_def
long long int apply_longlong_op(int op_index, long long int x1,
                                long long int x2)
/***************************************************************************
Applies a binary operator to the two integers x1 and x2, returning the 
result.  The binary operation is determined by the integer op_index.  The 
mapping of op_index to operation is determined by the array 
MPITEST_default_ops[], which is set at runtime in the main test code.
**************************************************************************/
{
    long long int value = 0;
    switch (op_index) {
    case 0:                    /* maximum */
        value = (x1 > x2 ? x1 : x2);
        break;
    case 1:                    /* minimum */
        value = (x1 < x2 ? x1 : x2);
        break;
    case 2:                    /* sum */
        value = x1 + x2;
        break;
    case 3:                    /* product */
        value = x1 * x2;
        break;
    case 4:                    /* logical and */
        value = x1 && x2;
        break;
    case 5:                    /* logical or */
        value = x1 || x2;
        break;
    case 6:                    /* logical xor */
        value = (x1 || x2) && !(x1 && x2);
        break;
    case 7:                    /* bitwise and */
        value = x1 & x2;
        break;
    case 8:                    /* bitwise or */
        value = x1 | x2;
        break;
    case 9:                    /* bitwise xor */
        value = x1 ^ x2;
        break;
    }
    return value;
}
#endif


double apply_double_op(int op_index, double x1, double x2)
/***************************************************************************
Applies a binary operator to the two doubles x1 and x2, returning the 
result.  The binary operation is determined by the integer op_index.  The 
mapping of op_index to operation is determined by the array 
MPITEST_default_ops[], which is set at runtime in the main test code.
**************************************************************************/
{
    double value = 0;
    switch (op_index) {
    case 0:                    /* maximum */
        value = (x1 > x2 ? x1 : x2);
        break;
    case 1:                    /* minimum */
        value = (x1 < x2 ? x1 : x2);
        break;
    case 2:                    /* sum */
        value = x1 + x2;
        break;
    case 3:                    /* product */
        value = x1 * x2;
        break;
    }
    return value;
}


#if MPITEST_long_double_def
long double apply_long_double_op(int op_index, long double x1,
                                 long double x2)
/***************************************************************************
Applies a binary operator to the two long doubles x1 and x2, returning the
result.  The binary operation is determined by the integer op_index.  The
mapping of op_index to operation is determined by the array
MPITEST_default_ops[], which is set at runtime in the main test code.
**************************************************************************/
{
    long double value = 0;
    switch (op_index) {
    case 0:                    /* maximum */
        value = (x1 > x2 ? x1 : x2);
        break;
    case 1:                    /* minimum */
        value = (x1 < x2 ? x1 : x2);
        break;
    case 2:                    /* sum */
        value = x1 + x2;
        break;
    case 3:                    /* product */
        value = x1 * x2;
        break;
    }
    return value;
}
#endif



int get_reduce_answer(int op_index, int nump, struct dataTemplate *answer)
/************************************************************************
Apply the binary operation specified by op_index to the numbers 
(0, 1, 2, ..., nump-1), and fill in the dataTamplate object based on the 
results.  The mapping of op_index to operation is determined by the array 
MPITEST_default_ops[], which is set at runtime in the main test code. 
In order for the answer produced by this routine to match the
answer generated by the MPI_Reduce() operation in the test code, the
send buffer of process with rank "i" must have been initialized with "i". 

This routine applies the operation to both integers and to doubles, 
in case the double and float buffers are initialized differently than the 
integer buffers.
***********************************************************************/
{
    long x1 = 1, x2 = 2, ianswer;
#if MPITEST_longlong_def
    long long int lx1 = 1, lx2 = 2, lanswer;
#endif
    double dx1 = 1.0, dx2 = 2.0, danswer;
#if MPITEST_long_double_def
    long double ldx1 = 1.0, ldx2 = 2.0, ldanswer;
#endif

    if (nump == 1) {
        MPITEST_dataTemplate_init(answer, 1);
        return 0;
    }

    ianswer = apply_int_op(op_index, x1, x2);

    for (x1 = 3, x2 = 3; x2 <= nump; x1++, x2++) {
        if ((x1 > 2) && (op_index == 3))
            x1 = 1;
        ianswer = apply_int_op(op_index, ianswer, x1);
    }

    MPITEST_dataTemplate_init(answer, ianswer);

#if MPITEST_longlong_def
    lanswer = apply_longlong_op(op_index, lx1, lx2);
    for (lx2 = 3, x2 = 3; x2 <= nump; x2++, lx2 += 1) {
        if ((lx2 > 2) && (op_index == 3))
            lx2 = 1;
        lanswer = apply_longlong_op(op_index, lanswer, lx2);
    }
    answer->LongLong = lanswer;
#endif

    /* now take care of the real datatypes */
    if (op_index < 4) {
        danswer = apply_double_op(op_index, dx1, dx2);
        for (dx2 = 3.0, x2 = 3; x2 <= nump; x2++, dx2 += 1.0) {
            if ((dx2 > 2) && (op_index == 3))
                dx2 = 1;
            danswer = apply_double_op(op_index, danswer, dx2);
        }
        answer->Float = (float) danswer;
        answer->Double = danswer;

#if MPITEST_long_double_def
        ldanswer = apply_long_double_op(op_index, ldx1, ldx2);
        for (ldx2 = 3.0, x2 = 3; x2 <= nump; x2++, ldx2 += 1.0) {
            if ((ldx2 > 2) && (op_index == 3))
                ldx2 = 1;
            ldanswer = apply_long_double_op(op_index, ldanswer, ldx2);
        }
        answer->LongDouble = ldanswer;
#endif
    }

    return 0;

}
