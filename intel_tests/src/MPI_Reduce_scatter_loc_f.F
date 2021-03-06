C ----------------------------------------------------------------------
C MESSAGE PASSING INTERFACE TEST CASE SUITE
C 
C Copyright - 1996 Intel Corporation
C 
C Intel Corporation hereby grants a non-exclusive license under Intel's
C copyright to copy, modify and distribute this software for any purpose 
C and without fee, provided that the above copyright notice and the
C following paragraphs appear on all copies.
C 
C Intel Corporation makes no representation that the test cases
C comprising this suite are correct or are an accurate representation
C of any standard.
C 
C IN NO EVENT SHALL INTEL HAVE ANY LIABILITY FOR ANY DIRECT, INDIRECT
C OR SPECULATIVE DAMAGES, (INCLUDING WITHOUT LIMITING THE FOREGOING,
C CONSEQUENTIAL, INCIDENTAL AND SPECIAL DAMAGES) INCLUDING, BUT NOT
C LIMITED TO INFRINGEMENT, LOSS OF USE, BUSINESS INTERRUPTIONS, AND
C LOSS OF PROFITS, IRRESPECTIVE OF WHETHER INTEL HAS ADVANCE NOTICE OF
C THE POSSIBILITY OF ANY SUCH DAMAGES.
C 
C INTEL CORPORATION SPECIFICALLY DISCLAIMS ANY WARRANTIES INCLUDING,
C BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS
C FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.  THE SOFTWARE PROVIDED
C HEREUNDER IS ON AN "AS IS" BASIS AND INTEL CORPORATION HAS NO
C OBLIGATION TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS OR
C MODIFICATIONS.
C 
C ----------------------------------------------------------------------
C **********************************************************************
C                        Test for MPI_Reduce_scatter_loc()
C 
C This code tests the MPI_Reduce_scatter() function.
C MPI_Reduce_scatter() performs a binary, associative operation in such
C a way that data from each process is operand exactly once.  The result
C is scattered to all processes.  Process "i" receives recv_counts[i] elements
C of data.
C 
C The operations to be looped over are in the array MPITEST_default_ops[].
C This array must be initialized at runtime, after the call to MPI_Init().
C 
C **********************************************************************

#include "foptions.h"

#include "mpitest_cfgf.h"

      PROGRAM MAIN

#include "mpitestf.h"
      INCLUDE 'mpif.h'
      INCLUDE 'externalf.h'
      INCLUDE 'reducef.h'

      INTEGER MPITEST_DEFAULT_OPS(10)

      INTEGER    COMM_INDEX
C                               the array index of the current comm
      INTEGER    COMM_TYPE
C                               the index of the current communicator type
      INTEGER    COMM_COUNT
C                               loop counter for communicator loop
      INTEGER    COMM_SIZE
C                               number of ranks in the current communicator
      LOGICAL    INTER_FLAG
C                               used to test if intercommunicator
      INTEGER    TYPE_COUNT
C                               number of data types to test
      INTEGER    TEST_TYPE
C                               index of current data type
      INTEGER    LENGTH_COUNT
C                               number of data lengths to test
      INTEGER    MAX_LENGTH
C                               max message length to test
      INTEGER    MAX_BYTE_LENGTH
C                               max byte length to test
      INTEGER    MAX_BUFF_LENGTH
C                               max length that will fit in buffers
      INTEGER    BYTE_LENGTH
C                               current byte length to test
      INTEGER    LENGTH
C                               current length to test 
      INTEGER    NUM_OPS
C                               number of operations to test
      INTEGER    OP_COUNT
C                               index of current operation to test

      INTEGER    FAIL
      INTEGER    LOOP_CNT
C                               counts total number of failures, loops
      INTEGER    ERROR
C                               number of data errors found in buffer
      INTEGER    IERR
      INTEGER    ERR
C                               return value from MPI calls
      INTEGER    COMM
C                               MPI communicator under test
      INTEGER    I, J, K, L, M

      LOGICAL    HASOP
      INTEGER    LOCALTYPE

      MPITEST_BUF_TYPE  SEND_BUFFER(MAX_BUFF_SIZE*2+MAX_RANKS)
      MPITEST_BUF_TYPE  RECV_BUFFER(MAX_BUFF_SIZE*2+MAX_RANKS)
      INTEGER    ERRSIZE

      MPITEST_BUF_TYPE     VALUE(2)
      INTEGER    COUNTS(MAX_RANKS)

      CHARACTER*(IOSIZE)  INFOBUF
      CHARACTER*32   TESTNAME

 99   FORMAT(A,INT_FMT)
 179  FORMAT(A,I6,A,I6,A,I6,A,I3,A,INT_FMT,A,I3)
 189  FORMAT(INT_FMT,A,A,I6,A,I6,A,I6,A,I3,A,INT_FMT,A,I3)

C
C     Initialize the MPI environment
C
      CALL MPI_INIT(IERR)
      IF (IERR .NE. MPI_SUCCESS) THEN
        WRITE(INFOBUF,99) 'MPI_INIT() returned', IERR
        CALL MPITEST_MESSAGE(MPITEST_FATAL, INFOBUF)
      END IF
C
C     Initialize the MPITEST environment
C
      CALL MPITEST_INIT(IERR)
      TESTNAME = 'MPI_Reduce_scatter_loc()'
      IF (MPITEST_ME .EQ. 0) THEN
        INFOBUF = 'Starting test '//TESTNAME
        CALL MPITEST_MESSAGE(MPITEST_INFO0, INFOBUF)
      END IF


C
C     Initialize the loop/error counters
C
      LOOP_CNT = 0
      FAIL = 0

C
C     Get number of communicators, data types and message lengths to test
C
      CALL MPITEST_NUM_COMM_SIZES(COMM_COUNT)
      CALL MPITEST_NUM_DATATYPES(TYPE_COUNT)
      CALL MPITEST_NUM_MESSAGE_LENGTHS(LENGTH_COUNT)

C
C     Get max message length for test
C
      CALL MPITEST_GET_MAX_MESSAGE_LENGTH(MAX_BYTE_LENGTH)

C
C     Initialize operations
C
      CALL SET_DEFAULT_OPS(MPITEST_DEFAULT_OPS, NUM_OPS)

C
C     Loop through communicators
C
      DO 100 I=1,COMM_COUNT
        CALL MPITEST_GET_COMM_INDEX(I, COMM_INDEX)
        CALL MPITEST_GET_COMM_TYPE(I, COMM_TYPE)

        CALL MPITEST_GET_COMMUNICATOR(COMM_TYPE, COMM_INDEX, COMM,
     $        COMM_SIZE)

C       Skip to end of loop if this node is not part of current
C       communicator

        IF (COMM .EQ. MPI_COMM_NULL) GO TO 201

C       Skip if intercommunicator

        CALL MPI_COMM_TEST_INTER(COMM, INTER_FLAG, IERR)
        IF (INTER_FLAG .EQV. .TRUE.) GO TO 201

C
C       Loop through data types.
C
        DO 200 J=1,TYPE_COUNT

          CALL MPITEST_GET_DATATYPE(J, TEST_TYPE)
C
C         Convert the number of bytes in the maximum length message
C         into the number of elements of the current type 
          CALL MPITEST_BYTE_TO_ELEMENT(TEST_TYPE, MAX_BYTE_LENGTH,
     $          MAX_LENGTH)
          CALL MPITEST_BYTE_TO_ELEMENT(TEST_TYPE,
     $          MAX_BUFF_SIZE * MPITEST_BUF_EXTENT, MAX_BUFF_LENGTH)

          IF (MAX_LENGTH .GE. (MAX_BUFF_LENGTH / MPITEST_NUMP - 1))
     $         MAX_LENGTH = MAX_BUFF_LENGTH / MPITEST_NUMP - 1
          IF (MAX_LENGTH .LE. 0) MAX_LENGTH = 0


C
C         Loop through data lengths
C
          DO 300 K=1,LENGTH_COUNT
            CALL MPITEST_GET_MESSAGE_LENGTH(K, BYTE_LENGTH)
            CALL MPITEST_BYTE_TO_ELEMENT(TEST_TYPE, BYTE_LENGTH,
     $          LENGTH)
            IF (LENGTH .GT. MAX_LENGTH) LENGTH = MAX_LENGTH

            DO 250 M=1,COMM_SIZE
              COUNTS(M) = LENGTH
 250        CONTINUE

C
C           Loop through the different operations
C
            DO 350 OP_COUNT=1,NUM_OPS
              CALL HAS_OP(OP_COUNT, TEST_TYPE, HASOP)
              IF (HASOP .EQV. .FALSE.)
     $          GO TO 350

C             Print an informational message
              IF (MPITEST_CURRENT_RANK .EQ. 0) THEN
                WRITE (INFOBUF,179) 'commindex',COMM_INDEX,
     $              ', commsize', COMM_SIZE, 
     $              ', commtype', COMM_TYPE,
     $              ', datatype', TEST_TYPE, ', length',  LENGTH,
     $              '. op', OP_COUNT
                CALL MPITEST_MESSAGE(MPITEST_INFO1, INFOBUF)
              END IF
              LOOP_CNT = LOOP_CNT + 1

C
C             Initialize send data, receive buffer
C
                CALL MPITEST_DATATEMPLATE_INIT_POS(VALUE,
     $              MPITEST_CURRENT_RANK + 1, 1, TEST_TYPE)
                CALL MPITEST_DATATEMPLATE_INIT_POS(VALUE,
     $              MPITEST_CURRENT_RANK + 1, 2, TEST_TYPE)
                CALL MPITEST_INIT_BUFFER_LOC(TEST_TYPE,
     $                                       LENGTH * COMM_SIZE,
     $                                       VALUE, SEND_BUFFER)
                CALL MPITEST_DATATEMPLATE_INIT(VALUE, -1,
     $                TEST_TYPE)
                CALL MPITEST_INIT_BUFFER_LOC(TEST_TYPE, LENGTH + 1,
     $                                      VALUE, RECV_BUFFER)

                IF (MPITEST_MPI_DATATYPES(TEST_TYPE) .EQ. MPI_INTEGER)
     $            THEN
                     LOCALTYPE = MPI_2INTEGER
                ELSE IF (MPITEST_MPI_DATATYPES(TEST_TYPE) .EQ. MPI_REAL)
     $            THEN
                     LOCALTYPE = MPI_2REAL
                ELSE IF (MPITEST_MPI_DATATYPES(TEST_TYPE) .EQ.
     $                   MPI_DOUBLE_PRECISION) THEN
                  LOCALTYPE = MPI_2DOUBLE_PRECISION
                END IF

              IERR = 0
              IF (LENGTH .NE. 0)
     $          CALL MPI_REDUCE_SCATTER(SEND_BUFFER,RECV_BUFFER,COUNTS,
     $             LOCALTYPE, MPITEST_DEFAULT_OPS(OP_COUNT), COMM, IERR)
              IF (IERR .NE. MPI_SUCCESS) THEN
                WRITE(INFOBUF,99) 'MPI_REDUCE_SCATTER() returned', IERR
                CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
                INFOBUF=' '
                CALL MPI_ERROR_STRING(IERR, INFOBUF, ERRSIZE, ERR)
                CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
                FAIL = FAIL + 1
              END IF
C
C               Check data for errors and overflow
C
              CALL GET_REDUCE_ANSWER(OP_COUNT, COMM_SIZE, VALUE,
     $            TEST_TYPE)
              CALL MPITEST_BUFFER_ERRORS_LOC(TEST_TYPE, LENGTH,
     $            VALUE, RECV_BUFFER, ERROR)
              IF (ERROR .NE. 0) THEN
                IF (IERR .EQ. MPI_SUCCESS) FAIL = FAIL + 1
                WRITE (INFOBUF,189) ERROR,
     $            ' errors in buffer, ',
     $            'commindex',COMM_INDEX,', commsize', COMM_SIZE,
     $            ', commtype', COMM_TYPE,
     $            ', datatype', TEST_TYPE, ', length',  LENGTH,
     $            '. op', OP_COUNT
                CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
              END IF


 350        CONTINUE

 300      CONTINUE

 200    CONTINUE
 201    CONTINUE

        CALL MPITEST_FREE_COMMUNICATOR(COMM_TYPE, COMM, IERR)

 100  CONTINUE

C
C     Report overall results
C
      CALL MPITEST_REPORT(LOOP_CNT - FAIL, FAIL, 0, TESTNAME)

      CALL MPI_FINALIZE(IERR)
      IF(FAIL .EQ. 77) THEN
          FAIL = FAIL + 1
      END IF
      CALL EXIT(FAIL)


      END  
  
  
#include "foptions.h"

      SUBROUTINE SET_DEFAULT_OPS(OP_ARRAY, NUMBER)
C ************************************************************************
C  Sets up the default operation array.  Returns the number of default 
C  operations.
C ************************************************************************
#include "mpitestf.h"
      INCLUDE 'mpif.h'

      INTEGER OP_ARRAY(10)
      INTEGER NUMBER


      OP_ARRAY(1) = MPI_MAXLOC
      OP_ARRAY(2) = MPI_MINLOC
  
      NUMBER = 2

      END


      SUBROUTINE HAS_OP(OP, TEST_TYPE, HASOP)
C ************************************************************************
C Determines whether a particular operation may be applied to a particular 
C data type, as specified in section 4.9.2 of the MPI Standard.
C ************************************************************************
#include "mpitestf.h"
      INCLUDE 'externalf.h'

      INTEGER OP
      INTEGER TEST_TYPE
      LOGICAL HASOP

      IF ((TEST_TYPE .EQ. MPITEST_INTEGER) .OR.
     $    (TEST_TYPE .EQ. MPITEST_REAL) .OR.
     $    (TEST_TYPE .EQ. MPITEST_DOUBLE_PRECISION)) THEN
        HASOP = .TRUE.
      ELSE
        HASOP = .FALSE.
      END IF

      END


      SUBROUTINE APPLY_INT_OP(OP_INDEX, X1, X2, ANSWER)
C ***************************************************************************
C Applies a binary operator to the two integers x1 and x2, returning the 
C result.  The binary operation is determined by the integer op_index.  The 
C mapping of op_index to operation is determined by the array 
C MPITEST_default_ops[], which is set at runtime in the main test code.
C ***************************************************************************
      INTEGER OP_INDEX
      INTEGER X1
      INTEGER X2
      INTEGER ANSWER(2)
  
      IF (OP_INDEX .EQ. 1) THEN
C       maximum
        ANSWER(1) = MAX(X1, X2)
        ANSWER(2) = MAX(X1, X2)
      ELSE IF (OP_INDEX .EQ. 2) THEN
C       minimum
        ANSWER(1) = MIN(X1, X2)
        ANSWER(2) = MIN(X1, X2)
      END IF
      END

      SUBROUTINE APPLY_DOUBLE_OP(OP_INDEX, X1, X2, ANSWER)
C ***************************************************************************
C Applies a binary operator to the two doubles x1 and x2, returning the 
C result.  The binary operation is determined by the integer op_index.  The 
C mapping of op_index to operation is determined by the array 
C MPITEST_default_ops[], which is set at runtime in the main test code.
C ***************************************************************************
      INTEGER OP_INDEX
      DOUBLE PRECISION X1
      DOUBLE PRECISION X2
      DOUBLE PRECISION ANSWER(2)

      IF (OP_INDEX .EQ. 1) THEN
C       maximum
        ANSWER(1) = MAX(X1, X2)
        ANSWER(2) = MAX(X1, X2)
      ELSE IF (OP_INDEX .EQ. 2) THEN
C       minimum
        ANSWER(1) = MIN(X1, X2)
        ANSWER(2) = MIN(X1, X2)
      END IF
      END


      SUBROUTINE APPLY_REAL_OP(OP_INDEX, X1, X2, ANSWER)
C ***************************************************************************
C Applies a binary operator to the two real x1 and x2, returning the 
C result.  The binary operation is determined by the integer op_index.  The 
C mapping of op_index to operation is determined by the array 
C MPITEST_default_ops[], which is set at runtime in the main test code.
C ***************************************************************************
      INTEGER OP_INDEX
      REAL X1
      REAL X2
      REAL ANSWER(2)

      IF (OP_INDEX .EQ. 1) THEN
C       maximum
        ANSWER(1) = MAX(X1, X2)
        ANSWER(2) = MAX(X1, X2)
      ELSE IF (OP_INDEX .EQ. 2) THEN
C       minimum
        ANSWER(1) = MIN(X1, X2)
        ANSWER(2) = MIN(X1, X2)
      END IF
      END


      SUBROUTINE GET_REDUCE_ANSWER(OP_INDEX, NUMP, ANSWER, TYPE)
C ************************************************************************
C Apply the binary operation specified by op_index to the numbers 
C (0, 1, 2, ..., nump-1), and fill in the dataTamplate object based on the 
C results.  The mapping of op_index to operation is determined by the array 
C MPITEST_default_ops[], which is set at runtime in the main test code. 
C In order for the answer produced by this routine to match the
C answer generated by the MPI_Reduce() operation in the test code, the
C send buffer of process with rank "i" must have been initialized with "i". 
C 
C This routine applies the operation to both integers and to doubles, 
C in case the double and float buffers are initialized differently than the 
C integer buffers.
C ************************************************************************
#include "mpitestf.h"
      INCLUDE 'externalf.h'

      INTEGER OP_INDEX
      INTEGER NUMP, TYPE
      MPITEST_BUF_TYPE ANSWER(2)

      INTEGER I1, I2
      REAL    R1, R2
      DOUBLE PRECISION D1, D2

      EXTERNAL APPLY_INT_OP
      EXTERNAL APPLY_REAL_OP
      EXTERNAL APPLY_DOUBLE_OP

      IF (NUMP .EQ. 1) THEN
        CALL MPITEST_DATATEMPLATE_INIT_POS(ANSWER, 1, 1, TYPE)
        CALL MPITEST_DATATEMPLATE_INIT_POS(ANSWER, 1, 2, TYPE)
        RETURN
      END IF

      I1 = 1
      I2 = 2

      R1 = 1.0
      R2 = 2.0

      D1 = 1.0
      D2 = 2.0

      IF (TYPE .EQ. MPITEST_INTEGER) THEN
         CALL APPLY_INT_OP(OP_INDEX, I1, I2, ANSWER)
      ELSE IF (TYPE .EQ. MPITEST_REAL) THEN
         CALL APPLY_REAL_OP(OP_INDEX, R1, R2, ANSWER)
      ELSE IF (TYPE .EQ. MPITEST_DOUBLE_PRECISION) THEN
         CALL APPLY_DOUBLE_OP(OP_INDEX, D1, D2, ANSWER)
      END IF

      IF (TYPE .EQ. MPITEST_INTEGER) THEN
         I1 = 3
         DO 100 I2=3, NUMP
           IF (I2 .GT. NUMP) GO TO 101
           IF ((I1 .GT. 2) .AND. (OP_INDEX .EQ. 4)) I1=1
           CALL APPLY_INT_OP(OP_INDEX, ANSWER, I1, ANSWER)
           I1 = I1 + 1
 100     CONTINUE
 101     CONTINUE
      ELSE IF (TYPE .EQ. MPITEST_REAL) THEN
         R1 = 3
         DO 200 I2=3, NUMP
           IF (I2 .GT. NUMP) GO TO 201
           IF ((R1 .GT. 2) .AND. (OP_INDEX .EQ. 4)) R1=1
           CALL APPLY_REAL_OP(OP_INDEX, ANSWER, R1, ANSWER)
           R1 = R1 + 1
 200     CONTINUE
 201     CONTINUE
      ELSE IF (TYPE .EQ. MPITEST_DOUBLE_PRECISION) THEN
         D1 = 3
         DO 400 I2=3, NUMP
           IF (I2 .GT. NUMP) GO TO 401
           IF ((D1 .GT. 2) .AND. (OP_INDEX .EQ. 4)) D1=1
           CALL APPLY_DOUBLE_OP(OP_INDEX, ANSWER, D1, ANSWER)
           D1 = D1 + 1
 400     CONTINUE
 401     CONTINUE
      END IF
  
      END

