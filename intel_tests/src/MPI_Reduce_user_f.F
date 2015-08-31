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
C                        Test for MPI_Reduce_user()
C 
C This code tests the MPI_Reduce() function with user-defined operation.
C 
C In addition to the normal loops provided by the MPITEST environment(message
C length, communicator size and type, and data type) this test also loops over
C the identity of the root process and over the operation to be used in the
C reduce call.
C 
C The operations to be looped over are in the array MPITEST_default_ops[].
C This array must be initialized at runtime, after the call to MPI_Init().
C This will test both commutative and non-commutative operations.
C 
C **********************************************************************

#include "foptions.h"

#include "mpitest_cfgf.h"

      PROGRAM MAIN

#include "mpitestf.h"
      INCLUDE 'externalf.h'
      INCLUDE 'reducef.h'
      INCLUDE 'mpif.h'

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
      INTEGER    ERROR, ERROR2
C                               number of data errors found in buffer
      INTEGER    IERR
      INTEGER    ERR
C                               return value from MPI calls
      INTEGER    ROOT
C                               the current Reduce root
      INTEGER    COMM
C                               MPI communicator under test
      INTEGER    I, J, K, L

      LOGICAL    HASOP

      MPITEST_BUF_TYPE SEND_BUFFER(MAX_BUFF_SIZE+MAX_RANKS)
      MPITEST_BUF_TYPE RECV_BUFFER(MAX_BUFF_SIZE+MAX_RANKS)
      INTEGER    ERRSIZE

      MPITEST_BUF_TYPE VALUE

      CHARACTER*(IOSIZE)  INFOBUF
      CHARACTER*32   TESTNAME

 99   FORMAT(A,INT_FMT)
 179  FORMAT(A,I6,A,I6,A,I6,A,I6,A,I3,A,INT_FMT,A,I3)
 189  FORMAT(INT_FMT,A,A,I6,A,I6,A,I6,A,I6,A,I3,A,INT_FMT,A,I3)

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
      TESTNAME = 'MPI_Reduce_user()'
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

          IF (MAX_LENGTH .GE. (MAX_BUFF_LENGTH - 1))
     $         MAX_LENGTH = MAX_BUFF_LENGTH - 1
          IF (MAX_LENGTH .LE. 0) MAX_LENGTH = 0


C
C         Loop through data lengths
C
          DO 300 K=1,LENGTH_COUNT
            CALL MPITEST_GET_MESSAGE_LENGTH(K, BYTE_LENGTH)
            CALL MPITEST_BYTE_TO_ELEMENT(TEST_TYPE, BYTE_LENGTH,
     $          LENGTH)
            IF (LENGTH .GT. MAX_LENGTH) LENGTH = MAX_LENGTH

C
C           Loop through the different operations
C
            DO 350 OP_COUNT=1,NUM_OPS
              CALL HAS_OP(OP_COUNT, TEST_TYPE, HASOP)
              IF (HASOP .EQV. .FALSE.)
     $          GO TO 350

C
C             Loop over different processes being the root
C
              DO 400 ROOT = 0, COMM_SIZE-1
C               Print an informational message
                IF (MPITEST_CURRENT_RANK .EQ. 0) THEN
                  WRITE (INFOBUF,179) 'commindex',COMM_INDEX,
     $                ', commsize', COMM_SIZE, 
     $                ', commtype', COMM_TYPE, ', root', ROOT,
     $                ', datatype', TEST_TYPE, ', length',  LENGTH,
     $                '. op', OP_COUNT
                  CALL MPITEST_MESSAGE(MPITEST_INFO1, INFOBUF)
                END IF
                LOOP_CNT = LOOP_CNT + 1

C
C               Initialize send data, receive buffer
C
                CALL MPITEST_DATATEMPLATE_INIT(VALUE,
     $              MPITEST_CURRENT_RANK+1, TEST_TYPE)
                CALL MPITEST_INIT_BUFFER(TEST_TYPE, LENGTH, VALUE,
     $                SEND_BUFFER)
                CALL MPITEST_DATATEMPLATE_INIT(VALUE, -1,
     $                TEST_TYPE)
                CALL MPITEST_INIT_BUFFER(TEST_TYPE, LENGTH+1, VALUE,
     $                RECV_BUFFER)


                IERR = 0
                IF (LENGTH .NE. 0)
     $            CALL MPI_REDUCE(SEND_BUFFER, RECV_BUFFER, LENGTH,
     $               MPITEST_MPI_DATATYPES(TEST_TYPE),
     $               MPITEST_DEFAULT_OPS(OP_COUNT), ROOT, COMM, IERR)
                IF (IERR .NE. MPI_SUCCESS) THEN
                  WRITE(INFOBUF,99) 'MPI_REDUCE() returned', IERR
                  CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
                  INFOBUF=' '
                  CALL MPI_ERROR_STRING(IERR, INFOBUF, ERRSIZE, ERR)
                  CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
                  FAIL = FAIL + 1
                END IF
C
C               Check data for errors and overflow
C
                IF (MPITEST_CURRENT_RANK .EQ. ROOT) THEN
                  CALL GET_REDUCE_ANSWER(OP_COUNT, COMM_SIZE,
     $                VALUE, TEST_TYPE)
                  CALL MPITEST_BUFFER_ERRORS(TEST_TYPE, LENGTH, VALUE,
     $                RECV_BUFFER, ERROR)
                  CALL MPITEST_DATATEMPLATE_INIT(VALUE,
     $                -1, TEST_TYPE)
                  CALL MPITEST_BUFFER_ERRORS_OV(TEST_TYPE, LENGTH,
     $                VALUE, RECV_BUFFER, ERROR2)
                  IF ((ERROR + ERROR2) .NE. 0) THEN
                    IF (IERR .EQ. MPI_SUCCESS) FAIL = FAIL + 1
                    WRITE (INFOBUF,189) ERROR+ERROR2,
     $                ' errors in buffer, ',
     $                'commindex',COMM_INDEX,', commsize', COMM_SIZE,
     $                ', commtype', COMM_TYPE, ', root', ROOT,
     $                ', datatype', TEST_TYPE, ', length',  LENGTH,
     $                '. op', OP_COUNT
                    CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
                  END IF
                END IF


 400          CONTINUE

 350        CONTINUE

 300      CONTINUE

 200    CONTINUE
 201    CONTINUE

        CALL MPITEST_FREE_COMMUNICATOR(COMM_TYPE, COMM, IERR)

 100  CONTINUE

C
C     Free the operations
C
      DO 600 I=1,NUM_OPS
        LOOP_CNT = LOOP_CNT + 1
        CALL MPI_OP_FREE(MPITEST_DEFAULT_OPS(I), IERR)
        IF (IERR .NE. MPI_SUCCESS) THEN
          WRITE(INFOBUF,99) 'MPI_OP_FREE() returned', IERR
          CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
          INFOBUF=' '
          CALL MPI_ERROR_STRING(IERR, INFOBUF, ERRSIZE, ERR)
          CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
          FAIL = FAIL + 1
        END IF
 600  CONTINUE

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

      INTEGER FUNCTION ADDOP(INVEC, INOUTVEC, LEN, DTYPE)
C *************************************************************************
C   Commutative user Op (addition)
C *************************************************************************
#include "mpitestf.h"
      INCLUDE 'mpif.h'

      INTEGER  INVEC(*)
      INTEGER  INOUTVEC(*)
      INTEGER  LEN
      INTEGER  DTYPE
      INTEGER  I

      EXTERNAL ADDOPR
      EXTERNAL ADDOPD
      EXTERNAL ADDOPC

C     Prevent compiler warning
      ADDOP = MPI_SUCCESS

      IF (LEN .EQ. 0) GO TO 101
      IF (DTYPE .EQ. MPI_INTEGER) THEN
        DO 100 I=1,LEN
          INOUTVEC(I) = INVEC(I) + INOUTVEC(I)
 100    CONTINUE
      ELSE IF (DTYPE .EQ. MPI_REAL) THEN
        CALL ADDOPR(INVEC, INOUTVEC, LEN)
      ELSE IF (DTYPE .EQ. MPI_DOUBLE_PRECISION) THEN
        CALL ADDOPD(INVEC, INOUTVEC, LEN)
      ELSE IF (DTYPE .EQ. MPI_COMPLEX) THEN
        CALL ADDOPC(INVEC, INOUTVEC, LEN)
      END IF

 101  CONTINUE

      END


      SUBROUTINE ADDOPR(INVEC, INOUTVEC, LEN)
C *************************************************************************
C   Commutative user Op (addition) for REAL
C *************************************************************************

      REAL     INVEC(*)
      REAL     INOUTVEC(*)
      INTEGER  LEN
      INTEGER  I

      DO 100 I=1,LEN
        INOUTVEC(I) = INVEC(I) + INOUTVEC(I)
 100  CONTINUE

      END


      SUBROUTINE ADDOPD(INVEC, INOUTVEC, LEN)
C *************************************************************************
C   Commutative user Op (addition) for DOUBLE PRECISION
C *************************************************************************

      DOUBLE PRECISION  INVEC(*)
      DOUBLE PRECISION  INOUTVEC(*)
      INTEGER  LEN
      INTEGER  I

      DO 100 I=1,LEN
        INOUTVEC(I) = INVEC(I) + INOUTVEC(I)
 100  CONTINUE

      END

      SUBROUTINE ADDOPC(INVEC, INOUTVEC, LEN)
C *************************************************************************
C   Commutative user Op (addition) for COMPLEX
C *************************************************************************

      COMPLEX  INVEC(*)
      COMPLEX  INOUTVEC(*)
      INTEGER  LEN
      INTEGER  I

      DO 100 I=1,LEN
        INOUTVEC(I) = INVEC(I) + INOUTVEC(I)
 100  CONTINUE

      END


      INTEGER FUNCTION INCOP(INVEC, INOUTVEC, LEN, DTYPE)
C *************************************************************************
C Non-commutative user Op.
C *************************************************************************
#include "mpitestf.h"
      INCLUDE 'mpif.h'

      INTEGER  INVEC(*)
      INTEGER  INOUTVEC(*)
      INTEGER  LEN
      INTEGER  DTYPE
      INTEGER  I

      EXTERNAL INCOPR
      EXTERNAL INCOPD
      EXTERNAL INCOPC

C     Prevent compiler warning
      INCOP = MPI_SUCCESS

      IF (LEN .EQ. 0) GO TO 101
      IF (DTYPE .EQ. MPI_INTEGER) THEN
        DO 100 I=1,LEN
          IF (INOUTVEC(I) .GT. INVEC(I)) THEN
            INOUTVEC(I) = INVEC(I)
          ELSE
            INOUTVEC(I) = MPITEST_NUMP + 2
          END IF
 100    CONTINUE
      ELSE IF (DTYPE .EQ. MPI_REAL) THEN
        CALL INCOPR(INVEC, INOUTVEC, LEN)
      ELSE IF (DTYPE .EQ. MPI_DOUBLE_PRECISION) THEN
        CALL INCOPD(INVEC, INOUTVEC, LEN)
      ELSE IF (DTYPE .EQ. MPI_COMPLEX) THEN
        CALL INCOPC(INVEC, INOUTVEC, LEN)
      END IF

 101  CONTINUE
      END


      SUBROUTINE INCOPR(INVEC, INOUTVEC, LEN)
C *************************************************************************
C Non-commutative user Op for REAL.
C *************************************************************************
#include "mpitestf.h"

      REAL     INVEC(*)
      REAL     INOUTVEC(*)
      INTEGER  LEN
      INTEGER  DTYPE
      INTEGER  I

      DO 100 I=1,LEN
        IF (INOUTVEC(I) .GT. INVEC(I)) THEN
          INOUTVEC(I) = INVEC(I)
        ELSE
          INOUTVEC(I) = MPITEST_NUMP + 2
        END IF
 100  CONTINUE

      END

      SUBROUTINE INCOPD(INVEC, INOUTVEC, LEN)
C *************************************************************************
C Non-commutative user Op for DOUBLE PRECISION.
C *************************************************************************
#include "mpitestf.h"

      DOUBLE PRECISION  INVEC(*)
      DOUBLE PRECISION  INOUTVEC(*)
      INTEGER  LEN
      INTEGER  DTYPE
      INTEGER  I

      DO 100 I=1,LEN
        IF (INOUTVEC(I) .GT. INVEC(I)) THEN
          INOUTVEC(I) = INVEC(I)
        ELSE
          INOUTVEC(I) = MPITEST_NUMP + 2
        END IF
 100  CONTINUE

      END

      SUBROUTINE INCOPC(INVEC, INOUTVEC, LEN)
C *************************************************************************
C Non-commutative user Op for COMPLEX.
C *************************************************************************
#include "mpitestf.h"

      COMPLEX  INVEC(*)
      COMPLEX  INOUTVEC(*)
      INTEGER  LEN
      INTEGER  DTYPE
      INTEGER  I

      DO 100 I=1,LEN
        IF (REAL(INOUTVEC(I)) .GT. REAL(INVEC(I))) THEN
          INOUTVEC(I) = INVEC(I)
        ELSE
          INOUTVEC(I) = CMPLX(MPITEST_NUMP + 2)
        END IF
 100  CONTINUE

      END

      SUBROUTINE SET_DEFAULT_OPS(OP_ARRAY, NUMBER)
C ************************************************************************
C  Sets up the default operation array.  Returns the number of default 
C  operations.
C ************************************************************************
#include "mpitestf.h"
      INCLUDE 'mpif.h'
      INCLUDE 'externalf.h'

      INTEGER OP_ARRAY(10)
      INTEGER NUMBER

      EXTERNAL ADDOP
      EXTERNAL INCOP

      CHARACTER*(IOSIZE) INFOBUF
      INTEGER IERR
      INTEGER ERRSIZE
      INTEGER ERR

 99   FORMAT(A,INT_FMT)


      CALL MPI_OP_CREATE(ADDOP, .TRUE., OP_ARRAY(1), IERR)
      IF (IERR .NE. MPI_SUCCESS) THEN
        WRITE(INFOBUF,99) 'MPI_OP_CREATE() returned', IERR
        CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
        INFOBUF=' '
        CALL MPI_ERROR_STRING(IERR, INFOBUF, ERRSIZE, ERR)
        CALL MPITEST_MESSAGE(MPITEST_FATAL, INFOBUF)
      END IF
 
      CALL MPI_OP_CREATE(INCOP, .FALSE., OP_ARRAY(2), IERR)
      IF (IERR .NE. MPI_SUCCESS) THEN
        WRITE(INFOBUF,99) 'MPI_OP_CREATE() returned', IERR
        CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
        INFOBUF=' '
        CALL MPI_ERROR_STRING(IERR, INFOBUF, ERRSIZE, ERR)
        CALL MPITEST_MESSAGE(MPITEST_FATAL, INFOBUF)
      END IF

      NUMBER = 2

      END


      SUBROUTINE HAS_OP(OP, TEST_TYPE, HASOP)
C ************************************************************************
C Determines whether a particular operation may be applied to a particular 
C data type.
C ************************************************************************
#include "mpitestf.h"
      INCLUDE 'externalf.h'

      INTEGER OP
      INTEGER TEST_TYPE
      LOGICAL HASOP

      IF ((TEST_TYPE .EQ. MPITEST_INTEGER) .OR.
     $    (TEST_TYPE .EQ. MPITEST_REAL) .OR.
     $    (TEST_TYPE .EQ. MPITEST_DOUBLE_PRECISION) .OR.
     $    (TEST_TYPE .EQ. MPITEST_COMPLEX)) THEN
        HASOP = .TRUE.
      ELSE
        HASOP = .FALSE.
      END IF

      END


      SUBROUTINE GET_INT_REDUCE(OP_INDEX, NUMP, ANSWER)
C ************************************************************************
C Get result of integer Reduction
C ************************************************************************
C
#include "mpitestf.h"
      INCLUDE 'mpif.h'
      INCLUDE 'externalf.h'

      INTEGER OP_INDEX, NUMP

      EXTERNAL ADDOP
      EXTERNAL INCOP
      INTEGER ADDOP
      INTEGER INCOP
      INTEGER BOGUS
      INTEGER I
      INTEGER INVEC(MAX_RANKS), ANSWER, INOUTVEC(MAX_RANKS)

      IF (NUMP .EQ. 1) THEN
         ANSWER = 1

      ELSE IF (NUMP .GT. 1) THEN
         INVEC(1) = 1
         INOUTVEC(1) = 2

         IF (OP_INDEX .EQ. 1) THEN
            BOGUS = ADDOP(INVEC, INOUTVEC, 1, MPI_INTEGER)
         ELSE IF (OP_INDEX .EQ. 2) THEN
            BOGUS = INCOP(INVEC, INOUTVEC, 1, MPI_INTEGER)
         END IF

         IF (NUMP .GT. 2) THEN
            DO 100 I = 3, NUMP
C
C              Since the operator may not be commutative
C
               INVEC(1) = INOUTVEC(1)

               INOUTVEC(1) = I

               IF (OP_INDEX .EQ. 1) THEN
                  BOGUS = ADDOP(INVEC, INOUTVEC, 1, MPI_INTEGER)
               ELSE IF (OP_INDEX .EQ. 2) THEN
                  BOGUS = INCOP(INVEC, INOUTVEC, 1, MPI_INTEGER)
               END IF

 100        CONTINUE

         END IF

         ANSWER = INOUTVEC(1)

      END IF

      END


      SUBROUTINE GET_REAL_REDUCE(OP_INDEX, NUMP, ANSWER)
C ************************************************************************
C Get result of integer Reduction
C ************************************************************************
C
#include "mpitestf.h"
      INCLUDE 'mpif.h'
      INCLUDE 'externalf.h'

      INTEGER OP_INDEX, NUMP

      EXTERNAL ADDOP
      EXTERNAL INCOP
      INTEGER ADDOP
      INTEGER INCOP
      INTEGER BOGUS
      INTEGER I
      REAL INVEC(MAX_RANKS), ANSWER, INOUTVEC(MAX_RANKS)

      IF (NUMP .EQ. 1) THEN
         ANSWER = 1

      ELSE IF (NUMP .GT. 1) THEN
         INVEC(1) = 1
         INOUTVEC(1) = 2

         IF (OP_INDEX .EQ. 1) THEN
            BOGUS = ADDOP(INVEC, INOUTVEC, 1, MPI_REAL)
         ELSE IF (OP_INDEX .EQ. 2) THEN
            BOGUS = INCOP(INVEC, INOUTVEC, 1, MPI_REAL)
         END IF

         IF (NUMP .GT. 2) THEN
            DO 100 I = 3, NUMP
C
C              Since the operator may not be commutative
C
               INVEC(1) = INOUTVEC(1)

               INOUTVEC(1) = I

               IF (OP_INDEX .EQ. 1) THEN
                  BOGUS = ADDOP(INVEC, INOUTVEC, 1, MPI_REAL)
               ELSE IF (OP_INDEX .EQ. 2) THEN
                  BOGUS = INCOP(INVEC, INOUTVEC, 1, MPI_REAL)
               END IF

 100        CONTINUE

         END IF

         ANSWER = INOUTVEC(1)

      END IF

      END


      SUBROUTINE GET_COMPLEX_REDUCE(OP_INDEX, NUMP,
     &                              ANSWER)
C ************************************************************************
C Get result of integer Reduction
C ************************************************************************
C
#include "mpitestf.h"
      INCLUDE 'mpif.h'
      INCLUDE 'externalf.h'

      INTEGER OP_INDEX, NUMP

      EXTERNAL ADDOP
      EXTERNAL INCOP
      INTEGER ADDOP
      INTEGER INCOP
      INTEGER BOGUS
      INTEGER I
      COMPLEX INVEC(MAX_RANKS), ANSWER, INOUTVEC(MAX_RANKS)

      IF (NUMP .EQ. 1) THEN
         ANSWER = 1

      ELSE IF (NUMP .GT. 1) THEN
         INVEC(1) = 1
         INOUTVEC(1) = 2

         IF (OP_INDEX .EQ. 1) THEN
            BOGUS = ADDOP(INVEC, INOUTVEC, 1, MPI_COMPLEX)
         ELSE IF (OP_INDEX .EQ. 2) THEN
            BOGUS = INCOP(INVEC, INOUTVEC, 1, MPI_COMPLEX)
         END IF

         IF (NUMP .GT. 2) THEN
            DO 100 I = 3, NUMP
C
C              Since the operator may not be commutative
C
               INVEC(1) = INOUTVEC(1)

               INOUTVEC(1) = I

               IF (OP_INDEX .EQ. 1) THEN
                  BOGUS = ADDOP(INVEC, INOUTVEC, 1, MPI_COMPLEX)
               ELSE IF (OP_INDEX .EQ. 2) THEN
                  BOGUS = INCOP(INVEC, INOUTVEC, 1, MPI_COMPLEX)
               END IF

 100        CONTINUE

         END IF

         ANSWER = INOUTVEC(1)

      END IF

      END


      SUBROUTINE GET_DOUBLE_REDUCE(OP_INDEX, NUMP,
     &                              ANSWER)
C ************************************************************************
C Get result of integer Reduction
C ************************************************************************
C
#include "mpitestf.h"
      INCLUDE 'mpif.h'
      INCLUDE 'externalf.h'

      INTEGER OP_INDEX, NUMP

      EXTERNAL ADDOP
      EXTERNAL INCOP
      INTEGER ADDOP
      INTEGER INCOP
      INTEGER BOGUS
      INTEGER I
      DOUBLE PRECISION INVEC(MAX_RANKS), ANSWER, INOUTVEC(MAX_RANKS)

      IF (NUMP .EQ. 1) THEN
         ANSWER = 1

      ELSE IF (NUMP .GT. 1) THEN
         INVEC(1) = 1
         INOUTVEC(1) = 2

         IF (OP_INDEX .EQ. 1) THEN
            BOGUS = ADDOP(INVEC, INOUTVEC, 1, MPI_DOUBLE_PRECISION)
         ELSE IF (OP_INDEX .EQ. 2) THEN
            BOGUS = INCOP(INVEC, INOUTVEC, 1, MPI_DOUBLE_PRECISION)
         END IF

         IF (NUMP .GT. 2) THEN
            DO 100 I = 3, NUMP
C
C              Since the operator may not be commutative
C
               INVEC(1) = INOUTVEC(1)

               INOUTVEC(1) = I

               IF (OP_INDEX .EQ. 1) THEN
                 BOGUS = ADDOP(INVEC, INOUTVEC, 1, MPI_DOUBLE_PRECISION)
               ELSE IF (OP_INDEX .EQ. 2) THEN
                 BOGUS = INCOP(INVEC, INOUTVEC, 1, MPI_DOUBLE_PRECISION)
               END IF

 100        CONTINUE

         END IF

         ANSWER = INOUTVEC(1)

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
C
#include "mpitestf.h"
      INCLUDE 'externalf.h'

      INTEGER OP_INDEX
      INTEGER NUMP, TYPE
      MPITEST_BUF_TYPE ANSWER

      IF (TYPE .EQ. MPITEST_INTEGER) THEN
         CALL GET_INT_REDUCE(OP_INDEX, NUMP, ANSWER)
      ELSE IF (TYPE .EQ. MPITEST_REAL) THEN
         CALL GET_REAL_REDUCE(OP_INDEX, NUMP, ANSWER)
      ELSE IF (TYPE .EQ. MPITEST_DOUBLE_PRECISION) THEN
         CALL GET_DOUBLE_REDUCE(OP_INDEX, NUMP, ANSWER)
      ELSE IF (TYPE .EQ. MPITEST_COMPLEX) THEN
         CALL GET_COMPLEX_REDUCE(OP_INDEX, NUMP, ANSWER)
      END IF

      END

