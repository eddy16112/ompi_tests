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
C                   Test for MPI_win_get_name()
C 
C This test calls MPI_win_get_name() on each rank and prints it.
C 
C You must manually verify that the name is correct for each rank.
C 
C Test history:
C    1  05/02/96     gt   Created
C
C **********************************************************************
#include "foptions.h"


      PROGRAM MAIN

#include "mpitestf.h"
      INCLUDE 'externalf.h'
      INCLUDE 'mpif.h'

C                           counts total number of PASSes/FAILures/VERIFY
      INTEGER LOOP_CNT
      INTEGER FAIL              
C                           return value from MPI calls
      INTEGER IERR
      INTEGER ERR
C                           error class of IERR
      INTEGER ERRORCLASS
      INTEGER LEN           
      INTEGER ERRSIZE 
      INTEGER I
      INTEGER WIN

      CHARACTER*(IOSIZE)  INFOBUF
      CHARACTER*(MPI_MAX_OBJECT_NAME) NAME
      CHARACTER*32   TESTNAME
      CHARACTER*(MPI_MAX_OBJECT_NAME) WIN_NAME
      INTEGER(MPI_ADDRESS_KIND) SIZE

 97   FORMAT(A,A,A,A,A)
 98   FORMAT(A,INT_FMT,A)
 99   FORMAT(A,INT_FMT)
 100  FORMAT(A200)


C
C  Initialize the MPI environment and test environment.
C
      CALL MPI_INIT(IERR)
      IF (IERR .NE. MPI_SUCCESS) THEN
        WRITE(INFOBUF,99) 'MPI_INIT() returned', IERR 
        CALL MPITEST_MESSAGE(MPITEST_FATAL, INFOBUF)
      END IF

      TESTNAME='MPI_win_get_name'
      CALL MPITEST_INIT(IERR)
      IF (MPITEST_ME .EQ. 0) THEN
        INFOBUF = 'Starting test '//TESTNAME
        CALL MPITEST_MESSAGE(MPITEST_INFO0, INFOBUF)
      END IF

      LOOP_CNT = 1
      FAIL = 0

C There are no predefined windows, so make one
      SIZE = 1
      CALL MPI_WIN_CREATE(INFOBUF, SIZE, 1, MPI_INFO_NULL, 
     $    MPI_COMM_WORLD, WIN, IERR)
      DO 200 I=0, MPITEST_NUMP-1
        IF (I .EQ. MPITEST_ME) THEN
          IF (IERR .NE. MPI_SUCCESS) THEN
            WRITE(INFOBUF, 99) 'MPI_WIN_CREATE RETURNED', IERR
            CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
            INFOBUF = ' '
            CALL MPI_ERROR_STRING(IERR, INFOBUF, ERRSIZE, ERR)
            CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
            FAIL = FAIL + 1
          END IF
        END IF
        CALL MPI_BARRIER(MPI_COMM_WORLD, IERR)
 200  CONTINUE

C Set a name on the window
      LOOP_CNT = LOOP_CNT + 1
      WIN_NAME = 'Test window name'
      CALL MPI_WIN_SET_NAME(WIN, WIN_NAME, IERR)
      DO 201 I=0, MPITEST_NUMP-1
        IF (I .EQ. MPITEST_ME) THEN
          IF (IERR .NE. MPI_SUCCESS) THEN
            WRITE(INFOBUF, 99) 'MPI_WIN_SET_NAME RETURNED', IERR
            CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
            INFOBUF = ' '
            CALL MPI_ERROR_STRING(IERR, INFOBUF, ERRSIZE, ERR)
            CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
            FAIL = FAIL + 1
          END IF
        END IF
        CALL MPI_BARRIER(MPI_COMM_WORLD, IERR)
 201    CONTINUE

C Now get the name and check it

      LOOP_CNT = LOOP_CNT + 1
      NAME = ' '
      CALL MPI_WIN_GET_NAME(WIN, NAME, LEN, IERR)
      DO 202 I=0, MPITEST_NUMP-1
        IF (I .EQ. MPITEST_ME) THEN
          IF (IERR .NE. MPI_SUCCESS) THEN
            WRITE(INFOBUF, 99) 'MPI_WIN_GET_NAME RETURNED', IERR
            CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
            INFOBUF = ' '
            CALL MPI_ERROR_STRING(IERR, INFOBUF, ERRSIZE, ERR)
            CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
            FAIL = FAIL + 1
          ELSE IF (LEN .GT. MPI_MAX_OBJECT_NAME) THEN
            WRITE(INFOBUF, 98) 'Returned length =', LEN,
     $             ' greater than MPI_MAX_OBJECT_NAME'
            CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
            FAIL = FAIL + 1
          ELSE IF (NAME .NE. WIN_NAME) THEN
            WRITE(INFOBUF, 97) 'Got wrong window name back ("', NAME,
     $             '", expected "', WIN_NAME, '")'
            CALL MPITEST_MESSAGE(MPITEST_VERIFY, INFOBUF)
          END IF
             
        END IF
        CALL MPI_BARRIER(MPI_COMM_WORLD, IERR)
 202  CONTINUE

C
C  Report overall results
C
      CALL MPITEST_REPORT(LOOP_CNT-FAIL, FAIL, 0, TESTNAME)
      CALL MPI_WIN_FREE(WIN, IERR)

      CALL MPI_FINALIZE(IERR)
      IF (FAIL .EQ. 77) THEN
          FAIL = FAIL + 1
      END IF
      CALL EXIT(FAIL)
      END
