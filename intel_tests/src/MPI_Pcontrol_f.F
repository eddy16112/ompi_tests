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
C                   Test for MPI_Pcontrol()
C 
C This test calls MPI_Pcontrol(), with various values.
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

C                           counts total number of PASSes/FAILures
      INTEGER PASS
      INTEGER FAIL              
C                           return value from MPI calls
      INTEGER IERR
      INTEGER ERR
      INTEGER ERRSIZE
C                           error class of IERR
      INTEGER ERRORCLASS
      INTEGER LEN           
      INTEGER SIZE 

      CHARACTER*(IOSIZE)  INFOBUF
      CHARACTER*(MPI_MAX_ERROR_STRING)  ERRORSTRING
      CHARACTER*32   TESTNAME


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

      TESTNAME='MPI_Pcontrol'
      CALL MPITEST_INIT(IERR)
      IF (MPITEST_ME .EQ. 0) THEN
        INFOBUF = 'Starting test '//TESTNAME
        CALL MPITEST_MESSAGE(MPITEST_INFO0, INFOBUF)
      END IF

      PASS = 0
      FAIL = 0

      CALL MPI_PCONTROL(0, IERR)
      IF (IERR .NE. MPI_SUCCESS) THEN
        WRITE(INFOBUF, 99) 'MPI_PCONTROL(0) RETURNED', IERR
        CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
        INFOBUF = ' '
        CALL MPI_ERROR_STRING(IERR, INFOBUF, ERRSIZE, ERR)
        CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
        FAIL = FAIL + 1
      ELSE
        PASS = PASS + 1
      END IF

      CALL MPI_PCONTROL(1, IERR)
      IF (IERR .NE. MPI_SUCCESS) THEN
        WRITE(INFOBUF, 99) 'MPI_PCONTROL(1) RETURNED', IERR
        CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
        INFOBUF = ' '
        CALL MPI_ERROR_STRING(IERR, INFOBUF, ERRSIZE, ERR)
        CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
        FAIL = FAIL + 1
      ELSE
        PASS = PASS + 1
      END IF

      CALL MPI_PCONTROL(2, IERR)
      IF (IERR .NE. MPI_SUCCESS) THEN
        WRITE(INFOBUF, 99) 'MPI_PCONTROL(2) RETURNED', IERR
        CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
        INFOBUF = ' '
        CALL MPI_ERROR_STRING(IERR, INFOBUF, ERRSIZE, ERR)
        CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
        FAIL = FAIL + 1
      ELSE
        PASS = PASS + 1
      END IF

      CALL MPI_PCONTROL(3, IERR)
      IF (IERR .NE. MPI_SUCCESS) THEN
        WRITE(INFOBUF, 99) 'MPI_PCONTROL(3) RETURNED', IERR
        CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
        INFOBUF = ' '
        CALL MPI_ERROR_STRING(IERR, INFOBUF, ERRSIZE, ERR)
        CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
        FAIL = FAIL + 1
      ELSE
        PASS = PASS + 1
      END IF

C
C  Report overall results
C
      CALL MPITEST_REPORT(PASS, FAIL, 0, TESTNAME)

      CALL MPI_FINALIZE(IERR)
      IF(FAIL .EQ. 77) THEN
          FAIL = FAIL + 1
      END IF
      CALL EXIT(FAIL)


      END
