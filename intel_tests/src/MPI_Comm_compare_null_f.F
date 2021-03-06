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
C***********************************************************************
C 
C                    MPI_Comm_compare_null
C 
C This test verifies that the correct result is returned if MPI_Comm_compare()
C is called with MPI_COMM_NULL.
C 
C All other rank(s) will simply do nothing.
C 
C MPI Calls dependencies for this test:
C   MPI_Comm_compare(), MPI_Init(), MPI_Finalize()
C   MPI_Error_string(),
C   [MPI_Comm_compare()]
C***********************************************************************

#include "foptions.h"

      PROGRAM MAIN

#include "mpitestf.h"
      INCLUDE 'externalf.h'
      INCLUDE 'mpif.h'

      CHARACTER  TEST_DESC*(*)
      PARAMETER( TEST_DESC ='MPI_Comm_compare: call with MPI_COMM_NULL')


      INTEGER PASS
      INTEGER FAIL              
C                           counts total number of PASSes/FAILures
      INTEGER IERR
      INTEGER ERR
C                           return value from MPI calls
      INTEGER ERRORCLASS
      INTEGER SIZE 
C                           error class of IERR
      INTEGER RESULT 
C                           result from Comm_compare
      CHARACTER*(IOSIZE)  INFOBUF
C                           buffer for passing messages to MPITEST
      CHARACTER*(MPI_MAX_ERROR_STRING)  ERRORSTRING
C                           string describing error flag
      CHARACTER*32   TESTNAME
C                           name of this test
      INTEGER   COMM1
      INTEGER   COMM2
C                           communicators
C-----------------------------------------------------------------------------
 98   FORMAT(A,A)
 100  FORMAT(A200)
 101  FORMAT(A, A, INT_FMT)
 102  FORMAT(A, INT_FMT, A, INT_FMT)
 103  FORMAT(A, INT_FMT)
 104  FORMAT(A, INT_FMT, A)
C
C  Initialize the MPI environment and test environment.
C
      CALL MPI_INIT(IERR)
      IF (IERR .NE. MPI_SUCCESS) THEN
        WRITE(INFOBUF,103) 'MPI_INIT() returned', IERR 
        CALL MPITEST_MESSAGE(MPITEST_FATAL, INFOBUF)
      END IF

      TESTNAME='MPI_Comm_compare_null'
      CALL MPITEST_INIT(IERR)
      IF (MPITEST_ME .EQ. 0) THEN
        INFOBUF = 'Starting test '//TESTNAME
        CALL MPITEST_MESSAGE(MPITEST_INFO0, INFOBUF)
      END IF

      PASS = 0
      FAIL = 0

C
C Do a quick/safe OMPI test to see if we're checking MPI
C parameters or not
C
      CALL MPI_COMM_RANK(MPI_COMM_NULL, RESULT, IERR)
      IF (RESULT .EQ. MPI_PROC_NULL) THEN
         if (MPITEST_ME .EQ. 0) THEN
            INFOBUF = "MPI is not checking API params; skipping test"
            CALL MPITEST_MESSAGE(MPITEST_INFO0, INFOBUF)
         END IF
         CALL MPI_FINALIZE(IERR)
         CALL EXIT(77)
      END IF

C     Set variables for call
      COMM1 = MPI_COMM_NULL
      COMM2 = MPI_COMM_WORLD

      CALL MPI_COMM_COMPARE(COMM1, COMM2, RESULT, IERR)
      IF (IERR .EQ. MPI_SUCCESS) THEN
          WRITE(INFOBUF,104)
     $     'SUCCESS return code from MPI_COMM_COMPARE 1(',
     $     IERR, ')'
          CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
          ERRORSTRING = ' '
          CALL MPI_ERROR_STRING(IERR, ERRORSTRING, SIZE, ERR)
          WRITE(INFOBUF,100) ERRORSTRING(1:SIZE)
          CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
          FAIL = FAIL + 1
      ELSE
        PASS = PASS + 1
      END IF

C     Set variables for call
      COMM1 = MPI_COMM_WORLD
      COMM2 = MPI_COMM_NULL

      CALL MPI_COMM_COMPARE(COMM1, COMM2, RESULT, IERR)
      IF (IERR .EQ. MPI_SUCCESS) THEN
          WRITE(INFOBUF,104)
     $     'SUCCESS return code from MPI_COMM_COMPARE 2(',
     $     IERR, ')'
          CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
          ERRORSTRING = ' '
          CALL MPI_ERROR_STRING(IERR, ERRORSTRING, SIZE, ERR)
          WRITE(INFOBUF,100) ERRORSTRING(1:SIZE)
          CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
          FAIL = FAIL + 1
      ELSE
        PASS = PASS + 1
      END IF

C     Set variables for call
      COMM1 = MPI_COMM_NULL
      COMM2 = MPI_COMM_SELF

      CALL MPI_COMM_COMPARE(COMM1, COMM2, RESULT, IERR)
      IF (IERR .EQ. MPI_SUCCESS) THEN
          WRITE(INFOBUF,104)
     $     'SUCCESS-Zero return code from MPI_COMM_COMPARE 3(',
     $     IERR, ')'
          CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
          ERRORSTRING = ' '
          CALL MPI_ERROR_STRING(IERR, ERRORSTRING, SIZE, ERR)
          WRITE(INFOBUF,100) ERRORSTRING(1:SIZE)
          CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
          FAIL = FAIL + 1
      ELSE
        PASS = PASS + 1
      END IF

C     Set variables for call
      COMM1 = MPI_COMM_SELF
      COMM2 = MPI_COMM_NULL

      CALL MPI_COMM_COMPARE(COMM1, COMM2, RESULT, IERR)
      IF (IERR .EQ. MPI_SUCCESS) THEN
          WRITE(INFOBUF,104)
     $     'SUCCESS return code from MPI_COMM_COMPARE 4(',
     $     IERR, ')'
          CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
          ERRORSTRING = ' '
          CALL MPI_ERROR_STRING(IERR, ERRORSTRING, SIZE, ERR)
          WRITE(INFOBUF,100) ERRORSTRING(1:SIZE)
          CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
          FAIL = FAIL + 1
      ELSE
        PASS = PASS + 1
      END IF


C     Set variables for call
      COMM1 = MPI_COMM_NULL
      COMM2 = MPI_COMM_NULL

      CALL MPI_COMM_COMPARE(COMM1, COMM2, RESULT, IERR)
      IF (IERR .EQ. MPI_SUCCESS) THEN
          WRITE(INFOBUF,104)
     $     'SUCCESS return code from MPI_COMM_COMPARE 5(',
     $     IERR, ')'
          CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
          ERRORSTRING = ' '
          CALL MPI_ERROR_STRING(IERR, ERRORSTRING, SIZE, ERR)
          WRITE(INFOBUF,100) ERRORSTRING(1:SIZE)
          CALL MPITEST_MESSAGE(MPITEST_NONFATAL, INFOBUF)
          FAIL = FAIL + 1
      ELSE
        PASS = PASS + 1
      END IF



      CALL MPITEST_REPORT(PASS, FAIL, 0, TESTNAME)
      CALL MPI_FINALIZE(IERR)
      IF (FAIL .EQ. 77) THEN
          FAIL = FAIL + 1
      END IF
      CALL EXIT(FAIL)

      END
