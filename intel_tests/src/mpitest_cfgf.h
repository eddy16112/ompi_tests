      BLOCK DATA CONFIG
C     ******************************************************************
C     Block data construct to initialize the configuration data.
C
C     Note that the number of tokens up to and including MPITEST_END_TOKEN
C     must be given by the parameter numtok.  This is so that
C     FORTRAN can know how many zeros to add to the end of the array.
C
C     This file should be include'd in one and only one module
C     in the application.  This module initializes the MPITEST_comms()
C     configuration array.
C
C     History :
C     12/07/95        Created       Greg Morrow
C     ******************************************************************
#include "mpitestf.h"

      INTEGER KILO
      PARAMETER(KILO=1024)

      INTEGER KILOMIN8, KILOADD8
      PARAMETER(KILOMIN8 = KILO - 8, KILOADD8 = KILO + 8)

C     Each array is MPITEST_CFGSIZ entries long.  Since Fortran
C     requires an entire array to be initialized, we have to pad
C     the end.  Thus NUMTOKx is the number of elements in the array,
C     so NUMZERx is how much padding is required.
C
C     Use caution when creating these arrays.  The must be valid
C     for the size and type of your machine; needless test errors
C     will occur if (for example) you configure a communicator
C     with more ranks than exist.  This example requires 3 ranks.

      INTEGER NUMTOK1
      INTEGER NUMZER1

#ifdef LARGE_CLUSTER
C Scale things back so we can run on large clusters
      PARAMETER( NUMTOK1 = 18, NUMZER1=MPITEST_CFGSIZ-NUMTOK1)
      DATA MPITEST_COMMS /
     & MPITEST_COMM_WORLD,
     & MPITEST_COMM_DUP,
     & MPITEST_COMM_COMINC, 0, MPITEST_COMM_LASTRNK, 3,
     & MPITEST_COMM_INTER,
     & MPITEST_COMM_CREATE, 
     & MPITEST_COMM_COMINC, 0, 2, 2,
     & MPITEST_COMM_CREATE, 
     & MPITEST_COMM_COMINC, 1, MPITEST_COMM_LASTRNK, 2,
     & MPITEST_END_TOKEN, NUMZER1*0/
#else
      PARAMETER( NUMTOK1 = 40, NUMZER1=MPITEST_CFGSIZ-NUMTOK1)
      DATA MPITEST_COMMS /
     & MPITEST_COMM_WORLD,
     & MPITEST_COMM_SELF,
     & MPITEST_COMM_MERGE,
     & MPITEST_COMM_CREATE, 
     & MPITEST_COMM_COMINC, 0, 2, 2,
     & MPITEST_COMM_CREATE,
     & MPITEST_COMM_COMINC, 1, MPITEST_COMM_LASTRNK, 2,
     & MPITEST_COMM_SPLIT,
     & MPITEST_COMM_COMINC, 1, MPITEST_COMM_LASTRNK, 2,
     & MPITEST_COMM_SPLIT,
     & MPITEST_COMM_RNKLST, 2, 0, 2,
     & MPITEST_COMM_DUP,
     & MPITEST_COMM_COMINC, 0, MPITEST_COMM_LASTRNK, 3,
     & MPITEST_COMM_INTER,
     & MPITEST_COMM_CREATE,
     & MPITEST_COMM_COMINC, 0, 2, 2,
     & MPITEST_COMM_CREATE,
     & MPITEST_COMM_COMINC, 1, MPITEST_COMM_LASTRNK, 2,
     & MPITEST_END_TOKEN, NUMZER1*0/
#endif

C
C     Message lengths.  CAUTION:  the largest message length allowed
C     in the array is MPITEST_BUFF_EXTENT (see foptions.h
C     times MAX_BUFF_SIZE (see mpitestf.h).  To do otherwise
C     will cause data to overflow pre-allocated buffers (we do
C     not try to use some form of malloc in Fortran).
C
      INTEGER NUMTOK2
      INTEGER NUMZER2

#ifdef LARGE_CLUSTER
C Scale things back so we can run on large clusters
      PARAMETER( NUMTOK2 =  3, NUMZER2=MPITEST_CFGSIZ-NUMTOK2)
      DATA MPITEST_MESSAGE_LENGTHS /
     & 48, 
     & 65536, 
     & MPITEST_END_TOKEN, NUMZER2*0 /
#else
      PARAMETER( NUMTOK2 = 15, NUMZER2=MPITEST_CFGSIZ-NUMTOK2)
      DATA MPITEST_MESSAGE_LENGTHS /
     & 0, 
     & MPITEST_MULT_INC, 8, 8000, 10,
     & MPITEST_REPEAT, 320, 8,
     & 48,
     & MPITEST_ADD_INC, KILOMIN8, KILOADD8, 8,
     & 65536,
     & MPITEST_END_TOKEN, NUMZER2*0 /
#endif

C
C     Data types.  Do not use an optional type unless you have
C     allowed it in foptions.h or link errors will occur.
C
      INTEGER NUMTOK3
      INTEGER NUMZER3

#ifdef LARGE_CLUSTER
      PARAMETER( NUMTOK3 = 3, NUMZER3=MPITEST_CFGSIZ-NUMTOK3)
      DATA MPITEST_TYPES /
     & MPITEST_INTEGER, MPITEST_DOUBLE_PRECISION,
     & MPITEST_END_TOKEN, NUMZER3*0 /
#else
      PARAMETER( NUMTOK3 = 13, NUMZER3=MPITEST_CFGSIZ-NUMTOK3)
      DATA MPITEST_TYPES /
     & MPITEST_INTEGER, MPITEST_REAL, MPITEST_DOUBLE_PRECISION,
     & MPITEST_COMPLEX, MPITEST_LOGICAL, MPITEST_CHARACTER,
     & MPITEST_INTEGER1, MPITEST_INTEGER2, MPITEST_INTEGER4, 
     & MPITEST_REAL4, MPITEST_REAL8, 
     & MPITEST_DOUBLE_COMPLEX,
     & MPITEST_END_TOKEN, NUMZER3*0 /
#endif
C
C  Add MPITEST_REAL2, if supported.
C
      END
C     end of 'BLOCK DATA CONFIG'      
