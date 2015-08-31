C
C     Macro section
C
C     CAUTION.  Some pre-processors substitute everything after the
C     first space after the defined name, which could cause so
C     lines to overflow.  Keep definitions compact!
C
C     Fortran datatype for buffer (send and recv)
C     User must choose the biggest Fortran datatype
C     on the target platforms/host compiler (we do not use malloc).
C
C     For error checking, we also need to know how many bytes
C     one of these occupies.
C
C     See also MAX_BUFF_SIZE in mpitestf.h and data sizes in
C     mpitest_cfgf.h
C
#define MPITEST_BUF_TYPE DOUBLE COMPLEX
#define MPITEST_BUF_EXTENT 16

C     Integer which will hold an address.
#define MPITEST_AINT INTEGER

C     Comment out any of the following defines if each feature
C     is not supported by the host language compiler
C

C     Fortran command line options handling.
C     If iargc() or getargv() are not available in the host
C     compiler, comment this following MACRO out so that the
C     command line option processing will be disabled for the
C     node executable.  The library is easily modified if other
C     means are available to obtain arguments.
C     NOTE: all ranks must be passed the arguments, or customization is
C     required.

C define MPITEST_FCMDLINE 1

C     Fortran optional datatypes
C     REAL*2, *4, *8, DOUBLE COMPLEX, and INTEGER*1, *2, *4
C define MPITEST_FREAL2_DEF          2
#define MPITEST_FREAL4_DEF           3
#define MPITEST_FREAL8_DEF           4
#define MPITEST_FDOUBLE_COMPLEX_DEF  5
#define MPITEST_FINTEGER1_DEF        6
#define MPITEST_FINTEGER2_DEF        7
#define MPITEST_FINTEGER4_DEF        8
#define MPITEST_FINTEGER8_DEF        9

C
C     Fortran display FORMAT for various datatype
C
#define INT_FMT I10
#define REAL_FMT F10.2
#define COMPLEX_FMT F10.2, F10.2
#define DOUBLE_FMT D20.4
#define CHAR_FMT A
#define LOG_FMT  L1

C    Fortran optional supported datatypes
#define INT1_FMT I3
#define INT2_FMT I5
#define INT4_FMT I10
#define INT8_FMT I20

#define REAL2_FMT F10.2
#define REAL4_FMT F10.2
#define REAL8_FMT F10.2

#define DCOMPLEX_FMT D20.4, D20.4

C  This needs to be set up for different platform/host language
#define MPITEST_MAX_INT 2147483647

C  This is the minimum MPI_TAG_UB value required by the MPI Standard
#define MPITEST_TAG_UB_MIN 32767

C Compiler directives:  A few tests that don't use the common library
C (mostly derived datatypes) use these.

C Comment the following #define out to turn off receiving buffer
#define MPITEST_BUFFER_RECV   1

C Comment the following #define out to turn off recv buffer checking
C MPITEST_BUFFER_CHK should not be defined if MPITEST_BUFFER_RECV is
C NOT defined
#define MPITEST_BUFFER_CHK    2

C Comment out the following MACRO to turn off displaying
C the entire sneder & receiver buffer when error occurs
C (output file could be huge! Test may takes a long time to run
C but it is useful for debugging test failures) Off by default.
C #define MPITEST_DISP_BUF      3

C Comment the following #define out to turn off verification for
C the status object returned from MPI_Recv()
#define MPITEST_STATUS_CHK    4

C Determine whether all node should be synchronized after each
C test is done, comment it out if not desired.
#define MPITEST_SYNC          5

C Comment out the following MACRO if user wish to see all
C errors in received data buffer, otherwise, only the first
C error in each transmission will be displayed
#define MPITEST_1ST_ERR       6

