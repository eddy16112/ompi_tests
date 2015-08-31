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
/*
 * History:
 *  1/15/96    gt    Added additional data types.
 *  2/15/96    gt    Added MPITEST_comm_type_self, _merge.
 *  2/20/96    gt    Added derived data types.
 *
 */

#ifndef MPITEST_DEF_H_INCLUDED
#define MPITEST_DEF_H_INCLUDED


#include <stdint.h>
#include <mpi.h>


/* Optional data types.  Set to 1 if implementation under test supports
   long long int and/or long double */
#define MPITEST_longlong_def 0
#define MPITEST_long_double_def 1

/* Set to 1 if the implementation under test supports MPI 2.2
   datatypes */
#define MPITEST_2_2_datatype 0
#if defined(OPEN_MPI)
#    if (OMPI_MAJOR_VERSION > 1) || (OMPI_MAJOR_VERSION == 1 && (OMPI_MINOR_VERSION >= 6 || (OMPI_MINOR_VERSION == 5 && OMPI_RELEASE_VERSION >= 4)))
#        undef MPITEST_2_2_datatype
#        define MPITEST_2_2_datatype 1
#    endif
#endif
#if MPI_VERSION > 2 || (MPI_VERSION == 2 && MPI_SUBVERSION >= 2)
#    undef MPITEST_2_2_datatype
#    define MPITEST_2_2_datatype 1
#endif

/* C data type that corresponds to MPI_byte */
#define MPITEST_byte_def unsigned char

/* The default length of specification arrays.  This number should be
   set large enough to accommodate all foreseeable default specs. */
#define MPITEST_CONFIG_ARRAY_SIZE 128

/* The global end token.  This delimits the end of the 
   specification arrays.  */
#define MPITEST_END_TOKEN -999999


/* Magic numbers for message length specification */ 
#define MPITEST_ADD_INC -1
#define MPITEST_MULT_INC -2
#define MPITEST_REPEAT -3
#define MPITEST_MIN_LENGTH -3


/* communicator size tokens */
#define MPITEST_comm_one -1
#define MPITEST_comm_half_of_all -2
#define MPITEST_comm_all_but_one -3
#define MPITEST_comm_all -4
#define MPITEST_comm_last_rank -5
#define MPITEST_comm_inc -6
#define MPITEST_node_list -7
#define MPITEST_comm_size_min -7


/* communicator type tokens */
#define MPITEST_comm_type_world -10
#define MPITEST_comm_type_self -11
#define MPITEST_comm_type_create -12
#define MPITEST_comm_type_split -13
#define MPITEST_comm_type_dup -14
#define MPITEST_comm_type_inter -15
#define MPITEST_comm_type_merge -16
#define MPITEST_comm_type_min -17


/* These defines are indices into the MPITEST_mpi_datatypes[] array defined
   in mpitest_user.h.  Append entries here and initialize in mpitest_user.h,
   but do not modify these or you may cause unnecessary failures.  */
#define MPITEST_int 0
#define MPITEST_short_int 1
#define MPITEST_long 2
#define MPITEST_unsigned_short 3
#define MPITEST_unsigned 4
#define MPITEST_unsigned_long 5
#define MPITEST_float 6
#define MPITEST_double 7
#define MPITEST_char 8
#define MPITEST_unsigned_char 9
#define MPITEST_longlong 10
#define MPITEST_long_double 11
#define MPITEST_byte 12
#define MPITEST_derived1 13
#define MPITEST_derived2 14
/* MPI 2.2 datatypes */
#define MPITEST_int8_t   15
#define MPITEST_uint8_t  16
#define MPITEST_int16_t  17
#define MPITEST_uint16_t 18
#define MPITEST_int32_t  19
#define MPITEST_uint32_t 20
#define MPITEST_int64_t  21
#define MPITEST_uint64_t 22
#define MPITEST_aint     23
#define MPITEST_offset   24
#define MPITEST_datatype_max 24


/* This magic number is used to indicate that the corresponding parameter
   has been given on the command line */
#define MPITEST_COMMAND_LINE -999

/* This definition is used to create a global "not assigned" value
   which can be assigned to command-line parameters.  This value
   is assigned if there is no specification of the given parameter on
   the command line. */
#define MPITEST_NOT_ASSIGNED -111

/* Max ranks */
#ifdef LARGE_CLUSTER
/* Increase count so we can run on large clusters */
#define MPITEST_MAX_RANKS  1024
#else
#define MPITEST_MAX_RANKS  256
#endif

/*  For future enhancement  */
#define MPITEST_COMM_WORLD MPI_COMM_WORLD

/* Structure used for data assignments.  This structure should contain
   a member for each data type which may be used by a test. */

struct dataTemplate
{
  int Int;
  short int ShortInt;
  long int Long;
  unsigned short UnsignedShort;
  unsigned  Unsigned;
  unsigned long UnsignedLong;
  float Float;
  double Double;
  char Char;
  unsigned char UnsignedChar;
#if MPITEST_longlong_def
  long long int LongLong;
#endif
#if MPITEST_long_double_def
  long double LongDouble;
#endif
  MPITEST_byte_def Byte;
#if MPITEST_2_2_datatype
  int8_t         int8;
  uint8_t        uint8;
  int16_t        int16;
  uint16_t       uint16;
  int32_t        int32;
  uint32_t       uint32;
  int64_t        int64;
  uint64_t       uint64;
  MPI_Aint       aint;
  MPI_Offset     offset;
#endif
};

/* Structures used for derived data types. */

typedef struct derived1
{
  int            Int[2];
  short int      ShortInt[2];
  long int       Long[2];
  unsigned short UnsignedShort[2];
  unsigned       Unsigned[2];
  unsigned long  UnsignedLong[2];
  float          Float[2];
  char           Char[2];
  double         Double[2];
  unsigned char  UnsignedChar[2];
#if MPITEST_longlong_def
  long long int   LongLong[2];
#endif
#if MPITEST_long_double_def
  long double    LongDouble[2];
#endif
#if MPITEST_2_2_datatype
  int8_t         int8[2];
  uint8_t        uint8[2];
  int16_t        int16[2];
  uint16_t       uint16[2];
  int32_t        int32[2];
  uint32_t       uint32[2];
  int64_t        int64[2];
  uint64_t       uint64[2];
  MPI_Aint       aint[2];
  MPI_Offset     offset[2];
#endif
} derived1;


#endif
