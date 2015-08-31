/*
 * Copyright (c) 2011      Oracle and/or its affiliates.  All rights reserved.
 */

/*
 * C programs can use preprocessor symbols to deduce OMPI and MPI version
 * numbers and thereby conditionally compile code that involves MPI 2.2
 * data types.
 *
 * Since Fortran compilers do not similarly integrate preprocessing,
 * there are no corresponding Fortran preprocessor symbols for OMPI or
 * MPI version numbers.  (There are Fortran MPI version parameters,
 * but they can't help us with conditional compilation.)
 *
 * In this short program, we use C preprocessor information to generate
 * a Fortran preprocessor symbol.
 */

#include <stdio.h>
#include "mpitest_def.h"

int main(int argc, char **argv) {
#if MPITEST_2_2_datatype
    printf("#define MPITEST_MPI2_2 1\n");
#endif
    return 0;
}

