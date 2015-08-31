/*  CUDA utility file */

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "../mpitest_def.h"
#include "cudatest.h"
#include "cuda.h"

static int verbose = 0;
static char *s;
static int pid;
static CUcontext ctx;
static char hostname[256] = "UNKNOWN HOST";

#define OUTPUT_VERBOSE output_verbose
static inline void output_verbose(int level, const char* msg, ...)
{
    va_list ap;

    if (verbose >= level) {
        printf("CUDATEST(host=%s,pid=%d): ", hostname, pid);
        va_start(ap, msg);
        vprintf(msg, ap);
        va_end(ap);
    }
}

/**
 * Initialize so we can run the tests with GPU buffers.  If we get an error
 * during this function, just call exit.  This is called after MPI_Init() has
 * been called.
 */
int CUDATEST_init(int exclusive_only) {
    CUresult res;
    CUdevice cuDev;
    int device, cuDevCount;
    int exclusive = 1;
    char cuName[256];
    int i;
    int attribute, saved_attribute = -1;

    pid = getpid();
    gethostname(hostname, sizeof(hostname));

    /* Really helpful if we want to debug a problem */
    if (NULL != getenv("CUDATEST_DEBUG")) {
        int i = 0;
        printf("PID %d on %s ready for attach\n", getpid(), hostname);
        fflush(stdout);
        while (0 == i) {
            sleep(5);
            printf("PID %d on %s ready for attach\n", getpid(), hostname);
        }
    }

    if (NULL != (s = getenv("CUDATEST_VERBOSE"))) {
        if (!strcmp("", s)) {
            /* If env var is set, but not to a value, default to 1 */
            verbose = 1;
        } else {
            verbose = strtol(s, NULL, 10);
        }
        /* In case someone sets it negative */
        if (verbose < 0) {
            verbose = 0;
        }
    }
    res = cuInit(0);
    if (CUDA_SUCCESS != res) {
        OUTPUT_VERBOSE(0, " cuInit returned %d, CUDATEST_INIT failed.\n", res);
        exit(1);
    } else {
        OUTPUT_VERBOSE(10, "cuInit returned %d\n", res);
    }

    if(CUDA_SUCCESS != cuDeviceGetCount(&cuDevCount)) {
        OUTPUT_VERBOSE(0, "Failed to get the number of devices.  Exiting...\n");
        exit(2);
    }

    /* If any of the devices do not support UVA, then return an error. */
    for (device = 0; device < cuDevCount; device++) {
        if (CUDA_SUCCESS != (res = cuDeviceGet(&cuDev, device))) {
            OUTPUT_VERBOSE(0, " cuDeviceGet returned %d, CUDATEST_INIT failed.\n", res);
            exit(1);
        }
        if (CUDA_SUCCESS != (res = cuDeviceGetAttribute(&attribute, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, cuDev))) {
            OUTPUT_VERBOSE(0, " cuDeviceGetAttribute returned %d, CUDATEST_INIT failed.\n", res);
            exit(1);
        }

        if (!attribute) {
            if (CUDA_SUCCESS != cuDeviceGetName(cuName, 256, cuDev)) {
                strcpy(cuName, "Unknown");
            }
            printf("Device=%d, Name=%s does not support Unified Virtual Addressing.\n", device, cuName);
            return CUDATEST_UNSUPPORTED_DEVICES;
        }
    }

    /* Check to see if GPUs are in exclusive mode.  If they are, then we are
     * running one context per GPU.  Otherwise, we are non-exclusive and
     * we will round robin between whatever GPUs are available.  Also make sure
     * all GPUs are in the same mode.  If not, exit. */
    for (device = 0; device < cuDevCount; device++) {
        if (CUDA_SUCCESS != (res = cuDeviceGet(&cuDev, device))) {
            OUTPUT_VERBOSE(0, " cuDeviceGet returned %d, CUDATEST_INIT failed.\n", res);
            exit(1);
        }
        if (CUDA_SUCCESS != (res = cuDeviceGetAttribute(&attribute, CU_DEVICE_ATTRIBUTE_COMPUTE_MODE, cuDev))) {
            OUTPUT_VERBOSE(0, " cuDeviceGetAttribute returned %d, CUDATEST_INIT failed.\n", res);
            exit(1);
        }
        switch (attribute) {
        case CU_COMPUTEMODE_DEFAULT:
            OUTPUT_VERBOSE(5, "Device=%d has Compute Mode=DEFAULT.\n", cuDev);
            exclusive = 0;
            break;
        case CU_COMPUTEMODE_EXCLUSIVE:
            OUTPUT_VERBOSE(5, "Device=%d has Compute Mode=EXCLUSIVE\n", cuDev);
            break;
        case CU_COMPUTEMODE_PROHIBITED:
            OUTPUT_VERBOSE(5, "Device=%d has Compute Mode=PROHIBITED.\n", cuDev);
            exclusive = 0;
            break;
        case CU_COMPUTEMODE_EXCLUSIVE_PROCESS:
            OUTPUT_VERBOSE(5, "Device=%d has Compute Mode=EXCLUSIVE_PROCESS.\n", cuDev);
            exclusive = 1;
            break;
        default:
            OUTPUT_VERBOSE(0, "Device=%d has Compute Mode=UNKNOWN.  Exiting...\n", cuDev);
            break;
        }
        if (-1 == saved_attribute) {
            saved_attribute = attribute;
        } else {
            if (saved_attribute != attribute) {
                OUTPUT_VERBOSE(0, "Devices have mixed Compute Modes.  Make sure all GPUs have the same mode.  Exiting...\n");
                exit(3);
            }
        }
    }

    if (exclusive_only && !exclusive) {
        OUTPUT_VERBOSE(0, "Devices must all be in Compute Mode=EXCLUSIVE.  Exiting...\n");
        exit(4);
    }

    if (exclusive) {
        /* Now we walk through each device and attempt to create a 
         * context on it.  Since we know we are running in EXCLUSIVE
         * mode, this means only one context per device and allows 
         * us to have things set up prior to calling MPI_Init().
         */
        ctx = NULL;
        for (device = 0; device < cuDevCount; device++) {
            if (CUDA_SUCCESS != (res = cuDeviceGet(&cuDev, device))) {
                OUTPUT_VERBOSE(0, "cuDeviceGet returned %d, exiting.\n", res);
                exit(4);
            }
            if (CUDA_SUCCESS != cuCtxCreate(&ctx, 0, cuDev)) {
                /* Just a warning. The other process must have grabbed this one. */
                OUTPUT_VERBOSE(5, "Failed to get context on device=%d, try again\n", device);
            } else {
                cuDeviceGetName(cuName, 256, cuDev);
                OUTPUT_VERBOSE(1, "Device=%d, name=%s created context.\n", cuDev, cuName);
                break;
            }
            i++;
        }
        if (NULL == ctx) {
            OUTPUT_VERBOSE(0, "Could not create a context on any devices.  Exiting...\n");
            exit(3);
        }
    } else {

#if defined(OPEN_MPI)
        /* Use the local rank setting as this is what we have users do */
        if (NULL != (s = getenv("OMPI_COMM_WORLD_LOCAL_RANK"))) {
            if (!strcmp("", s)) {
                printf("Cannot determine local rank for device selection.  Exiting...\n");
                exit(4);
            } else {
                device = strtol(s, NULL, 10);
                if (device >= cuDevCount) {
                    OUTPUT_VERBOSE(0,"Adjusting device as we are oversubscribing devices: %d -> %d (rank=%s)\n",
                                   device, device % cuDevCount, s);
                    device = device % cuDevCount;
                }
                OUTPUT_VERBOSE(10, "My device is %d (LOCAL_RANK=%s)\n", cuDev, s);
            }
        } else {
            OUTPUT_VERBOSE(0,"Cannot determine local rank for device selection.  Exiting...\n");
            exit(4);
        }
#else
        /* Use the pid to try and round robin over however many GPUs we have.
         * Typically, the pids are contiguous values. This is not perfect,
         * but is just a way to spread the load when testing larger np with
         * fewer devices. */
        device = pid % cuDevCount;
#endif
        if (CUDA_SUCCESS != (res = cuDeviceGet(&cuDev, device))) {
            OUTPUT_VERBOSE(0, "cuDeviceGet returned %d, exiting.\n", res);
            exit(4);
        }

        if (CUDA_SUCCESS != (res = cuCtxCreate(&ctx, 0, cuDev))) {
            OUTPUT_VERBOSE(0, "cuCtxCreate returned %d (cuDev=%d), exiting.\n", res, cuDev);
            exit(4);
        } else {
            cuDeviceGetName(cuName, 256, cuDev);
            OUTPUT_VERBOSE(1, "Device=%d, name=%s created context.\n", cuDev, cuName);
        }

        /* Double check if we can get context because I have seen errors in MPI library */
        if (CUDA_SUCCESS != (res = cuCtxGetCurrent(&ctx))) {
            OUTPUT_VERBOSE(0, "cuCtxGetCurrent returned %d, exiting.\n", res);
            exit(4);
        } else {
            cuDeviceGetName(cuName, 256, cuDev);
            OUTPUT_VERBOSE(1, "Device=%d, name=%s successfully called cuCtxGetCurrent.\n", cuDev, cuName);
        }
    }
    return 0;
}

int CUDATEST_get_gpu_buffer(int buffer_type, int length, CUdeviceptr *buffer, size_t *psize)
/*****************************************************************
  Allocate a GPU buffer of type specified by buffer_type and number
  of elements given by length.
*****************************************************************/
{
    switch (buffer_type) {
    case MPITEST_int:
        *psize = (length + 1) * sizeof(int);
        EXIT_ON_ERROR(cuMemAlloc(buffer, *psize));
        break;
    case MPITEST_short_int:
        *psize = (length + 1) * sizeof(short int);
        EXIT_ON_ERROR(cuMemAlloc((CUdeviceptr *)buffer, *psize));
        break;
    case MPITEST_long:
        *psize = (length + 1) * sizeof(long);
        EXIT_ON_ERROR(cuMemAlloc((CUdeviceptr *)buffer, *psize));
        break;
    case MPITEST_unsigned_short:
        *psize = (length + 1) * sizeof(unsigned short);
        EXIT_ON_ERROR(cuMemAlloc((CUdeviceptr *)buffer, *psize));
        break;
    case MPITEST_unsigned:
        *psize = (length + 1) * sizeof(unsigned);
        EXIT_ON_ERROR(cuMemAlloc((CUdeviceptr *)buffer, *psize));
        break;
    case MPITEST_unsigned_long:
        *psize = (length + 1) * sizeof(unsigned long);
        EXIT_ON_ERROR(cuMemAlloc((CUdeviceptr *)buffer, *psize));
        break;
    case MPITEST_float:
        *psize = (length + 1) * sizeof(float);
        EXIT_ON_ERROR(cuMemAlloc((CUdeviceptr *)buffer, *psize));
        break;
    case MPITEST_double:
        *psize = (length + 1) * sizeof(double);
        EXIT_ON_ERROR(cuMemAlloc((CUdeviceptr *)buffer, *psize));
        break;
    case MPITEST_char:
        *psize = (length + 1) * sizeof(char);
        EXIT_ON_ERROR(cuMemAlloc((CUdeviceptr *)buffer, *psize));
        break;
    case MPITEST_unsigned_char:
        *psize = (length + 1) * sizeof(unsigned char);
    EXIT_ON_ERROR(cuMemAlloc((CUdeviceptr *)buffer, *psize));
        break;
#if MPITEST_longlong_def
    case MPITEST_longlong:
        *psize = (length + 1) * sizeof(long long int);
        EXIT_ON_ERROR(cuMemAlloc((CUdeviceptr *)buffer, *psize));
        break;
#endif
#if MPITEST_long_double_def
    case MPITEST_long_double:
        *psize = (length + 1) * sizeof(long double);
        EXIT_ON_ERROR(cuMemAlloc((CUdeviceptr *)buffer, *psize));
        break;
#endif
    case MPITEST_byte:
        *psize = (length + 1) * sizeof(MPITEST_byte_def);
        EXIT_ON_ERROR(cuMemAlloc((CUdeviceptr *)buffer, *psize));
        break;
    case MPITEST_derived1:
    case MPITEST_derived2:
        *psize = (length + 1) * sizeof(derived1);
        EXIT_ON_ERROR(cuMemAlloc((CUdeviceptr *)buffer, *psize));
        break;
#if MPITEST_2_2_datatype
        case MPITEST_int8_t:
            *psize = (length + 1) * sizeof(int8_t);
            EXIT_ON_ERROR(cuMemAlloc((CUdeviceptr *)buffer, *psize));
            break;
        case MPITEST_uint8_t:
            *psize = (length + 1) * sizeof(uint8_t);
            EXIT_ON_ERROR(cuMemAlloc((CUdeviceptr *)buffer, *psize));
            break;
        case MPITEST_int16_t:
            *psize = (length + 1) * sizeof(int16_t);
            EXIT_ON_ERROR(cuMemAlloc((CUdeviceptr *)buffer, *psize));
            break;
        case MPITEST_uint16_t:
            *psize = (length + 1) * sizeof(uint16_t);
            EXIT_ON_ERROR(cuMemAlloc((CUdeviceptr *)buffer, *psize));
            break;
        case MPITEST_int32_t:
            *psize = (length + 1) * sizeof(int32_t);
            EXIT_ON_ERROR(cuMemAlloc((CUdeviceptr *)buffer, *psize));
            break;
        case MPITEST_uint32_t:
            *psize = (length + 1) * sizeof(uint32_t);
            EXIT_ON_ERROR(cuMemAlloc((CUdeviceptr *)buffer, *psize));
            break;
        case MPITEST_int64_t:
            *psize = (length + 1) * sizeof(int64_t);
            EXIT_ON_ERROR(cuMemAlloc((CUdeviceptr *)buffer, *psize));
            break;
        case MPITEST_uint64_t:
            *psize = (length + 1) * sizeof(uint64_t);
            EXIT_ON_ERROR(cuMemAlloc((CUdeviceptr *)buffer, *psize));
            break;
        case MPITEST_aint:
            *psize = (length + 1) * sizeof(MPI_Aint);
            EXIT_ON_ERROR(cuMemAlloc((CUdeviceptr *)buffer, *psize));
            break;
        case MPITEST_offset:
            *psize = (length + 1) * sizeof(MPI_Offset);
            EXIT_ON_ERROR(cuMemAlloc((CUdeviceptr *)buffer, *psize));
            break;
#endif
    }
    return 0;
}

int CUDATEST_finalize(void) {
    /* Avoid leaking some shared memory segments */
    cuCtxDestroy(ctx);
    return 0;
}
