#include "cuda.h"

/* Always exit when we stumble into an error.  Arbitrarily
 * pick an exit code of 11. */
#define EXIT_ON_ERROR(func)                             \
    { CUresult res;                                     \
      res = func;                                       \
      if (CUDA_SUCCESS != res) {                        \
          printf("%s returned error=%d\n", #func, res); \
          exit(11);                                     \
      }                                                 \
    }                             

int CUDATEST_init(int exclusive_only);
int CUDATEST_finalize(void);
int CUDATEST_get_gpu_buffer( int buffer_type, int length, CUdeviceptr *buffer, size_t *psize);

#define CUDATEST_UNSUPPORTED_DEVICES 1
