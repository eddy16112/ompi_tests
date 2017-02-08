#!/bin/bash

module purge

export PATH=/home-2/wwu/build-lx/bin/:$PATH

export LD_LIBRARY_PATH=/home-2/wwu/build-lx/lib/:$LD_LIBRARY_PATH

module add cuda/7.5.18
module add slurm

mpirun --map-by core ./bcast -l 16000000
#mpirun --mca btl_smcuda_cuda_max_send_size 50000000 --mca btl_smcuda_cuda_ddt_pipeline_size 4000000 ./datatype_send_recv -l 2000 -b 2000 -s 4000 -i 2
#mpirun --map-by core --mca btl_sm_with_cma 1 --mca btl sm,self,openib bcast -l 160000000
