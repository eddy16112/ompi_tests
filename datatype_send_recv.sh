#!/bin/bash

module purge

export PATH=/home-2/wwu/build-gpu-new/bin/:$PATH

export LD_LIBRARY_PATH=/home-2/wwu/build-gpu-new/lib/:$LD_LIBRARY_PATH

module add cuda/7.5.18
module add slurm


#mpirun --mca btl_smcuda_cuda_max_send_size 50000000 --mca btl_smcuda_cuda_ddt_pipeline_size 4000000 ./datatype_send_recv -l 2000 -b 2000 -s 4000 -i 2
mpirun --mca btl openib --mca btl_openib_max_send_size 2000032 ./datatype_send_recv -l 1000 -b 1000 -s 3000 -i 10
