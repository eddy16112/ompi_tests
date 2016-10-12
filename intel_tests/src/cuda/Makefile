SHELL	= /bin/sh
# Allow environment to override setting of CUDA_HOME
ifndef CUDA_HOME
  CUDA_HOME:=/cm/shared/apps/cuda75/toolkit/7.5.18
endif
CC	= mpicc
CFLAGS = -I.. -g -I$(CUDA_HOME)/include -L$(CUDA_HOME)/lib64 -lcudart -lcuda -Wl,-rpath,$(CUDA_HOME)/lib64 -lm
CPPFLAGS= -Isrc 
LDLIBS	=
INTELDIR = ..

all: $(INTELDIR)/libmpitest.o libcudatest.o bsend isend send ssend coll

clean:
	$(RM) *_c *.o core* *~

$(INTELDIR)/libmpitest.o: $(INTELDIR)/libmpitest.c
	(cd ..; gmake libmpitest.o)

bsend: MPI_Bsend_init_rtoa_cuda_c MPI_Bsend_init_overtake_cuda_c
isend: MPI_Isend_ator_cuda_c MPI_Isend_ator2_cuda_c MPI_Isend_fairness_cuda_c MPI_Isend_flood_cuda_c MPI_Isend_overtake2_cuda_c
send: MPI_Send_ator_cuda_c MPI_Send_ator2_cuda_c MPI_Send_init_ator_cuda_c MPI_Send_init_overtake_cuda_c MPI_Send_overtake_cuda_c MPI_Send_rtoa_cuda_c MPI_Send_rtoa_ridiculously_simple_cuda_c MPI_Waitall_cuda_c
ssend: MPI_Ssend_overtake_cuda_c
coll: MPI_Alltoall_cuda_c MPI_Alltoallv_cuda_c MPI_Scatter_cuda_c MPI_Scatterv_cuda_c MPI_Bcast_cuda_c MPI_Gather_cuda_c MPI_Gatherv_cuda_c MPI_Reduce_cuda_c

libcudatest.o: libcudatest.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CUDAFLAGS) -c -o $@  $^ $(LDLIBS)

%cuda_c: %cuda_c.c $(INTELDIR)/libmpitest.o libcudatest.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CUDAFLAGS) -o $@ $^ $(LDLIBS)

mpicudatest.tar:
	cd ../..; \
	tar cvf mpicudatest.tar Makefile cuda_c check_results.pl; \
	tar rvf mpicudatest.tar src/libmpitest.c src/mpitest_cfg.h src/mpitest_def.h src/mpitest.h; \
	tar rvf mpicudatest.tar src/Makefile; \
	tar rvf mpicudatest.tar src/cuda/libcudatest.c src/cuda/cudatest.h src/cuda/Makefile; \
	tar rvf mpicudatest.tar `ls src/cuda/*_cuda_c.c`
	@echo "Tar done.  File is mpicudatest.tar"



