#MPI_ROOT = /home-2/wwu/build-gpu-rebase
MPI_ROOT = /home-2/wwu/build-lx
#MPI_ROOT = /home-2/wwu/build-test
#MPI_ROOT = /home/wwu12/ompi/build-gpu
CUDA_ROOT = /cm/shared/apps/cuda75/toolkit/7.5.18
#CUDA_ROOT = /mnt/sw/cuda

CC = $(MPI_ROOT)/bin/mpicc

SRC:= \
	to_self.c	\
	ddt_send_recv.c	\
	datatype_send_recv.c	\
	vector_send_recv.c	\
	cuda_send_recv.c	\
	hello_world.c		\
	bcast.c	\
	reduce.c \
	allreduce.c \
	send_m.c	\
	mybcast.c	\
    multi_isend.c   \

CFLAGS = -g
INC = -I$(MPI_ROOT)/include -I$(CUDA_ROOT)/include
LIB = -ldl -L$(MPI_ROOT)/lib -lmpi -L$(CUDA_ROOT)/lib64 -lcudart -lcuda

.PHONY: all clean

all: ddt_send_recv datatype_send_recv vector_send_recv cuda_send_recv bcast bcast_node bcast_random reduce allreduce send_m mybcast multi_isend

%.o: %.c
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

ddt_send_recv: ddt_send_recv.o
	$(CC) $(CFLAGS) $(INC) $(LIB) $^ -o $@

datatype_send_recv: datatype_send_recv.o
	$(CC) $(CFLAGS) $(INC) $(LIB) $^ -o $@
	
vector_send_recv: vector_send_recv.o
	$(CC) $(CFLAGS) $(INC) $(LIB) $^ -o $@
	
cuda_send_recv: cuda_send_recv.o
	$(CC) $(CFLAGS) $(INC) $(LIB) $^ -o $@
	
bcast: bcast.o
	$(CC) $(CFLAGS) $(INC) $(LIB) $^ -o $@

bcast_node: bcast_node.o
	$(CC) $(CFLAGS) $(INC) $(LIB) $^ -o $@
	
bcast_random: bcast_random.o
	$(CC) $(CFLAGS) $(INC) $(LIB) $^ -o $@

reduce: reduce.o
	$(CC) $(CFLAGS) $(INC) $(LIB) $^ -o $@
	
allreduce: allreduce.o
	$(CC) $(CFLAGS) $(INC) $(LIB) $^ -o $@
	
send_m: send_m.o
	$(CC) $(CFLAGS) $(INC) $(LIB) $^ -o $@

mybcast: mybcast.o
	$(CC) $(CFLAGS) $(INC) $(LIB) $^ -o $@

multi_isend: multi_isend.o
	$(CC) $(CFLAGS) $(INC) $(LIB) $^ -o $@

clean:
	rm -f *.o 
	rm -f to_self
	rm ddt_send_recv datatype_send_recv vector_send_recv cuda_send_recv bcast reduce allreduce send_m mybcast multi_isend
