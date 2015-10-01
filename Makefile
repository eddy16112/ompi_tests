MPI_ROOT = /home-2/wwu/build-gpu

CC = $(MPI_ROOT)/bin/mpicc

SRC:= \
	to_self.c	\
	ddt_send_recv.c	\
	datatype_send_recv.c	\
	vector_send_recv.c	\
	cuda_send_recv.c	\
	hello_world.c		\

CFLAGS = -g
INC = -I$(MPI_ROOT)/include -I/shared/apps/cuda/CUDA-v7.5.18/include
LIB = -I$(MPI_ROOT)/lib -lmpi -L/shared/apps/cuda/CUDA-v7.5.18/lib64 -lcudart

.PHONY: all clean

all: ddt_send_recv datatype_send_recv vector_send_recv cuda_send_recv

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

clean:
	rm -f *.o 
	rm -f to_self
