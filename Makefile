MPI_ROOT = /home/wwu12/ompi/build-gpu

CC = $(MPI_ROOT)/bin/mpicc

SRC:= \
	to_self.c	\
	ddt_send_recv.c	\
	datatype_send_recv.c	\
	vector_send_recv.c	\
	cuda_send_recv.c	\
	hello_world.c		\

CFLAGS = -g
INC = -I$(MPI_ROOT)/include -I/mnt/sw/cuda/include
LIB = -I$(MPI_ROOT)/lib -lmpi -L/mnt/sw/cuda/lib64 -lcudart

.PHONY: all clean

all: to_self ddt_send_recv datatype_send_recv vector_send_recv cuda_send_recv hello_world

%.o: %.c
	$(CC) $(CFLAGS) $(INC) -c $< -o $@

to_self: to_self.o
	$(CC) $(CFLAGS) $(INC) $(LIB) $^ -o $@

ddt_send_recv: ddt_send_recv.o
	$(CC) $(CFLAGS) $(INC) $(LIB) $^ -o $@

datatype_send_recv: datatype_send_recv.o
	$(CC) $(CFLAGS) $(INC) $(LIB) $^ -o $@
	
vector_send_recv: vector_send_recv.o
	$(CC) $(CFLAGS) $(INC) $(LIB) $^ -o $@
	
cuda_send_recv: cuda_send_recv.o
	$(CC) $(CFLAGS) $(INC) $(LIB) $^ -o $@

hello_world: hello_world.o
	$(CC) $(CFLAGS) $(INC) $(LIB) $^ -o $@

clean:
	rm -f *.o 
	rm -f to_self
