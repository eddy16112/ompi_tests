.PHONY:	clean phony
.DONE:
.INIT:

SHELL	= /bin/sh
CC	= mpicc
FC	= mpif77
CFLAGS	= -g -O0 -I/mnt/sw/cuda/include -L/mnt/sw/cuda/lib64 -lcudart
FFLAGS	= -g -Isrc -I.
# We don't need to see all the deprecated warnings -- we *know* we're
# testing deprecated interfaces.
CPPFLAGS= -Isrc "-DOMPI_WANT_MPI_INTERFACE_WARNING=0"
LDLIBS	= -lm

all: test coll wait datatype

clean:
	$(RM) *_c *_f *.o core*

libmpitest.o: libmpitest.c
libmpitestf.o: libmpitestf.F mpitest_deff.h

mpitest_deff.h: mpitest_deff.c mpitest_def.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -o mpitest_deff mpitest_deff.c
	rm -f $@
	mpirun --quiet ./mpitest_deff > $@
	rm -f mpitest_deff

coll: coll_c coll_f
coll_c: MPI_Allgather_c MPI_Allgatherv_c MPI_Allreduce_c MPI_Allreduce_loc_c  MPI_Allreduce_user_c MPI_Alltoall_c MPI_Alltoallv_c MPI_Barrier_c MPI_Bcast_c MPI_Gather_c MPI_Gatherv_c MPI_Reduce_c MPI_Reduce_loc_c MPI_Reduce_scatter_c MPI_Reduce_scatter_loc_c MPI_Reduce_scatter_user_c MPI_Reduce_user_c MPI_Scan_c MPI_Scan_loc_c MPI_Scan_user_c MPI_Scatter_c MPI_Scatterv_c
coll_f: MPI_Allgather_f MPI_Allgatherv_f MPI_Allreduce_f MPI_Allreduce_loc_f  MPI_Allreduce_user_f MPI_Alltoall_f MPI_Alltoallv_f MPI_Barrier_f MPI_Bcast_f MPI_Gather_f MPI_Gatherv_f MPI_Reduce_f MPI_Reduce_loc_f MPI_Reduce_scatter_f MPI_Reduce_scatter_loc_f MPI_Reduce_scatter_user_f MPI_Reduce_user_f MPI_Scan_f MPI_Scan_loc_f MPI_Scan_user_f MPI_Scatter_f MPI_Scatterv_f

test: test_c test_f
test_c: MPI_Test_c MPI_Test_p_c MPI_Testall_c MPI_Testall_p_c MPI_Testany_c MPI_Testany_p_c MPI_Testsome_c MPI_Testsome_p_c
test_f: MPI_Test_f MPI_Test_p_f MPI_Testall_f MPI_Testall_p_f MPI_Testany_f MPI_Testany_p_f MPI_Testsome_f MPI_Testsome_p_f

wait: wait_c wait_f
wait_c: MPI_Waitall_c MPI_Waitall_p_c MPI_Waitany_c MPI_Waitany_p_c MPI_Waitsome_c MPI_Waitsome_p_c
wait_f: MPI_Waitall_f MPI_Waitall_p_f MPI_Waitany_f MPI_Waitany_p_f MPI_Waitsome_f MPI_Waitsome_p_f

datatype:datatype_c datatype_f
datatype_c: MPI_Get_elements_basic_type_c MPI_Pack_basic_type_c MPI_Pack_size_types_c MPI_Pack_user_type_c \
    MPI_Type_contiguous_basic_c MPI_Type_contiguous_idispls_c MPI_Type_contiguous_types_c \
    MPI_Type_extent_MPI_LB_UB_c MPI_Type_extent_types_c MPI_Type_free_pending_msg_c MPI_Type_free_types_c \
    MPI_Type_hindexed_basic_c MPI_Type_hindexed_blklen_c MPI_Type_hindexed_displs_c MPI_Type_hindexed_types_c \
    MPI_Type_hvector_basic_c MPI_Type_hvector_blklen_c MPI_Type_hvector_stride_c MPI_Type_hvector_types_c \
    MPI_Type_indexed_basic_c MPI_Type_indexed_blklen_c MPI_Type_indexed_displs_c MPI_Type_indexed_types_c \
    MPI_Type_lb_2MPI_LB_c MPI_Type_lb_MPI_LB_c MPI_Type_lb_neg_displ_c MPI_Type_lb_pos_displ_c \
    MPI_Type_size_MPI_LB_UB_c MPI_Type_size_basic_c MPI_Type_size_types_c MPI_Type_struct_basic_c \
    MPI_Type_struct_blklen_c MPI_Type_struct_displs_c MPI_Type_struct_types_c MPI_Type_ub_2MPI_UB_c \
    MPI_Type_ub_MPI_UB_c MPI_Type_ub_neg_displ_c MPI_Type_ub_pos_displ_c MPI_Type_vector_basic_c \
    MPI_Type_vector_blklen_c MPI_Type_vector_stride_c MPI_Type_vector_types_c

datatype_f: MPI_Get_elements_basic_type_f MPI_Pack_size_types_f MPI_Pack_user_type_f \
    MPI_Type_contiguous_basic_f MPI_Type_contiguous_idispls_f MPI_Type_contiguous_types_f \
    MPI_Type_extent_MPI_LB_UB_f MPI_Type_extent_types_f MPI_Type_free_pending_msg_f MPI_Type_free_types_f \
    MPI_Type_hindexed_basic_f MPI_Type_hindexed_blklen_f MPI_Type_hindexed_displs_f MPI_Type_hindexed_types_f \
    MPI_Type_hvector_basic_f MPI_Type_hvector_blklen_f MPI_Type_hvector_stride_f MPI_Type_hvector_types_f \
    MPI_Type_indexed_basic_f MPI_Type_indexed_blklen_f MPI_Type_indexed_displs_f MPI_Type_indexed_types_f \
    MPI_Type_lb_2MPI_LB_f MPI_Type_lb_MPI_LB_f MPI_Type_lb_neg_displ_f MPI_Type_lb_pos_displ_f \
    MPI_Type_size_MPI_LB_UB_f MPI_Type_size_basic_f MPI_Type_size_types_f MPI_Type_struct_basic_f \
    MPI_Type_struct_blklen_f MPI_Type_struct_displs_f MPI_Type_struct_types_f MPI_Type_ub_2MPI_UB_f \
    MPI_Type_ub_MPI_UB_f MPI_Type_ub_neg_displ_f MPI_Type_ub_pos_displ_f MPI_Type_vector_basic_f \
    MPI_Type_vector_blklen_f MPI_Type_vector_stride_f MPI_Type_vector_types_f

%_c: %_c.c libmpitest.o
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $^ $(LDLIBS)

%_f: %_f.F libmpitestf.o
	$(FC) $(FFLAGS) $(CPPFLAGS) -o $@ $^ $(LDLIBS)

# This test requires mpif90 because it uses INTEGER(MPI_ADDRESS_KIND)
MPI_Win_get_name_f: MPI_Win_get_name_f.F libmpitestf.o
	mpif90 $(FFLAGS) $(CPPFLAGS) -o $@ $^ $(LDLIBS)

