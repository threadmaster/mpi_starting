program mpitest

USE MPI

integer :: rank, size, ierror

call MPI_Init(ierror)
call MPI_COMM_size(MPI_COMM_WORLD, size, ierror)
call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierror)

write(*,*) 'Hello World, I am ', rank, ' of ', size

call MPI_Finalize(ierror)

end

