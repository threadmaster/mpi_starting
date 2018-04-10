#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

/* MPI Matrix Multiplication as a FUNCTION
 * Andrew J. Pounds, Ph.D.
 * Spring 2018
 *
 * This code copies matrix A to all of the systems and only
 * the required columns of matrix B to each system that is
 * responsible for computing that particular piece of the matrix.
 * This should save on communication overhead.
 *
 * This code also attempts to do matrix multiplication as a 
 * function of the main program with the majority of the 
 * MPI work hidden from the user in the functions.   The caveat
 * is the a few variables that define the systems have to 
 * be left in a global space.
 */


#define MAT_DIM  1000 

// Put rank and size in global space
int rank, size;
int MASTER;

// Define the main MPI function to be called
void mmm( int N, double* matA, double* matB, double* matC ){

    int tag = 0;
    extern int rank, size; 
    extern int MASTER;
    MPI_Status *status;

    // get the size of the parallel system and my rank
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    if ( rank == MASTER ) {

        /* Here we decide how to break up the matrix multiplication
         *and determine how much informatino needs to be sent to 
         *each process and over which parts of the matrix each process
         * will work.  todo contains the number of columns each process
         * will be responsible for and start and stop contain the starting
         * and stopping column B column indices for each process */

        int *todo;
        int *start;
        int *stop;

        todo = (int*) calloc(size,sizeof(int));
        start = (int*) calloc(size,sizeof(int));
        stop = (int*) calloc(size,sizeof(int));

        // initiate to 0
        for(int i=0; i < size; i++)
            *(todo+i) = 0;

        for(int i=0; i < N; i++)
            *(todo+i%size) = *(todo+i%size) + 1;

#ifdef DEBUG
        for(int i=0; i < size; i++)
            printf("Process %d will handle %d columns of B\n", i, *(todo+i));
#endif
        /* at this point we know how many columns of B each
         * process is responsible for -- now use that info to 
         * set up the starting and stopping arrays  */

        *(start) = 0;
        *(stop) = *(todo) - 1; 
        for (int i=1; i<size; i++) {
            *(start+i) = *(stop+i-1) + 1;
            *(stop+i)  = *(start+i)  + *(todo+i) - 1;
        }

#ifdef DEBUG
        for (int i=0; i<size; i++) 
            printf("Process info: %d %d %d\n", *(todo+i), *(start+i), *(stop+i));
#endif

        // Notice -- element 0 is left for the master process

        for (int i=1; i<size; i++) { 

            // send matrix A to everybody 
            int ONE = 1;
            MPI_Send(&N, ONE,  MPI_INT, i, tag, MPI_COMM_WORLD); 
            printf("just sent size %d to process %d\n", N, i);
            MPI_Send(matA, N*N, MPI_DOUBLE, i, tag, MPI_COMM_WORLD);
            printf("just matA to process %d\n",  i);

            // Send the starting and stopping arrays to every process
            MPI_Send(start, size, MPI_INT, i, tag, MPI_COMM_WORLD);
            MPI_Send(stop, size, MPI_INT, i, tag, MPI_COMM_WORLD);

            // Now start sending specific columns of matrix B to everybody

            // Step one -- build a buffer of the exact size for sending data 
            int columns = *(stop+i)-*(start+i)+1;
            double *buf; 
            buf = (double*) calloc(N*columns, sizeof(double));

            // Step two -- Pack the buffer with the appropriate matrix elements 
            int bufIndex = 0;
            for (int j=*(start+i); j<=*(stop+i); j++)  
                for (int k=0; k<N; k++) { 
                    *(buf+bufIndex) = *(matB+k*N+j);
                    bufIndex++;
                }

            // Step three -- Send the buffer to the subprocess 
            MPI_Send(buf, N*columns, MPI_DOUBLE, i, tag, MPI_COMM_WORLD);

            // Step four -- free the buffer so we can use the name again
            free(buf);  

        }

        /* At this point all of the processes have their matrix elements and should be
           computing the multiplication process for their selected elements.  The master
           process will do its part and compute the matrix product for the first set of columns. */

        for (int i=0; i<N; i++) 
            for (int j=*(start); j<=*(stop); j++)
                for (int k=0; k<N; k++)
                    *(matC+i*N+j) +=  *(matA+i*N+k) * *(matB+k*N+j); 

        /* Now start accumulating results from other processes */

        for (int i=1; i<size; i++) {  // i goes over ranks of all subprocesses //

            // Step one -- Allocate a buffer to receive elements //
            int columns = *(stop+i)-*(start+i)+1;
            double *buf; 
            buf = (double*) calloc(N*columns, sizeof(double));

            // Step two -- Get the data from the process
            MPI_Recv(buf, N*columns, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, status);

            // Step three -- Unpack the buffer to receive into the correct matrix elements
            int bufIndex = 0; 
            for (int j=*(start+i); j<=*(stop+i); j++) 
                for (int k=0; k<N; k++) { 
                    *(matC+k*N+j) = *(buf+bufIndex);
                    bufIndex++;
                } 

            // Step four -- Free the buffer so we can use it again
            free(buf);

        }

        // Free the arrays containing the information related to the scope of the subprocesses
        free(todo);
        free(start);
        free(stop);
    }

    else {  // Non-Master Processes //

        int ONE = 1;
        int matDim;

        // First receive Matrix A

        // Step one -- get the matrix dimension 
        MPI_Recv(&matDim, ONE, MPI_INT, MASTER, tag, MPI_COMM_WORLD, status);

        int NDIM = matDim;

        // Step two -- create a buffer to receive the matrix 
        double *subA;
        subA = (double*) calloc(NDIM*NDIM, sizeof(double));
        MPI_Recv(subA, NDIM*NDIM, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD, status );

        // Now start setting up to receive the components of B
        int *start;
        int *stop;

        // Note -- size is number of parallel processes, not the matrix size. This
        // is for retreiving the arrays that determine which parts of the matrix
        // we are processing.
        start = (int*) calloc(size, sizeof(int));
        stop = (int*) calloc(size, sizeof(int));

        // Step three -- get both arrays
        MPI_Recv(start, size, MPI_INT, MASTER, tag, MPI_COMM_WORLD, status);
        MPI_Recv(stop, size, MPI_INT, MASTER, tag, MPI_COMM_WORLD, status);

        // Now that this process knows which elements it is processing,
        // start receiving those elements.
        
        // Step four --- create a buffer to get the actual elements for B
        // notice the use of star, stop, and rank

        int columns = *(stop+rank)-*(start+rank)+1;
        double *subB; 
        subB = (double*) calloc(NDIM*columns, sizeof(double));

        // Step five -- get the buffer from the MASTER 
        MPI_Recv(subB, NDIM*columns, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD, status);

        // At this point the sub process should have all of the informatin it needs
        // to start working on the matrix components over which it is responsible.
        
        // Create a buffer for the computed dot products -- should be same size as buf

        double *subC; 
        subC = (double*) calloc(NDIM*columns, sizeof(double));

        // Matrix multiplication time... remember B is already ordered and is only 
        // a subset of the full matrix  

        int bufIndex = 0;
        for ( int i=0; i<NDIM; i++)
            for (int j=*(start+rank); j<=*(stop+rank); j++) { 
                for (int k=0; k<NDIM; k++) 
                    *(subC+bufIndex) += *(subA+i*NDIM+k) * *(subB+((j-*(start+rank))*NDIM)+k);
                bufIndex++;
            }

        // Now send the C buffer back to the MASTER.  The master knows how much
        // information is in the buffer so we don't have to tell it this time. 
        MPI_Send(subC, NDIM*columns, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD);

        free(subA);  
        free(subB);  
        free(subC);

    }
}


int main(int argc, char** argv){

    double *A, *B, *C;

    extern int rank, size;
    extern int MASTER; 
    int DIM = MAT_DIM;
    int MAT_SIZE = MAT_DIM*MAT_DIM;

    A = (double*) calloc(MAT_SIZE, sizeof(double));
    B = (double*) calloc(MAT_SIZE, sizeof(double));
    C = (double*) calloc(MAT_SIZE, sizeof(double));

    // Fill matrices A and B

    for (int i=0; i<DIM; i++)
        for (int j=0; j<DIM; j++) {
            *(A+i*DIM+j) = 1.0L / (double) DIM;
            *(B+i*DIM+j) = 1.0L / (double) DIM;
        }

    // Set the ID of MASTER and start MPI
    MASTER = 0;
    MPI_Init(&argc, &argv);

    mmm( DIM, A, B, C );

    // NOTE -- rank will not be defined until MPI_Comm_rank is called in mmm 
    if (rank == MASTER) {

#ifdef DEBUGPRINT
    printf("Matrix A\n", "");
    for (int i=0; i<DIM; i++) {
        for (int j=0; j<DIM; j++) 
            printf("%f ", *(A+i*DIM+j));
        printf("\n");
    }       

    printf("Matrix B\n", "");
    for (int i=0; i<DIM; i++) {
        for (int j=0; j<DIM; j++) 
            printf("%f ", *(B+i*DIM+j));
        printf("\n");
    }       

    printf("Product Matrix\n", "");
    for (int i=0; i<DIM; i++) {
        for (int j=0; j<DIM; j++) 
            printf("%f ", *(C+i*DIM+j));
        printf("\n");
    }       
#endif

    /* Compute trace of Matrix */

    double trace = 0.0;
    for (int i=0; i<DIM; i++) 
        for (int j=0; j<DIM; j++) 
            trace = trace + *(C+i*DIM+j);

    printf("Dimension of matrix is %d\n", DIM);
    printf("Sum of diagonal is %f\n", trace);
    printf("About to free A, B, and C\n");

    free(A);
    free(B);
    free(C);

    printf("Program Complete\n");

    }
    MPI_Finalize();

} 
