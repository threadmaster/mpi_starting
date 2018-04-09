#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>


#define MAT_DIM  100 

//void mmm ( int N, double *A, double *B, double *C );

void mmm( int N, double* matA, double* matB, double* matC ){

    int rank, size;
    int tag = 0;;
    int MASTER = 0;

    MPI_Status *status;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    if ( rank == MASTER ) {

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

        for(int i=0; i < size; i++)
            printf(" %d\n", *(todo+i));

        *(start) = 0;
        *(stop) = *(todo) - 1; 
        for (int i=1; i<size; i++) {
            *(start+i) = *(stop+i-1) + 1;
            *(stop+i)  = *(start+i)  + *(todo+i) - 1;
        }

        for (int i=0; i<size; i++) 
            printf("%d %d %d\n", *(todo+i), *(start+i), *(stop+i));

        // Notice -- element 0 is left for the master process

        for (int i=1; i<size; i++) { 

            // send matrix A to everybody 
            int ONE = 1;
            MPI_Send(&N, ONE,  MPI_INT, i, tag, MPI_COMM_WORLD); 
            MPI_Send(matA, N*N, MPI_DOUBLE, i, tag, MPI_COMM_WORLD);

            // Send the starting and stopping arrays to every process
            MPI_Send(start, size, MPI_INT, i, tag, MPI_COMM_WORLD);
            MPI_Send(stop, size, MPI_INT, i, tag, MPI_COMM_WORLD);

            // Now start sending columns of matrix B to everybody

            int columns = *(stop+i)-*(start+i)+1;
            double *buf; 
            buf = (double*) calloc(N*columns, sizeof(double));

            // Pack the buffer with the appropriate matrix elements 
            int bufIndex = 0;
            for (int j=*(start+i); j<=*(stop+i); j++)  
                for (int k=0; k<N; k++) { 
                    *(buf+bufIndex) = *(matB+k*N+j);
                    bufIndex++;
                }

            // Now send the buffer to the subprocess 
            MPI_Send(buf, N*columns, MPI_DOUBLE, i, tag, MPI_COMM_WORLD);

            free(buf);  

        }

        // puts("About to start matrix multiplication in master process");

        /* At this point all of the processes have their matrix elements and should be
           computing the multiplication process for their selected elements.  The master
           process will do its part and compute the matrix product for the first set of rows. */

        for (int i=0; i<N; i++) 
            for (int j=*(start); j<=*(stop); j++)
                for (int k=0; k<N; k++)
                    *(matC+i*N+j) +=  *(matA+i*N+k) * *(matB+k*N+j); 

        puts("Finished matrix multiplication in master process");

        /* Now start accumulating results from other processes */

        for (int i=1; i<size; i++) {  // i goes over ranks of all subprocesses //
            // Allocate buffer to receive elements //
            int columns = *(stop+i)-*(start+i)+1;
            double *buf; 
            buf = (double*) calloc(N*columns, sizeof(double));
            MPI_Recv(buf, N*columns, MPI_DOUBLE, i, tag, MPI_COMM_WORLD, status);

            puts("Just tried to receive buffer from process");

            int bufIndex = 0; 
            for (int j=*(start+i); j<=*(stop+i); j++) 
                for (int k=0; k<N; k++) { 
                         printf ("The value of buf %d is %f\n", bufIndex, *(buf+bufIndex));
                    *(matC+k*N+j) = *(buf+bufIndex);
                    bufIndex++;
                } 

            free(buf);

        }

        //puts("Finished Matrix Multiplication ");

        free(todo);
        free(start);
        free(stop);
    }

    else {  // Non-Master Processes //

        int ONE = 1;
        int *matDim = malloc(sizeof(int));

        puts("In subprocess.");
        // Now receive matrix A by getting its size and allocating a buffer
        MPI_Recv(matDim, ONE, MPI_INT, MASTER, tag, MPI_COMM_WORLD, status);

        puts ("just finished call to MPI_Recv");

        int NDIM = *matDim;

        printf("The value of matDim is %d\n", NDIM);

        double *subA;
        subA = (double*) calloc(NDIM*NDIM, sizeof(double));
        MPI_Recv(subA, NDIM*NDIM, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD, status );

        puts ("Got A");

        // Now start setting up to receive the components of B
        int *start;
        int *stop;

        start = (int*) calloc(size, sizeof(int));
        stop = (int*) calloc(size, sizeof(int));

        MPI_Recv(start, size, MPI_INT, MASTER, tag, MPI_COMM_WORLD, status);
        MPI_Recv(stop, size, MPI_INT, MASTER, tag, MPI_COMM_WORLD, status);

        puts ("Got start and stop");

        // Create a buffer to get the actual elements for B

        int columns = *(stop+rank)-*(start+rank)+1;
        double *subB; 
        subB = (double*) calloc(NDIM*columns, sizeof(double));

        printf ("B should have %d columns\n", columns);

        // Now get the buffer from the MASTER 
        MPI_Recv(subB, NDIM*columns, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD, status);

        puts ("Got B");

        // Create a buffer for the computed dot products -- should be same size as buf

        double *subC; 
        subC = (double*) calloc(NDIM*columns, sizeof(double));

        puts ("Done callocing A, B, and C in subprocess");

        // Matrix multiplication time... remember B is already ordered and is only 
        // a subset of the full matrix  

        int bufIndex = 0;
        for ( int i=0; i<NDIM; i++)
            for (int j=*(start+rank); j<=*(stop+rank); j++) { 
                for (int k=0; k<NDIM; k++) 
                    //*(subC+i*NDIM+j-*(start+rank)) += *(subA+i*NDIM+k) * *(subB+((j-*(start+rank))*NDIM)+k);
                    *(subC+bufIndex) += *(subA+i*NDIM+k) * *(subB+((j-*(start+rank))*NDIM)+k);
                bufIndex++;
            }

        puts ("Preparing to send C back to master");

        // Now send the C buffer back to the MASTER 
        MPI_Send(subC, NDIM*columns, MPI_DOUBLE, MASTER, tag, MPI_COMM_WORLD);

        puts ("Just sent C back to master");

        free(subA);  
        free(subB);  
        free(subC);

        puts ("Freed A, B, and C in subprocess");


        /*
           printf("Matrix A in subprocess \n");


           for (int i=0; i<NDIM; i++) {
           for (int j=0; j<NDIM; j++) 
           printf("%f ", *(A+i*NDIM+j));
           printf("\n"); 
           }       
           */

    }
}


int main(int argc, char** argv){

    double *A, *B, *C;

    int DIM = MAT_DIM;
    int MAT_SIZE = MAT_DIM*MAT_DIM;
    int rank;

    A = (double*) calloc(MAT_SIZE, sizeof(double));
    B = (double*) calloc(MAT_SIZE, sizeof(double));
    C = (double*) calloc(MAT_SIZE, sizeof(double));


    //puts("Malloc worked");

    // Fill matrices A and B

/*
    for (int i=0; i<DIM; i++)
        for (int j=0; j<DIM; j++) {
            *(A+i*DIM+j) = fmin((double) (i+1) / (double) (j+1), (double) (j+1) / (double) (i+1));
            if ( i == j ) 
                *(B+i*DIM+j) = 1.0;
            else
                *(B+i*DIM+j) = 0.0;
        }
*/
    for (int i=0; i<DIM; i++)
        for (int j=0; j<DIM; j++) {
            *(A+i*DIM+j) = 1.0 / sqrt( (double) DIM ); 
            *(B+i*DIM+j) = 1.0 / sqrt( (double) DIM ); 
        }
    /* 
       for (int i=0; i<DIM; i++) {
       for (int j=0; j<DIM; j++) 
       printf("%f ", *(A+i*DIM+j));
       printf("\n");
       }       
       */

    //puts("Driver function initiating MPI");
    MPI_Init(&argc, &argv);

    mmm( DIM, A, B, C );

    if (rank == 0) {
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

        /* Compute trace of Matrix */

        double trace = 1.0;
        for (int i=0; i<DIM; i++) 
            for (int j=0; j<DIM; j++) 
                trace = trace * *(C+i*DIM+j);

        printf("Trace of matrix is %f\n", trace);
        printf("About to free A, B, and C\n");
        printf("Program Complete\n");

    }

    MPI_Finalize();
    free(A);
    free(B);
    free(C);

} 
