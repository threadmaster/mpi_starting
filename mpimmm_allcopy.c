#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#define MASTER 0

/* Simplified MPI Matrix Multiplication
 * Andrew J. Pounds, Ph.D.
 * Spring 2018
 *
/* Note -- in this version we are simpliying things by sending every system
 * complete copies of matrices A and B and asking each system to work over
 * only certain columns of B.  
 */

typedef struct limits {
    int start;
    int stop;
} LIMITS;

int main(int argc, char** argv)
{
    int i, j, N, procs;
    int start, finish;
    N = 2000;

    double* A = (double*) malloc(N*N*sizeof(double));
    double* B = (double*) malloc(N*N*sizeof(double));
    double* C = (double*) malloc(N*N*sizeof(double));

    int rank;
    MPI_Init(&argc, &argv);
    int num_tasks, num_workers;
    MPI_Status stat;

    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int threads = num_tasks;
    int* numToDo = (int*) malloc(threads*sizeof(int));
    int len = N;
    int dim = N;

    // Divide the code up equally amongst all of the threads
    if (  len % threads == 0 ) {
        for (i=0;i<threads;i++) *(numToDo+i) = len / threads;
    }
    else {
        for (i=0;i<threads;i++) *(numToDo+i) = len / threads;
        for (i=0;i<len % threads;i++) *(numToDo+i) = *(numToDo+i) + 1;
    }

    LIMITS* variableIndex =  (LIMITS*) malloc(threads*sizeof(LIMITS));
    start = 0;
    for ( i=0; i<threads; i++) {
        (variableIndex+i) -> start = start;
        (variableIndex+i) -> stop = start + *(numToDo+i) ;
        start = (variableIndex+i)-> stop;
    }


    // Master Function ---------------------------------------------------

    if (rank == MASTER)
    {
        int i, j, k;

        // Fill the matrices

        for(i=0; i<N; i++)
        {
            for(j=0; j<N; j++)
            {
                A[i*N+j] = 1.0L/(double)sqrt((double)N);
                B[i*N+j] = 1.0L/(double)sqrt((double)N);
                C[i*N+j] = 0;
            }
        }

        // SEND DATA TO EVERY MACHINE
        for(i = 1; i < threads; i++)
        {
            printf("Sending A and B to process %d....", i);
            MPI_Send(A, dim*dim, MPI_DOUBLE, i, MASTER, MPI_COMM_WORLD); // send mat A
            MPI_Send(B, dim*dim, MPI_DOUBLE, i, MASTER, MPI_COMM_WORLD); // send mat B
            printf("  Complete!\n");
        }

        // DO SECTION FOR WHICH MASTER IS RESPONSIBLE
        start = (variableIndex+rank)->start;
        finish = (variableIndex+rank)->stop; 
        int numRows = (finish - start);

        double *tmp = (double *)malloc( dim*(numRows)*sizeof(double));
        for(i = 0; i < dim*numRows; i++)
            tmp[i] = 0.0;

        for(i = start; i < finish; i++)
        {
            for(j = 0; j < dim; j++)
            {
                for(k = 0; k < dim; k++)
                {
                    tmp[(i-start)*dim+j] += A[i*dim+k] * B[k*dim+j];
                }
            }
        }

        for(i = start; i < finish; i++)
        {
            for(j = 0; j < dim; j++)
            {
                C[i*dim+j] =	tmp[(i-start)*dim+j];
            }
        }

        free(tmp); 


        // RECEIVE DATA FROM EVERY MACHINE
        for(k = 1; k < threads; k++) 
        {
            start = (variableIndex+k)->start;
            finish = (variableIndex+k)->stop; 
            int numRows = (finish - start);

            double *tmp = (double *)malloc( dim*(numRows)*sizeof(double));

            MPI_Recv(tmp, dim*numToDo[k], MPI_DOUBLE, k, 0, MPI_COMM_WORLD, &stat);
            for(i = start; i < finish; i++)
            {
                for(j = 0; j < dim; j++)
                {
                    C[i*dim+j] =	tmp[(i-start)*dim+j];
                }
            }
            free(tmp);
        }


        double trace=0.0;
        for (i=0;i<dim;i++)	trace += C[i*dim+i];

        printf("DIMENSION : %d   TRACE %15.13f\n", dim, trace);

    }

    // Worker Function ---------------------------------------------------

    else	
    {

        int i, j, k;	
        int start, finish, numRows;
        int dim = N;

        MPI_Recv( A,   dim*dim,  MPI_DOUBLE,   0, 0, MPI_COMM_WORLD, &stat );
        MPI_Recv( B,   dim*dim,  MPI_DOUBLE,   0, 0, MPI_COMM_WORLD, &stat );


        start = (variableIndex+rank)->start;
        finish = (variableIndex+rank)->stop; 
        numRows = (finish - start);

        double *tmp = (double *)malloc( dim*(numRows)*sizeof(double));
        for(i = 0; i < dim*numRows; i++)
            tmp[i] = 0.0;

        for(i = start; i < finish; i++)
        {
            for(j = 0; j < dim; j++)
            {
                for(k = 0; k < dim; k++)
                {
                    tmp[(i-start)*dim+j] += A[i*dim+k] * B[j*dim+k];
                }
            }
        }



        printf("Worker %d sending to Master\n", rank);
        MPI_Send(tmp, dim*numRows, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        printf("Worker %d finished sending\n", rank);


        //////////////////////////
        free(tmp);
        //////////////////////////
    }

    MPI_Finalize();
    //puts("MPI finalized");

    return 0;
}
