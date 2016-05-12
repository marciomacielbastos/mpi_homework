/*************************************************************************
 *
 * Author: Marcio Maciel Bastos
 * __________________
 *
 *  All Rights Reserved.
 *
 * NOTICE:  All information contained herein is, and remains
 * the property of Marcio Maciel Bastos.
 * The intellectual and technical concepts contained
 * herein are proprietary to Marcio Maciel Bastos
 * and its suppliers and may be covered by U.S. and Foreign Patents,
 * patents in process, and are protected by trade secret or copyright law.
 * Dissemination of this information or reproduction of this material
 * is strictly forbidden unless prior written permission is obtained
 * from Marcio Maciel Bastos.
 */


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>
#include <math.h>
#include <pthread.h>


#define MATRIX_ROWS 8
#define MIN 1
#define MAX 10
#define MESH_SIZE 2
#define NDIMENSIONS 2 

typedef struct {
   int      Size;     /* The number of processors. (Size = q_proc*q_proc)
  */
   int      p_proc;        /* The number of processors in a row (column).
*/
   int      Row;      /* The mesh row this processor occupies.        */
   int      Col;      /* The mesh column this processor occupies.     */
   int      MyRank;     /* This processors unique identifier.           */
   MPI_Comm Comm;     /* Communicator for all processors in the mesh. */
   MPI_Comm Row_comm; /* All processors in this processors row   .    */
   MPI_Comm Col_comm; /* All processors in this processors column.    */
} MESH_INFO_TYPE;

//Generate a random integer
int randint();
//Build a random entry matrix
void matrixBldr(int **matrix, int n);
//Set the Cartesian Topology
void SetUp_Mesh(MESH_INFO_TYPE *);
//Print matrix as 2d array
void printm(int** matrix, int n);
//Print matrix as 1d array
void printam(int* matrix, int n);
//Transform a 2d matrix in a 1d array
void setMatrix(int** matrix, int n, int* m, int sub_n);
//Verify if it is the correct node to work with
int verify_knot(int d, int sub_n, int row);
//Multiply each row by its diagonal element
void multiply_row(int *sub_matrix, int sub_n, int diagonal_position, int grid_row, int diagonal_value);
//Set chunks of matrix in a whole 1d matrix
void setMultipliedMatrix(int *multiplied_matrix, int n, int *sub_matrix, int n_sub, int rank);
//Set the row of summed elements of each column
void setSummedMatrix(int *summed_matrix, int *column_sum, int sub_n, int rank, int p_proc);
void summation(int *summed_matrix, int *column_sum, int *msg_sum, int* sub_matrix, int n, int sub_n, MESH_INFO_TYPE *grid);


int main(int argc, char* argv[]){
	int sub_n, n = MATRIX_ROWS, Root =0, val, processor_rank;
	int msize = n*n, sub_msize, diag, r, subrow, subcol;
	int **matrix = (int **)malloc(n*sizeof(int*));
	int msg_diag;
	int array_like_matrix[msize];
	int multiplied_matrix[msize];
	int summed_matrix[n];
	int both_matrix[msize];
	int *sub_matrix, *msg_matrix, *column_sum, *msg_sum;
	int Matrix_Size = n*n;
	//Grid of Cartesian Toplogy
	MESH_INFO_TYPE grid;
	MPI_Status status;
  //If it be passed the number of rows
  if(argc>1){ 
    n = atoi(argv[1]);
    matrix = (int **)malloc(n*sizeof(int*));
  }
 	/* Initialising */
  	MPI_Init (&argc, &argv);
  	/* Set up the MPI_COMM_WORLD and CARTESIAN TOPOLOGY */
  	SetUp_Mesh(&grid);

  	/*build matrix only in root*/
  	if(grid.MyRank == Root){
  		matrixBldr(matrix, n);
  		printf("Original matrix\n");
  		printm(matrix,n);
  	}
  	/* Blocks until all processes in the communicator have reached this routine.  */
  	MPI_Barrier(grid.Comm);
  	/*The Matrix has to be perfectly divisible by the number of processes*/
  	if( n % grid.p_proc != 0 ){   
  		MPI_Finalize();
  		if(grid.MyRank == Root){
  			printf("Matrices can't be divided among processors equally\n");
  		}
  		exit(-1);
 	}
  /*The dimension of submatrices is the size of the whole matrix divided by the
  number of processes (MUST BE DIVISIBLE!)*/
 	sub_n = n / grid.p_proc;
 	sub_msize = sub_n*sub_n;
 	sub_matrix = (int *) malloc (sub_msize * sizeof(int));
 	msg_matrix = (int *) malloc (sub_msize * sizeof(int));
 	if(grid.MyRank == Root){
 		setMatrix(matrix, n, array_like_matrix, sub_n);
 	}
 	/* Scatter the Data  to all processes by MPI_SCATTER */
 	MPI_Scatter (array_like_matrix, sub_msize, MPI_INT,sub_matrix,sub_msize , MPI_INT, 0, grid.Comm);
 	MPI_Barrier(grid.Comm);  
  if(grid.MyRank != 0){
    MPI_Send(sub_matrix, sub_msize, MPI_INT, 0, 0, grid.Comm);
  }

  if(grid.MyRank == 0){
    printf("rank = %d, sub_matrix:\n", grid.MyRank);
    printam(sub_matrix, sub_n);
    for(processor_rank = 1; processor_rank < grid.Size; processor_rank++){
      MPI_Recv(msg_matrix, sub_msize, MPI_INT, processor_rank, 0, grid.Comm, MPI_STATUS_IGNORE);
      printf("rank = %d, sub_matrix:\n", processor_rank);
      printam(msg_matrix, sub_n);
    }
  }

  MPI_Barrier(grid.Comm);
 	MPI_Bcast(&val, 1, MPI_INT, 0, grid.Comm);
 	MPI_Barrier(grid.Comm);
  for(diag = 0; diag < n; diag++){
 		if(grid.MyRank == Root){
 			msg_diag = matrix[diag][diag];
 		}
 		/*Root send diag value to others*/
    MPI_Barrier(grid.Comm);
 		MPI_Bcast (&msg_diag, 1, MPI_INT, 0, grid.Comm);
    // printf("%d, rank %d\n", msg_diag, grid.);
    MPI_Barrier(grid.Comm);
 		multiply_row(sub_matrix, sub_n, diag, grid.Row, msg_diag);
 	}
 	
 	MPI_Barrier(grid.Comm);
 	if(grid.MyRank != 0){
 		MPI_Send(sub_matrix, sub_msize, MPI_INT, 0, 0, grid.Comm);
 	}
 	
 	if(grid.MyRank == 0){
 		setMultipliedMatrix(multiplied_matrix, n, sub_matrix, sub_n, 0);
 		for(processor_rank = 1; processor_rank < grid.Size; processor_rank++){
 			MPI_Recv(msg_matrix, sub_msize, MPI_INT, processor_rank, 0, grid.Comm, MPI_STATUS_IGNORE);
 			setMultipliedMatrix(multiplied_matrix, n, msg_matrix, sub_n, processor_rank);
 		}
 	}

 	if(grid.MyRank == Root){
    free(matrix);
 		printf("Matrix of diagonal element times row:\n");
 		printam(multiplied_matrix, n);
 	}

  MPI_Scatter (array_like_matrix, sub_msize, MPI_INT,sub_matrix,sub_msize , MPI_INT, 0, grid.Comm);
  MPI_Barrier(grid.Comm);

  /*The sum of each column of each sub_matrix in each node*/
  summation(summed_matrix, column_sum, msg_sum, sub_matrix, n, sub_n, &grid);
 
  if(grid.MyRank == Root){
    printf("The summ of the elements of the original matrix is %d\n", summed_matrix[0]);
 	}


  for(subcol = 0; subcol < n; subcol++){
    summed_matrix[subcol] = 0;
  }
  MPI_Scatter (multiplied_matrix, sub_msize, MPI_INT,sub_matrix,sub_msize , MPI_INT, 0, grid.Comm);
  MPI_Barrier(grid.Comm);

  /*The sum of each column of each sub_matrix in each node*/
  summation(summed_matrix, column_sum, msg_sum, sub_matrix, n, sub_n, &grid);
  
  if(grid.MyRank == Root){
    printf("The summ of the elements of the multiplied matrix matrix is %d\n", summed_matrix[0]);
  }
  MPI_Finalize(); 	
	return 0;
}

void SetUp_Mesh(MESH_INFO_TYPE *grid) {

   int Periods[2];          /* For Wraparound in each dimension.           */
   int Dimensions[2];       /* Number of processors in each dimension.     */
   int Coordinates[2];      /* processor Row and Column identification     */
   int Remain_dims[2];      /* For row and column communicators.           */


   /* MPI rank and MPI size */
   MPI_Comm_size(MPI_COMM_WORLD, &(grid->Size));
   MPI_Comm_rank(MPI_COMM_WORLD, &(grid->MyRank));

   /* For square mesh */
   grid->p_proc = (int)sqrt((double) grid->Size);             
	if(grid->p_proc * grid->p_proc != grid->Size){
		 MPI_Finalize();
		 if(grid->MyRank == 0){
			 printf("Number of Processors should be perfect square\n");
		 }
		 exit(-1);
	}

   Dimensions[0] = Dimensions[1] = grid->p_proc;

   /* Wraparound mesh in both dimensions. */
   Periods[0] = Periods[1] = 1;    

   /*  Create Cartesian topology  in two dimnesions and  Cartesian 
       decomposition of the processes   */
   MPI_Cart_create(MPI_COMM_WORLD, NDIMENSIONS, Dimensions, Periods, 0, &(grid->Comm));
   MPI_Cart_coords(grid->Comm, grid->MyRank, NDIMENSIONS, Coordinates);

   grid->Row = Coordinates[0];
   grid->Col = Coordinates[1];

   /* Construction of row communicator and column communicators (use cartesian 
      row and columne machanism to get Row/Col Communicators)  */

   Remain_dims[0] = 0;            
   Remain_dims[1] = 1; 

   /* The output communicator represents the column containing the process */
   MPI_Cart_sub(grid->Comm, Remain_dims, &(grid->Row_comm));
   
   Remain_dims[0] = 1;
   Remain_dims[1] = 0;

   /* The output communicator represents the row containing the process */
   MPI_Cart_sub(grid->Comm, Remain_dims, &(grid->Col_comm));
 }

void matrixBldr(int **matrix,int n){
  int j, i;
  srand(time(0));
  for(i = 0; i < n; i++)
  	matrix[i] = (int *)malloc(n * sizeof(int));
  for(i = 0 ; i < n; i++){
    for(j = 0; j < n; j++){
        matrix[i][j]=randint();
    }
  }
}

void printm(int** matrix, int n){
  int i,j;
  for(i=0; i < n; i++){
    for(j=0; j<n; j++){
      printf("%d\t", matrix[i][j]);
    }
    printf("\n");
  }
}

void printam(int* matrix, int n){
  int i,j;
  for(i=0; i < n; i++){
    for(j=0; j<n; j++){
      printf("%d\t", matrix[i*n+j]);
    }
    printf("\n");
  }
}

int randint(){
  return rand()%(MAX-MIN)+MIN;
}

void setMatrix(int **matrix, int n, int* m, int sub_n){
	int mrow, mcol, subrow, subcol,i=0;
	int qnt = n / sub_n;
	for(mrow = 0; mrow < qnt; mrow++){
		for (mcol= 0; mcol < qnt; mcol++){
			for(subrow=0;subrow<sub_n;subrow++){
				for(subcol=0;subcol < sub_n; subcol++){
					m[i] = matrix[mrow*sub_n+subrow][mcol*sub_n+subcol];
					i++;
				}
			}
		}
	}	
}

int verify_knot(int d, int n_sub, int row){
	if(d/n_sub == row){
		return 1;
	}
	else{
		return 0;
	}
}

void multiply_row(int *sub_matrix, int sub_n, int diagonal_position, int grid_row, int diagonal_value){
	int subrow, subcol;
	if(verify_knot(diagonal_position, sub_n, grid_row)){
 		subrow = (diagonal_position)%(sub_n);
 		for(subcol = 0; subcol < sub_n; subcol++){
 			sub_matrix[subrow*sub_n+subcol] = diagonal_value*sub_matrix[subrow*sub_n+subcol];	
 		}
 	}
}

void setMultipliedMatrix(int *multiplied_matrix, int n, int *sub_matrix, int n_sub, int rank){
	int grid_row = (rank)/(n/n_sub);
	int grid_col = (rank)%(n/n_sub);
	int mrow = n/n_sub, mcol = n/n_sub;
	int subrow, subcol;
	for(subrow = 0; subrow < n_sub; subrow++){
		for (subcol = 0; subcol < n_sub; subcol++){
			multiplied_matrix[(n_sub*n_sub*mrow)*grid_row+grid_col*n_sub+subrow*n+subcol] = sub_matrix[subrow*n_sub+subcol];
		}
	}	
}

void setSummedMatrix(int *summed_matrix, int *column_sum, int sub_n, int rank, int p_proc){
  int col = rank%p_proc;
  int offset;
  for(offset = 0; offset < sub_n; offset++){
    summed_matrix[col*sub_n+offset] += column_sum[offset];
  }
}

void summation(int *summed_matrix, int *column_sum, int *msg_sum, int* sub_matrix, int n, int sub_n, MESH_INFO_TYPE *grid){
  int subrow, subcol, processor_rank;
  column_sum = (int *) malloc (sub_n * sizeof(int));
  msg_sum = (int *) malloc (sub_n * sizeof(int));
  for(subcol = 0; subcol < sub_n; subcol++){
    column_sum[subcol] = 0;
  }
  for(subcol = 0; subcol < n; subcol++){
    summed_matrix[subcol] = 0;
  }
  for(subcol = 0; subcol < sub_n ; subcol++){
    for(subrow = 0; subrow < sub_n; subrow++){
      column_sum[subcol] += sub_matrix[subrow*sub_n+subcol]; 
    }
  }
  MPI_Barrier(grid->Comm);
  if(grid->MyRank != 0){
    MPI_Send(column_sum, sub_n, MPI_INT, 0, 0, grid->Comm);
  }

  if(grid->MyRank == 0){
    setSummedMatrix(summed_matrix, column_sum, sub_n, 0, grid->p_proc);
    for(processor_rank = 1; processor_rank < grid->Size; processor_rank++){
      MPI_Recv(msg_sum, sub_n, MPI_INT, processor_rank, 0, grid->Comm, MPI_STATUS_IGNORE);
      setSummedMatrix(summed_matrix, msg_sum, sub_n, processor_rank, grid->p_proc);
    }
  }
  if(grid->MyRank == 0){
    for(subcol = 1; subcol < n; subcol++){
        summed_matrix[0] += summed_matrix[subcol];
    }
  }
}