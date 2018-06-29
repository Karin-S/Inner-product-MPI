#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

void create_matrix(int n, int m, float **init_matrix_1D, float ***init_matrix);
void create_result_matrix(int columns, int rows, float **results_1D, float ***results);
void initialize_matrix(int n, int m, float **mx);
float inner_product(float* matrix_a, float* matrix_b, int m);
void self_product(float** matrix, int rows, int m, float** results);
void print_initial_matrix(int n, int m, float** matrix);
void print_results(int columns, int rows, float** results);
void self_check(float **final_result, float **final_result_copy, int n);

int main(int argc, char** argv)
{
  int myid, numprocs, prev_id, next_id;
  MPI_Request *request = NULL;
  MPI_Status *status = NULL;

  int count = 0;
  int n, m;
  float** init_matrix = NULL;
  float* init_matrix_1D = NULL;
  int rows, max_columns, max_iter;
  int i, j;
  float** matrix_a = NULL;
  float* matrix_a_1D = NULL;
  float** matrix_b = NULL;
  float* matrix_b_1D = NULL;
  int final_result_count = 0;
  float** final_result = NULL;
  float* final_result_1D = NULL;
  float** final_result_copy = NULL;
  float* final_result_copy_1D = NULL;
  int processor_results_count = 0;
  float** processor_results = NULL;
  float* processor_results_1D = NULL;
  int current_iter = 1;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  if (argc != 3)
    {
      if (myid == 0)
        {
          printf("Illegal arguments.\n");
          printf("Please enter the command in the following format:\n");
          printf("mpirun -np [proc num] inner_products [the number of row] [the number of column]\n");
          printf("Note: [proc num] should be odd number; [the number of row] / [proc num] = 0");
          printf("\n");
        }
      goto EXIT;
    }

  n = atoi(argv[1]);
  m = atoi(argv[2]);

  if ((numprocs % 2 != 1) || (n % numprocs != 0))
    {
      if (myid == 0)
        {
          printf("Illegal arguments.\n");
          printf("Please enter the command in the following format:\n");
          printf("mpirun -np [proc num] inner_products [the number of row] [the number of column]\n");
          printf("Note: [proc num] should be odd number; [the number of row] / [proc num] = 0");
          printf("\n");
        }
      goto EXIT;
    }

  rows = n / numprocs;
  max_iter = (numprocs + 1) / 2;
  max_columns = max_iter * rows - 1;
  create_matrix(rows, m, &matrix_a_1D, &matrix_a);
  create_matrix(rows, m, &matrix_b_1D, &matrix_b);
  request = (MPI_Request*)malloc((numprocs) * sizeof(MPI_Request));
  status = (MPI_Status*)malloc((numprocs) * sizeof(MPI_Status));
  for (i = 0; i < numprocs; i++)
    {
      request[i] = MPI_REQUEST_NULL;
    }

  // Create matrix and send task matrix_a and matrix_b to other processors
  if (myid == 0)
    {
      create_matrix(n, m, &init_matrix_1D, &init_matrix);
      initialize_matrix(n, m, init_matrix);
      printf("The initial grid: \n");
      print_initial_matrix(n, m, init_matrix);

      final_result_count = n * (n - 1) / 2;
      create_result_matrix(n - 1, n - 1, &final_result_1D, &final_result);

      memcpy(matrix_a[0], init_matrix[0], rows * m * sizeof(float));
      for (i = 1; i < numprocs; i++)
        {
          MPI_Isend(init_matrix[i * rows], rows * m, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &request[0]);
        }

      memcpy(matrix_b[0], init_matrix[rows], rows * m * sizeof(float));
      for (i = 1; i < numprocs; i++)
        {
          j = (i + 1) % numprocs;
          MPI_Isend(init_matrix[j * rows], rows * m, MPI_FLOAT, i, 0, MPI_COMM_WORLD, &request[0]);
        }
    }
  else
    {
      // Receive initial left and right block
      MPI_Recv(matrix_a[0], rows * m, MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status[myid]);
      MPI_Recv(matrix_b[0], rows * m, MPI_FLOAT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status[myid]);
    }

  processor_results_count = n * (n - 1) / 2 / numprocs;
  create_result_matrix(max_columns, rows, &processor_results_1D, &processor_results);

  // First calculate the inner products in task matrix_a
  self_product(matrix_a, rows, m, processor_results);

  for (current_iter = 1; current_iter < max_iter; current_iter++)
    {
      // Calculate among left and right
      for (i = 0; i < rows; i++)
        {
          for (j = 0; j < rows; j++)
            {
              processor_results[i][j - i + rows * current_iter - 1] = inner_product(matrix_a[i], matrix_b[j], m);
            }
        }
      if (myid == 0)
        {
          //MPI_Wait(&request[numprocs - 2], &status[numprocs - 2]);
          MPI_Isend(matrix_b[0], rows * m, MPI_FLOAT, numprocs - 1, 0, MPI_COMM_WORLD, &request[0]);
          MPI_Recv(matrix_b[0], rows * m, MPI_FLOAT, (myid + 1) % numprocs, MPI_ANY_TAG, MPI_COMM_WORLD, &status[0]);
        }
      else
        {
          //MPI_Wait(&request[myid - 1], &status[myid - 1]);
          MPI_Isend(matrix_b[0], rows * m, MPI_FLOAT, myid - 1, 0, MPI_COMM_WORLD, &request[myid]);
          MPI_Recv(matrix_b[0], rows * m, MPI_FLOAT, (myid + 1) % numprocs, MPI_ANY_TAG, MPI_COMM_WORLD, &status[myid]);
        }
    }

  // Send results to master
  if (myid == 0)
    {
      // Receive results and sort

      int normal_count = n - 1;

      for (current_iter = 0; current_iter < numprocs;)
        {
          int current_max_columns = max_columns;
          for (i = 0; i < rows; i++)
            {
              // Ordered Cell
              for (j = 0; j < normal_count && j < current_max_columns; j++)
                {
                  final_result[i + current_iter * rows][j] = processor_results[i][j];
                }

              // Malposed cells
              for (j = normal_count; j < current_max_columns; j++)
                {
                  final_result[j - normal_count][n - 2 - j] = processor_results[i][j];
                }

              current_max_columns--;
              normal_count--;
            }
          current_iter++;
          if (current_iter < numprocs)
            MPI_Recv(processor_results[0], processor_results_count, MPI_FLOAT, current_iter, MPI_ANY_TAG, MPI_COMM_WORLD, &status[0]);
        }
    }
  else
    {
      // Send result to master processor
      MPI_Wait(&request[0], &status[0]);
      MPI_Send(processor_results[0], processor_results_count, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }

  if (myid == 0)
    {
      printf("\n");
      printf("The parallel computation result:\n");
      print_results(n - 1, n - 1, final_result);
      printf("\n");

      // Sequential computation
      create_result_matrix(n - 1, n - 1, &final_result_copy_1D, &final_result_copy);
      self_product(init_matrix, n, m, final_result_copy);

      printf("The sequential computation result: \n");
      print_results(n - 1, n - 1, final_result_copy);
      self_check(final_result, final_result_copy, n);
    }

  EXIT:
    MPI_Finalize();
    return 0;
}

void create_matrix(int n, int m, float **init_matrix_1D, float ***init_matrix)
{
  int count = m * n * 2;

  *init_matrix_1D = (float*)malloc(sizeof(float) * count);
  *init_matrix = (float**)malloc(sizeof(float *) * n);
  int i;

  for (i = 0; i < n; i++)
    {
      (*init_matrix)[i] = &((*init_matrix_1D)[i * m]);
    }
}

void create_result_matrix(int columns, int rows, float **results_1D, float ***results)
{
  int total = (columns * 2 - rows + 1) * rows;

  *results_1D = malloc(total * sizeof(float));
  *results = malloc(rows * sizeof(float*));
  int i, j = 0;

  for (i = 0; i < rows; i++)
    {
      (*results)[i] = &((*results_1D)[j]);
      j = j + columns;
      columns--;
    }
}

// initialize the matrix with the random float number between 0 and 1
void initialize_matrix(int n, int m, float **mx)
{
  time_t s;

  srand((unsigned)time(&s));
  int i, j;

  for (i = 0; i < n; i++)
    {
      for (j = 0; j < m; j++)
        {
          mx[i][j] = ((float)rand() / (float)(RAND_MAX)) * 1.0;
        }
    }
}

float inner_product(float* matrix_a, float* matrix_b, int m)
{
  float inner_product = 0;
  int i;

  for (i = 0; i < m; i++)
    {
      inner_product += matrix_a[i] * matrix_b[i];
    }

  return inner_product;
}

void self_product(float** matrix, int rows, int m, float** results)
{
  int i, j;
  int col = rows - 1;

  for (i = 0; i < rows - 1; i++)
    {
      for (j = 0; j < col; j++)
        {
          results[i][j] = inner_product(matrix[i], matrix[j + i + 1], m);
        }
      col--;
    }
}

void print_initial_matrix(int n, int m, float** matrix)
{
  int i, j;

  for (i = 0; i < n; i++)
    {
      for (j = 0; j < m; j++)
        {
          printf("%.2f ", matrix[i][j]);
        }
      printf("\n");
    }
}

void print_results(int columns, int rows, float** results)
{
  int i, j;

  for (i = 0; i < rows; i++)
    {
      for (j = 0; j < columns; j++)
        {
          printf("%.2f ", results[i][j]);
        }
      columns--;
      printf("\n");
    }
}

void self_check(float **final_result, float **final_result_copy, int n)
{
  int flag = 0;
  int i, j;

  for (i = 0; i < (n - 1); i++)
    {
      for (j = 0; j < (n - i - 1); j++)
        {
          if (final_result[i][j] != final_result_copy[i][j])
            {
              flag = 1;
            }
        }
    }
  printf("\n");
  if (flag == 0)
    {
      printf("Self-chech: The result of parallel program is the same as sequential program.\n");
      printf("            Means the result of the parallel movement is correct.\n");
    }
  else
    {
      printf("Self-chech: The result of parallel program is not the same as sequential program.\n");
      printf("            Means the result of the parallel movement is wrong.\n");
    }
  printf("\n");
}