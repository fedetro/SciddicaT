#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "util.hpp"

// ----------------------------------------------------------------------------
// I/O parameters used to index argv[]
// ----------------------------------------------------------------------------
#define HEADER_PATH_ID 1
#define DEM_PATH_ID 2
#define SOURCE_PATH_ID 3
#define OUTPUT_PATH_ID 4
#define STEPS_ID 5
// ----------------------------------------------------------------------------
// Simulation parameters
// ----------------------------------------------------------------------------
#define SIZE_OF_X 5
#define P_R 0.5
#define P_EPSILON 0.001
// ----------------------------------------------------------------------------
// Read/Write access macros linearizing single/multy layer buffer 2D indices
// ----------------------------------------------------------------------------
#define SET(M, columns, i, j, value) ((M)[(((i) * (columns)) + (j))] = (value))
#define GET(M, columns, i, j) (M[(((i) * (columns)) + (j))])
#define BUF_SET(M, rows, columns, n, i, j, value) ( (M)[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] = (value) )
#define BUF_GET(M, rows, columns, n, i, j) ( M[( ((n)*(rows)*(columns)) + ((i)*(columns)) + (j) )] )

// ----------------------------------------------------------------------------
// I/O functions
// ----------------------------------------------------------------------------
#define STRLEN 256
void readHeaderInfo(char* path, int &nrows, int &ncols, /*double &xllcorner, double &yllcorner, double &cellsize,*/ double &nodata)
{
  FILE* f;
  
  if ( (f = fopen(path,"r") ) == 0){
    printf("%s configuration header file not found\n", path);
    exit(0);
  }

  //Reading the header
  char str[STRLEN];
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); ncols = atoi(str);
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); nrows = atoi(str);
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); //xllcorner = atof(str);
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); //yllcorner = atof(str);
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); //cellsize = atof(str);
  fscanf(f,"%s",&str); fscanf(f,"%s",&str); nodata = atof(str);
}

bool loadGrid2D(double *M, int rows, int columns, char *path)
{
  FILE *f = fopen(path, "r");

  if (!f) {
    printf("%s grid file not found\n", path);
    exit(0);
  }

  char str[STRLEN];
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < columns; j++)
    {
      fscanf(f, "%s", str);
      SET(M, columns, i, j, atof(str));
    }

  fclose(f);

  return true;
}

bool saveGrid2Dr(double *M, int rows, int columns, char *path)
{
  FILE *f;
  f = fopen(path, "w");

  if (!f)
    return false;

  char str[STRLEN];
  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < columns; j++)
    {
      sprintf(str, "%f ", GET(M, columns, i, j));
      fprintf(f, "%s ", str);
    }
    fprintf(f, "\n");
  }

  fclose(f);

  return true;
}

double* addLayer2D(int rows, int columns)
{
  double *tmp = (double *)malloc(sizeof(double) * rows * columns);
  if (!tmp)
    return NULL;
  return tmp;
}

// ----------------------------------------------------------------------------
// init kernel, called once before the simulation loop
// ----------------------------------------------------------------------------
void sciddicaTSimulationInit(int i, int j, int r, int c, double* Sz, double* Sh, double* Sf)
{
  double z, h;
  h = GET(Sh, c, i, j);

  if (h > 0.0)
  {
    z = GET(Sz, c, i, j);
    SET(Sz, c, i, j, z - h);
  }

  for (int n = 1; n < SIZE_OF_X; n++)
    BUF_SET(Sf, r, c, n-1, i, j, 0);
}

// ----------------------------------------------------------------------------
// computing kernels, aka elementary processes in the XCA terminology
// ----------------------------------------------------------------------------
void sciddicaTFlowsComputation(int i, int j, int r, int c, double nodata, int* Xi, int* Xj, double *Sz, double *Sh, double *Sf, double p_r, double p_epsilon)
{
  bool again, eliminated_cells[SIZE_OF_X];
  int cells_count;
  double average, m, u[SIZE_OF_X];
  

  for (int n = 0; n < SIZE_OF_X; n++)
    eliminated_cells[n] = false;

  m = GET(Sh, c, i, j) - p_epsilon;
  u[0] = GET(Sz, c, i, j) + p_epsilon;
  for (int n = 1; n < SIZE_OF_X; n++)
    u[n] = GET(Sz, c, i+Xi[n], j+Xj[n]) + GET(Sh, c, i+Xi[n], j+Xj[n]);

  do {
    again = false;
    average = m;
    cells_count = 0;

    for (int n = 0; n < SIZE_OF_X; n++)
      if (!eliminated_cells[n]) {
        average += u[n];
        cells_count++;
      }

    if (cells_count != 0)
      average /= cells_count;

    for (int n = 0; n < SIZE_OF_X; n++)
      if ((average <= u[n]) && (!eliminated_cells[n])) {
        eliminated_cells[n] = true;
        again = true;
      }
  } while (again);


  for (int n = 1; n < SIZE_OF_X; n++)
    !eliminated_cells[n] ? BUF_SET(Sf, r, c, n-1, i, j, (average - u[n]) * p_r) : BUF_SET(Sf, r, c, n-1, i, j, 0);
}

void sciddicaTWidthUpdate(int i, int j, int r, int c, double nodata, int* Xi, int* Xj, double *Sz, double *Sh, double *Sf)
{
  double h_next;

  h_next = GET(Sh, c, i, j);
  for (int n = 1; n < SIZE_OF_X; n++)
    h_next += BUF_GET(Sf, r, c, (SIZE_OF_X -1 - n), i+Xi[n], j+Xj[n]) - BUF_GET(Sf, r, c, n-1, i, j);
  
  SET(Sh, c, i, j, h_next);
}

// ----------------------------------------------------------------------------
// Function main()
// ----------------------------------------------------------------------------
int main(int argc, char **argv)
{
  int rows, cols;
  double nodata;
  readHeaderInfo(argv[HEADER_PATH_ID], rows, cols, nodata);

  int r = rows;                          // r: grid rows
  int c = cols;                          // c: grid columns
  int i_start = 1, i_end = r-1;          // [i_start,i_end[: kernels application range along the rows
  int j_start = 1, j_end = c-1;          // [i_start,i_end[: kernels application range along the rows
  double *Sz;                            // Sz: substate (grid) containing the cells' altitude a.s.l.
  double *Sh;                            // Sh: substate (grid) containing the cells' flow thickness
  double *Sf;                            // Sf: 4 substates containing the flows towards the 4 neighs
  int Xi[] = {0, -1,  0,  0,  1};        // Xj: neighborhood row coordinates (see below)
  int Xj[] = {0,  0, -1,  1,  0};        // Xj: neighborhood col coordinates (see below)
  double p_r = P_R;                      // p_r: minimization algorithm outflows dumping factor
  double p_epsilon = P_EPSILON;          // p_epsilon: frictional parameter threshold
  int steps = atoi(argv[STEPS_ID]);      //steps: simulation steps

  // The adopted von Neuman neighborhood
  // Format: flow_index:cell_label:(row_index,col_index)
  //
  //   cell_label in [0,1,2,3,4]: label assigned to each cell in the neighborhood
  //   flow_index in   [0,1,2,3]: outgoing flow indices in Sf from cell 0 to the others
  //       (row_index,col_index): 2D relative indices of the cells
  //
  //               |0:1:(-1, 0)|
  //   |1:2:( 0,-1)| :0:( 0, 0)|2:3:( 0, 1)|
  //               |3:4:( 1, 0)|
  //
  //

  Sz = addLayer2D(r, c);               // Allocates the Sz substate grid
  Sh = addLayer2D(r, c);               // Allocates the Sh substate grid
  Sf = addLayer2D((SIZE_OF_X-1)*r, c); // Allocates the Sf substates grid, having one layer for each adjacent cell

  loadGrid2D(Sz, r, c, argv[DEM_PATH_ID]);   // Load Sz from file
  loadGrid2D(Sh, r, c, argv[SOURCE_PATH_ID]);// Load Sh from file

  // Apply the init kernel (elementary process) to the whole domain grid (cellular space)
#pragma omp parallel for
    for (int i = i_start; i < i_end; i++)
      for (int j = j_start; j < j_end; j++)
        sciddicaTSimulationInit(i, j, r, c, Sz, Sh, Sf);

  util::Timer cl_timer;
  // simulation loop
  for (int s = 0; s < steps; ++s)
  {
  // Apply the FlowComputation kernel to the whole domain
#pragma omp parallel for
    for (int i = i_start; i < i_end; i++)
      for (int j = j_start; j < j_end; j++)
        sciddicaTFlowsComputation(i, j, r, c, nodata, Xi, Xj, Sz, Sh, Sf, p_r, p_epsilon);

  // Apply the WidthUpdate mass balance kernel to the whole domain
#pragma omp parallel for
    for (int i = i_start; i < i_end; i++)
      for (int j = j_start; j < j_end; j++)
        sciddicaTWidthUpdate(i, j, r, c, nodata, Xi, Xj, Sz, Sh, Sf);
  }
  double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
  printf("Elapsed time: %lf [s]\n", cl_time);

  saveGrid2Dr(Sh, r, c, argv[OUTPUT_PATH_ID]);// Save Sh to file

  printf("Releasing memory...\n");
  delete[] Sz;
  delete[] Sh;
  delete[] Sf;

  return 0;
}
