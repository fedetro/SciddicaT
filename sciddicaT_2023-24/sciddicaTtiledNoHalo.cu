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
#define SIZE_OF_X 5 // dimensione della griglia
#define P_R 0.5     // la probabilità di deflusso
#define P_EPSILON 0.001 // parametro di attrito 
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

#define TILE_WIDTH 8

// legge le informazioni di intestazione da un file
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

// carica una griglia da un file
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

// salva una griglia su un file
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
// calcola gli indici i e j, verifica che siano all'interno dei limiti della griglia, e poi esegue le operazioni di inizializzazione della simulazione per la cella corrente.
void sciddicaTSimulationInit(int i, int j, int r, int c, double* Sz, double* Sh, double* Sf)
{
    double z, h;
    h = GET(Sh, c, i, j);

    if (h > 0.0)
    {
        z = GET(Sz, c, i, j);
        SET(Sz, c, i, j, z - h);
    }
    // Inizializza a zero tutti i flussi nella cella
	for (int n = 1; n < SIZE_OF_X; n++)
        BUF_SET(Sf, r, c, n-1, i, j, 0);
}

// ----------------------------------------------------------------------------
// computing kernels, aka elementary processes in the XCA terminology. calcola i flussi tra le celle adiacenti in base allo stato corrente del modello
// ----------------------------------------------------------------------------
void sciddicaTFlowsComputation(int r, int c, double *Sz, double *Sh, double *Sf, double p_r, double p_epsilon)
{
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  // Verifica se le coordinate sono all'interno della griglia
  if (i >= 0 && i < r && j >= 0 && j < c)
  {
    // Up, Left, Right, Down

    // Coordinate locali nel blocco per il thread
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Dichiarazione di memoria condivisa per la cache delle celle cioè per memorizzare le tile cells
    __shared__ double Sz_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ double Sh_shared[TILE_WIDTH][TILE_WIDTH];

    // Carico l'elemento principale nel tile con sincronizzazione. I thread del blocco copiano i dati dall'area globale della griglia (Sz e Sh) alla memoria condivisa
    Sz_shared[ty][tx] = GET(Sz, c, i, j);
    Sh_shared[ty][tx] = GET(Sh, c, i, j);
    __syncthreads();

    // Variabili e array per il calcolo dei flussi
    bool again, eliminated_cells[SIZE_OF_X];
    int cells_count;
    double average, m, u[SIZE_OF_X];
    double z, h;
    
    for (int n = 0; n < SIZE_OF_X; n++)
        eliminated_cells[n] = false;


/*  m = GET(Sh, c, i, j) - p_epsilon;
    u[0] = GET(Sz, c, i, j) + p_epsilon;
    for (int n = 1; n < SIZE_OF_X; n++)
        u[n] = GET(Sz, c, i+Xi[n], j+Xj[n]) + GET(Sh, c, i+Xi[n], j+Xj[n]); */

// Inizializzazione di m e u con i valori della cella principale
    m = Sh_shared[ty][tx] - p_epsilon;
    u[0] = Sz_shared[ty][tx] + p_epsilon;

// Calcolo dei flussi verso le celle adiacenti (UP, LEFT, RIGHT, DOWN) con gestione delle celle di bordo

    // UP
    if (ty > 0)
    { // Fetch from shared
        z = Sz_shared[ty - 1][tx];
        h = Sh_shared[ty - 1][tx];
    }
    else if (ty == 0)
    { // Fetch from global
        z = GET(Sz, c, i - 1, j);
        h = GET(Sh, c, i - 1, j);
    }
    u[1] = z + h;

    // LEFT

    if (tx > 0)
    { // Fetch from shared
        z = Sz_shared[ty][tx - 1];
        h = Sh_shared[ty][tx - 1];
    }
    else if (tx == 0)
    { // Fetch from global
        z = GET(Sz, c, i, j - 1);
        h = GET(Sh, c, i, j - 1);
    }
    u[2] = z + h;

    // RIGHT
    // se il thread corrente è minore o uguale al limite destro del blocco 
    if (tx <= blockDim.x - 2)
    { // Fetch from shared
        z = Sz_shared[ty][tx + 1];
        h = Sh_shared[ty][tx + 1];
    } // Se il thread è al limite inferiore del blocco
    else if (tx == blockDim.x - 1)
    { // Fetch from global
        z = GET(Sz, c, i, j + 1);
        h = GET(Sh, c, i, j + 1);
    }
    u[3] = z + h;

    // DOWN

    if (ty <= blockDim.y - 2)
    { // Fetch from shared
        z = Sz_shared[ty + 1][tx];
        h = Sh_shared[ty + 1][tx];
    }
    else if (ty == blockDim.y - 1)
    { // Fetch from global
        z = GET(Sz, c, i + 1, j);
        h = GET(Sh, c, i + 1, j);
    }
    u[4] = z + h;

    // eliminazione dei flussi che superano la media calcolata
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

    // Aggiornamento della larghezza del flusso in ciascuna cella
    for (int n = 1; n < SIZE_OF_X; n++)
        !eliminated_cells[n] ? BUF_SET(Sf, r, c, n-1, i, j, (average - u[n]) * p_r) : BUF_SET(Sf, r, c, n-1, i, j, 0);
  }
}

// aggiorna lo spessore del flusso in ciascuna cella in base ai flussi calcolati nella fase precedente.
void sciddicaTWidthUpdate(int r, int c, double *Sz, double *Sh, double *Sf)
{
    // Coordinate globali per il thread
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

    // Coordinate locali nel blocco per il thread
	int tx = threadIdx.x;
	int ty = threadIdx.y;

    // Dichiarazione di memoria condivisa per la cache dei flussi
	__shared__ double Sf_shared[TILE_WIDTH][TILE_WIDTH][(SIZE_OF_X-1)];
	Sf_shared[ty][tx][0] = BUF_GET(Sf, r, c, 0, i, j);
	Sf_shared[ty][tx][1] = BUF_GET(Sf, r, c, 1, i, j);
	Sf_shared[ty][tx][2] = BUF_GET(Sf, r, c, 2, i, j);
	Sf_shared[ty][tx][3] = BUF_GET(Sf, r, c, 3, i, j);
	__syncthreads();

	if (i >= 0 && i < r && j >= 0 && j < c)
	{
        // Calcolo della variazione dello spessore del flusso
		double h_next;
		h_next = GET(Sh, c, i, j);

		// accessi safe
		double b0 = Sf_shared[ty][tx][0];
		double b1 = Sf_shared[ty][tx][1];
		double b2 = Sf_shared[ty][tx][2];
		double b3 = Sf_shared[ty][tx][3];

		// accessi unsafe
		double bUp;
		if (ty > 0)
		{ // Fetch from shared
			bUp = Sf_shared[ty - 1][tx][3];
		}
		else if (ty == 0)
		{ // Fetch from global
			bUp = BUF_GET(Sf, r, c, 3, i - 1, j);
		}

		double bLeft;
		if (tx > 0)
		{ // Fetch from shared
			bLeft = Sf_shared[ty][tx - 1][2];
		}
		else if (tx == 0)
		{ // Fetch from global
			bLeft = BUF_GET(Sf, r, c, 2, i, j - 1);
		}

		double bRight;
		if (tx <= blockDim.x - 2)
		{ // Fetch from shared
			bRight = Sf_shared[ty][tx + 1][1];
		}
		else if (tx == blockDim.x - 1)
		{ // Fetch from global
			bRight = BUF_GET(Sf, r, c, 1, i, j + 1);
		}

		double bDown;
		if (ty <= blockDim.y - 2)
		{ // Fetch from shared
			bDown = Sf_shared[ty + 1][tx][0];
		}
		else if (ty == blockDim.y - 1)
		{ // Fetch from global
			bDown = BUF_GET(Sf, r, c, 0, i + 1, j);
		}

		// halo cells
		h_next += bUp - b0;	// UP
		h_next += bLeft - b1;  // LEFT
		h_next += bRight - b2; // RIGHT
		h_next += bDown - b3;  // DOWN

        SET(Sh, c, i, j, h_next);
    }
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
  double *Sz;                            // Sz: substate (grid) containing the cells' altitude a.s.l. sopra il livello del mare
  double *Sh;                            // Sh: substate (grid) containing the cells' flow thickness (spessore del flusso delle celle)
  double *Sf;                            // Sf: 4 substates containing the flows towards the 4 neighs (contenente i flussi verso i 4 vicini)
  int Xi[] = {0, -1,  0,  0,  1};        // Xj: neighborhood row coordinates (see below)
  int Xj[] = {0,  0, -1,  1,  0};        // Xj: neighborhood col coordinates (see below)
  double p_r = P_R;                      // p_r: minimization algorithm outflows dumping factor. Fattore di smorzamento degli effluenti
  double p_epsilon = P_EPSILON;          // p_epsilon: frictional parameter threshold. Soglia del parametro di attrito
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

  Sz = addLayer2D(r, c);               // Allocates the Sz substate grid. altitudine
  Sh = addLayer2D(r, c);               // Allocates the Sh substate grid. spessore del flusso
  Sf = addLayer2D((SIZE_OF_X-1)*r, c); // Allocates the Sf substates grid, having one layer for each adjacent cell. flussi

  loadGrid2D(Sz, r, c, argv[DEM_PATH_ID]);   // Load Sz from file
  loadGrid2D(Sh, r, c, argv[SOURCE_PATH_ID]);// Load Sh from file

  // Apply the init kernel (elementary process) to the whole domain grid (cellular space)
  for (int i = 0; i < r; i++)
      for (int j = 0; j < c; j++)
          sciddicaTSimulationInit(i, j, r, c, Sz, Sh, Sf);

  /* Allocation of Cuda Elements */
  double *Sz_device;
  cudaMalloc(&Sz_device, sizeof(double) * r * c);
  double *Sh_device;
  cudaMalloc(&Sh_device, sizeof(double) * r * c);
  double *Sf_device;
  cudaMalloc(&Sf_device, sizeof(double) * r * c * (SIZE_OF_X-1));

  cudaMemcpy(Sz_device, Sz, sizeof(double) * r * c, cudaMemcpyHostToDevice);
  cudaMemcpy(Sh_device, Sh, sizeof(double) * r * c, cudaMemcpyHostToDevice);
  cudaMemcpy(Sf_device, Sf, sizeof(double) * r * c * (SIZE_OF_X-1), cudaMemcpyHostToDevice);

  util::Timer cl_timer;
  // simulation loop

  double dimB = 8.0f;
  dim3 dimGrid(ceil(r / dimB), ceil(c / dimB), 1);
  dim3 dimBlock(dimB, dimB, 1);

  for (int s = 0; s < steps; ++s)
  {
    sciddicaTFlowsComputation<<<dimGrid, dimBlock>>>(r, c, Sz_device, Sh_device, Sf_device, p_r, p_epsilon);
	cudaDeviceSynchronize();

    sciddicaTWidthUpdate<<<dimGrid, dimBlock>>>(r, c, Sz_device, Sh_device, Sf_device);
    cudaDeviceSynchronize();
  }

  double cl_time = static_cast<double>(cl_timer.getTimeMilliseconds()) / 1000.0;
  cudaMemcpy(Sh, Sh_device, sizeof(double) * r * c, cudaMemcpyDeviceToHost);

  cudaMemcpy(Sf, Sf_device, sizeof(double) * r * c * (SIZE_OF_X-1), cudaMemcpyDeviceToHost);

  printf("Elapsed time: %lf [s]\n", cl_time);
  saveGrid2Dr(Sh, r, c, argv[OUTPUT_PATH_ID]);// Save Sh to file

  printf("Releasing memory...\n");
  cudaFree(Sh_device);
  cudaFree(Sz_device);
  cudaFree(Sf_device);
  delete[] Sz;
  delete[] Sh;
  delete[] Sf;

  return 0;
}
