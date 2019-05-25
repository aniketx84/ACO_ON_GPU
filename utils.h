#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <limits>

#define ALPHA 1
#define BETA 2
#define RHO 0.02
#define HEURISTIC(i, j) (1./(distD(cities, i, j)))

#define checkError(ans) { if(ans != cudaSuccess) std::cout<<cudaGetErrorString(ans)<<__LINE__<<std::endl; }


struct city{
  double x, y;
};

struct ant{
        int *tour;
        long tour_length;
        bool *visited;
};

extern int m, n;

extern double *pheromone, *total;
extern long optimal_solution;

__host__ __device__ long distD(city*, long, long);
void read_file(city**, int, char**);
long nn(city*);
void check_tour(int*);
void init_pheromone(double);
void compute_total_info(city*);
void evaporation(void);
void init_ants(ant**);
void update_pheromone(ant);
void check_pheromone_limits(double, double);
__device__ __host__ long get_tour_length(int*, city*, int);
/*float get_error(long);
void update_pheromone(int*, long);

*/
#endif

