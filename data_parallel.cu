#include "utils.h"
#include "cub/cub.cuh"
#include "gputimer.h"
#include <curand_kernel.h>


using namespace std;
template<
        int threads>
__global__ void cons_soln(ant *ants, double *total, city *cities, int n, int m){
    //reseting the ants
    if(threadIdx.x == 0){
        for(int i = 0; i < n; ++i) ants[blockIdx.x].visited[i] = 1;
    }

    typedef cub::BlockReduce<cub::KeyValuePair<int, double>, threads> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    //placing the ants at initital city
    if(threadIdx.x == 0){
        curandState rndstate;
        curand_init(blockIdx.x, 0, 0, &rndstate);
        int start = curand(&rndstate) % (n);
        ants[blockIdx.x].tour[0] = start;
        ants[blockIdx.x].tour[n] = start;
        ants[blockIdx.x].visited[start] = 0;
    }
    __syncthreads();

    cub::KeyValuePair<int, double> max, t1[35];
    int step = 1;
    int curr;
    while(step < n){
        curr = ants[blockIdx.x].tour[step-1];
        for(int i = 0; i < 35; ++i){
		    if(threadIdx.x*35+i < n){
                	t1[i].value = total[curr * n + threadIdx.x*35 + i] * ants[blockIdx.x].visited[threadIdx.x * 35 + i];
                	t1[i].key = threadIdx.x * 35 + i;
		    }
		    else{
			    t1[i].key = -1;
			    t1[i].value = 0;	
		    }
        }
        max = BlockReduce(temp_storage).Reduce(t1, cub::ArgMax());
	__syncthreads();
	if(threadIdx.x == 0){
		ants[blockIdx.x].tour[step] = max.key;
		ants[blockIdx.x].visited[max.key] = false;
	}
	step++;
    }
    __syncthreads();

    if(threadIdx.x == 0){
	    ants[blockIdx.x].tour_length = get_tour_length(ants[blockIdx.x].tour, cities, n);
    }
    
}

int main(int argc, char **argv){
    city *cities;
    ant *ants;
    
    GpuTimer kernelTime;

    ant *global_best_ant;
    cudaMallocManaged((void**)&global_best_ant, sizeof(ant));
    cudaMallocManaged((void**)&(global_best_ant->tour), sizeof(int)*(n+1));
    global_best_ant->tour_length = LONG_MAX;
    int itter_best_pos = 0;


    read_file(&cities, argc, argv);
    init_ants(&ants);
    double t_max,t_min;
    t_max = 1/(RHO*nn(cities));
    t_min = t_max/(2*n);
    cout<<t_max<<"\t"<<t_min<<endl;
    init_pheromone(t_max);
    compute_total_info(cities);
    for(int itter = 0; itter < 1000; ++itter){
        kernelTime.Start();
        cons_soln<1024><<<m,1024>>>(ants,total,cities,n,m);
        cudaDeviceSynchronize();

        long min_tour = ants[itter_best_pos].tour_length;
        for(int i = 0; i < m; ++i){
            if(min_tour > ants[i].tour_length){
                min_tour = ants[i].tour_length;
                itter_best_pos = i;
            }
        }
        if(ants[itter_best_pos].tour_length < global_best_ant->tour_length){
            global_best_ant->tour_length = ants[itter_best_pos].tour_length;
            for(int i = 0; i <= n; ++i){
                global_best_ant->tour[i] = ants[itter_best_pos].tour[i];
            }
            t_max = 1.0/(RHO*global_best_ant->tour_length);
            t_min = t_max/(2*n);

        }

        evaporation();
        if(!itter%10)
            update_pheromone(*global_best_ant);
        else
            update_pheromone(ants[itter_best_pos]);
        compute_total_info(cities);
        kernelTime.Stop();
        check_pheromone_limits(t_max, t_min);

        double error = (global_best_ant->tour_length-optimal_solution)/(float)optimal_solution * 100;
        cout<<itter<<","<<kernelTime.Elapsed()<<","<<ants[itter_best_pos].tour_length<<","<<global_best_ant->tour_length<<","<<error<<endl;
        

    }
    return 0;
}
