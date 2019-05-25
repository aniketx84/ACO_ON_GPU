#include "utils.h"
#include "cub/cub.cuh"
#include "gputimer.h"
#include <curand_kernel.h>


using namespace std;

__global__ void cons_soln(ant *ants, double *total, city *cities, int n, int m){
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx < m){
        //reseting the ants
        for(int i = 0; i < n; ++i) ants[idx].visited[i] = false;

        //Placing ants on initial city
        curandState rndstate;
        curand_init(idx, 0, 0, &rndstate);
        int start = curand(&rndstate) % (n);
        ants[idx].tour[0] = start;
        ants[idx].tour[n] = start;
        ants[idx].visited[start] = true;
        
        //constructing the solution
        int step = 1;
        double max_value = 0;
        while(step < n){
            int current = ants[idx].tour[step-1];
            int next = n;
            for(int i = 0; i < n; ++i){
                if(ants[idx].visited[i] == false && total[current * n + i] > max_value){
                    max_value = total[current * n + i];
                    next = i;
                }
            }
            ants[idx].tour[step] = next;
            ants[idx].visited[next] = true;
            max_value = 0;
            step++; 
        }
        ants[idx].tour_length = get_tour_length(ants[idx].tour, cities, n);
    }
    
}

int main(int argc, char **argv){
    city *cities;
    ant *ants, *global_best_ant;

    GpuTimer kernelTime;

    cudaMallocManaged((void**)&global_best_ant, sizeof(ant));
    cudaMallocManaged((void**)&(global_best_ant->tour), sizeof(int)*(n+1));
    int itter_best_ant_pos;
    double global_best_ants_value = LONG_MAX, itter_best_ant_value = LONG_MAX;
    read_file(&cities, argc, argv);
    init_ants(&ants);
    double t_max,t_min;
    t_max = 1/(RHO*nn(cities));
    t_min = t_max/(2*n);
    cout<<t_max<<"\t"<<t_min<<endl;
    init_pheromone(t_max);
    compute_total_info(cities);
    for(int itter = 0; itter < 10000; ++itter){
        kernelTime.Start();
        cons_soln<<<(m-1)/512.0 + 1,512>>>(ants,total,cities,n,m);
        cudaDeviceSynchronize();
        
        /*for(int i = 0; i < m; ++i){
            for(int j = 0; j <= n; ++j){
                cout<<ants[i].tour[j]<<" ";
            }
            cout<<"Tour length: "<<ants[i].tour_length;
            cout<<endl;
        }
        for(int i = 0; i < m; ++i){
            check_tour(ants[i].tour);
        }*/

        //Finding the best ant
        for(int i = 0; i < m; ++i){
            if(itter_best_ant_value > ants[i].tour_length){
                itter_best_ant_pos = i;
                itter_best_ant_value = ants[i].tour_length;
            }
        }

        if(ants[itter_best_ant_pos].tour_length < global_best_ants_value){
            global_best_ant->tour_length = ants[itter_best_ant_pos].tour_length;
            global_best_ants_value = ants[itter_best_ant_pos].tour_length;
            for(int i = 0; i <= n; ++i){
                global_best_ant->tour[i] = ants[itter_best_ant_pos].tour[i];
            }
        }

       // cout<<"Itter best: "<<itter_best_ant_value<<"\t"<<"Global best: "<<global_best_ant->tour_length<<endl;
        evaporation();
        if(itter%10)
            update_pheromone(ants[itter_best_ant_pos]);
        else
            update_pheromone(*global_best_ant);
        compute_total_info(cities);
        kernelTime.Stop();
        check_pheromone_limits(t_max, t_min);
        double error = (global_best_ant->tour_length-optimal_solution)/(float)optimal_solution * 100;
        cout<<itter<<","<<kernelTime.Elapsed()<<","<<ants[itter_best_ant_pos].tour_length<<","<<global_best_ant->tour_length<<","<<error<<endl;
    }
    return 0;
}
