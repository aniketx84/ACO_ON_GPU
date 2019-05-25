#include "utils.h"
#include "cub/cub.cuh"
#include "gputimer.h"
#include <curand_kernel.h>


using namespace std;

__device__ __host__ void reverse(int *A, int i, int j){
    for(int k = i+1, l = j; k<=(i+j)/2; k++,l--){
            int temp = A[k];
            A[k] = A[l];
            A[l] = temp;
    }
}



__global__ void tsp_tpred(city *cities, int *tour,long initcost,unsigned long long *dst_tid,long cit,long itr)
{
	long id,j,k;
	register long change,mincost=initcost,cost;
	long i=threadIdx.x+blockIdx.x*blockDim.x;
	if(i < cit)
	{
		for(k=0;k<itr;k++)
		{
			change = 0; cost=initcost;
			j=(i+1+k)%cit;
			change=distD(cities, tour[i], tour[j]) + distD(cities, tour[(i+1)%cit], tour[(j+1)%cit]) - distD(cities, tour[i], tour[(i+1)%cit]) - distD(cities, tour[j], tour[(j+1)%cit]);
			cost+=change;	
			if(cost < mincost)
			{
				mincost = cost;
				if(i < j)
					id = i * (cit-1)+(j-1)-i*(i+1)/2;	
				else
					id = j * (cit-1)+(i-1)-j*(j+1)/2;	

			}	 

		}
		if(mincost < initcost)
			 atomicMin(dst_tid, ((unsigned long long)mincost << 32) | id);
	}
}




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
    unsigned long long *dst_tid;
	cudaMallocManaged((void**)&dst_tid, sizeof(unsigned long long));
    for(int itter = 0; itter < 1000; ++itter){
        kernelTime.Start();
        cons_soln<1024><<<m,1024>>>(ants,total,cities,n,m);
        cudaDeviceSynchronize();
        cout<<cudaGetErrorString(cudaGetLastError())<<endl;
        //-----------------------------------parallel Two opt----------------------------------------------------------------
	*dst_tid = (((unsigned long long)ants[0].tour_length+1)  << 32) - 1;
	long sol = (n*(n-1))/2;
	long itr = floor(n/2.0);
	long dst;
	long previ;
	int blk,thrd;
	if(n<512){
		blk=1;
		thrd=n;
	}
	else{
		blk=(n-1)/512+1;
		thrd=512;
	}
    for(int i = 0; i < 200; ++i){
        dst = ants[i].tour_length;
	do{
		previ = dst;
		tsp_tpred<<<blk,thrd>>>(cities, ants[i].tour, dst, dst_tid, n, itr);
        cudaDeviceSynchronize();
        
		dst = *dst_tid>>32;
		int tid = *dst_tid & ((1ull << 32)-1);
		int x = n-2-floor((sqrt(8*(sol-tid-1)+1)-1)/2);
		int y = tid-x*(n-1)+(x*(x+1)/2)+1;
		if(dst < previ)
            reverse(ants[i].tour, x, y);
    }while(dst < previ);
    ants[i].tour_length = get_tour_length(ants[i].tour, cities, n);
}
	//-------------------------------------------------------------------------------------------------------------------



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
