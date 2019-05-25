#include "utils.h"


int m, n;

double *total, *pheromone;

long optimal_solution;

__host__ __device__ long distD(city *cities, long i , long j)
{
        float d1=cities[i].x - cities[j].x;
        float d2=cities[i].y - cities[j].y;
        return(sqrtf( (d1*d1) + (d2*d2) ));
}

__device__ __host__ long get_tour_length(int *tour, city* cities, int n){
	long tl = 0;
	for(int i = 0; i < n; ++i) tl+=distD(cities, tour[i], tour[i+1]);
	return tl;

}

void check_tour(int *tour){
        int *temp = (int*)calloc(n, sizeof(int));
	int flag = 0;
        for(int i = 0; i < n; ++i){
                temp[tour[i]]++;
        }
        for(int i = 0; i < n; ++i){
                if(temp[i]>1 || temp[i] == 0){
                        std::cout<<"Error at:"<<i<<std::endl;
			flag = 1;
                        break;
                }
        }
	if(flag == 0){
		std::cout<<"Tour ok"<<std::endl;
	}
        free(temp);
}


long nn(city *cities){
	int *tour = (int*)malloc(sizeof(long)*(n+1));
        srand(time(NULL));
        bool *visited = (bool*)malloc(sizeof(bool)*n);
        for (int i = 0; i < n; ++i)
        {
                visited[i] = false;
        }
	//srand(clock());
        int start_city = rand()%n;
        tour[0] = start_city;
        visited[start_city] = true;

        long next_city = n;
        int step = 1;
        double min_dist = std::numeric_limits<double>::max();
        while(step < n){
                for (int i = 0; i < n; ++i)
                {
                        if(visited[i] == false){
                                if(distD(cities, tour[step-1], i)< min_dist){
                                        min_dist = distD(cities, tour[step-1], i);
                                        next_city = i;
                                }
                        }

                }
                tour[step] = next_city;
                visited[next_city] = true;
                step++;
                min_dist = LONG_MAX;
        }
        tour[n] = tour[0];
	return get_tour_length(tour, cities, n);
}


void read_file(city **cities, int argc, char** argv){
        std::ifstream infile;
        infile.open(argv[1], std::ios::in);
        if(!infile){
                std::cerr<<"Error opening file "<<argv[1]<<"..!!"<<std::endl;
                exit(1);
        }

	m = std::stoi(argv[2]);
	optimal_solution = std::stoi(argv[3]);
        char temp[1024];
        infile.getline(temp,1024,'\n');
        infile.getline(temp,1024,'\n');
        infile.getline(temp,1024,'\n');
        infile.getline(temp,1024,':');
        infile>>n;
        int test;
        cudaMallocManaged(cities, sizeof(city)*n);
        infile.getline(temp,1024,'\n');
        infile.getline(temp,1024,'\n');
        infile.getline(temp,1024,'\n');

        for (int i = 0; i < n; ++i)
        {
                infile>>test>>(*cities)[i].x>>(*cities)[i].y;
        }

}

void init_ants(ant **ants){
        cudaMallocManaged((void**)ants, sizeof(ant)*m);
        for(int i = 0; i < m; ++i){
                cudaMallocManaged((void**)&((*ants)[i].visited),sizeof(bool)*n);
                cudaMallocManaged((void**)&((*ants)[i].tour),sizeof(int)*(n+1));
        }
}

void init_pheromone(double t_0){
        cudaMallocManaged((void**)&pheromone,n*n*sizeof(double));
        cudaMallocManaged((void**)&total, sizeof(double)*n*n);
        for(int i = 0; i < n; ++i){
                for(int j = 0; j <= i; ++j){
                        pheromone[i*n+j] = t_0;
                        pheromone[j*n+i] = pheromone[i*n+j];
                }
        }       
}

void compute_total_info(city *cities){
        for(int i = 0; i < n; ++i){
                for(int j = 0; j < i; ++j){
                        total[i*n+j] = pow(pheromone[i*n+j],ALPHA) * pow(HEURISTIC(i,j),BETA);
                        total[j*n+i] = total[i*n+j];
                }
        }
}

void evaporation(void){
        for(int i = 0; i < n; ++i){
                for(int j = 0; j < n; ++j){
                        pheromone[i*n+j] *=(1-RHO);
                }
        }       
}

void update_pheromone(ant a){
        double d = 1.0 / a.tour_length;
        for(int i = 0; i < n; ++i){
                pheromone[a.tour[i]*n+a.tour[i+1]] += d;
        }
}


void check_pheromone_limits(double t_max, double t_min){
        for(int i = 0; i < n; ++i){
                for(int j = 0; j < n; ++j){
                        if(pheromone[i*n+j] > t_max) pheromone[i*n+j] = t_max;
                        if(pheromone[i*n+j] < t_min) pheromone[i*n+j] = t_min;
                }
        }
}
