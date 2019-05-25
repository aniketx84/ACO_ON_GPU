# ACO_ON_GPU
This Repository contains the code for the parallel implemetation of MMAS forlarge TSP instances.
It can handle upto 35000 city instances.
There are three files namely data_parallel.cu, task_base and two_opt. It contains the data parallel approach, task parallel apporach and data parallel approach with two-opt local search repectively.
## Compiling
By default the sm_arch is set to 60 edit the make file to change it.
Run make to compile the files.
## Running
There will be three output files
data_pralallel, task_based and two_opt. Any can be run by the following syntax:
./<algorithm_name> <instance_location> <Number_of_ants> <optimal_solution(optional)>
the last parameter is used to calculate the error rate.
example:
./data_parallel instances/rat575 575 6773
