main: data_parallel.cu task_based.cu two_opt.cu utils.cu
	nvcc -std=c++11 -arch=sm_60 -dc utils.cu
	nvcc -std=c++11 -arch=sm_60 -dc data_parallel.cu
	nvcc -std=c++11 -arch=sm_60 -dc task_based.cu
	nvcc -std=c++11 -arch=sm_60 -dc two_opt.cu
	nvcc -arch=sm_60 data_parallel.o utils.o -o data_parallel
	nvcc -arch=sm_60 task_based.o utils.o -o task_based
	nvcc -arch=sm_60 two_opt.o utils.o -o two_opt
clean:
	rm -rf *.o
	rm data_parallel task_based two_opt