exe=./mpi_cuda_test.bin
deb=./exe_debug
N_Process=1
#mp=/autofs/nccs-svm1_sw/summit/.swci/1-compute/opt/spack/20180914/linux-rhel7-ppc64le/xl-16.1.1-3/spectrum-mpi-10.3.0.0-20190419-q75ow22tialepxlgxcs7esrptqwoheij/bin/mpirun
p0=0
p1=0
b=1
cucc= "$(shell which nvcc)" 
cc= "$(shell which mpicxx)"
g="$(shell which g++)"
commflags=-lcudart -L"$(shell dirname $(cucc))"/../lib64
deb_flag=-g -G
#cuflags= --cempiler-options -Wall -Xptxas -v -Xcudafe -\# --reseurce-usage 
  cuflags= --compiler-options -Wall -v -Xcudafe -\# --resource-usage  
#cuflags= --compiler-options -Wall -Xptxas -dlcm=cg ## disable L1 cache
objs	= $(patsubst %.cu,%.o,$(wildcard *.cu)) \
	$(patsubst %.cpp,%.o,$(wildcard *.cpp))

deps	= $(wildcard ./*.cuh) \

#disL1 = –Xptxas -v –dlcm=cg

%.o:%.cu $(deps)
	$(cucc) -c $(cuflags) $< -o $@


#lineflag=-w -lineinfo -std=c++11	
#%.o:%.cu $(deps)
#	$(cucc) -c $(cuflags) $(lineflag) $< -o $@

$(exe):$(objs)
	$(cc) $(objs) $(commflags) -o $(exe)


test:$(exe)
			#note summit seem to be only giving 4 GPU per node during interactive mode
			jsrun --nrs 1 --tasks_per_rs 1 --cpu_per_rs 1 --gpu_per_rs 4 --rs_per_host  1 $(exe) ../colbeg_11948.dat ../colend_11948.dat ../csr_11948.dat 1 128 0 0 1 4 4096
		 	# mpiexec -n 1 $(exe) ../colbeg_11948.dat ../colend_11948.dat ../csr_11948.dat 1 128 0 0 1 1 4096
clean:
	rm -rf *.o ${exe}


