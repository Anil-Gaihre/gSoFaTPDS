#include <stdio.h>
#include <string.h>
#include <mpi.h>
// #include "/home/gaihre/openmpi-4.0.4/ompi/include/mpi.h"
//#include "adder.cuh"
#include <stdlib.h>
#include <assert.h>
#include <iostream>
// #include "symbfact_two_sided_explicit_target.cuh"
// #include "symbfact_two_sided_change_scheduling.cuh"
#include "symbfact_warp_centric_new_scheduling.cuh"
// #include "symbfact_warp_centric_new_scheduling_stealing.cuh"
// #include "symbfact_warp_centric_new_scheduling_scaling.cuh"
 
//#include "symbfact_reordered.cuh"
#include "wtime.h"
#include <cuda.h>
#include <fstream>
int count=0;
int main(int argc, char *argv[])
{
    char message[20];
    int myrank, tag=99; 
    cout<<"Number of Processes: "<<argv[4]<<endl;
    unsigned int global_fill_in=0;
    int global_N_supernode=0;
    ull_t global_Process_TEPS=0;
    double max_time=0;
    double min_time=0;
    double begin_time=wtime();
     
    MPI_Status status;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    ull_t fill_count=0;
    int N_supernodes=0;
    double indiv_time=0; 
    ull_t Process_TEPS =0;
    //  myrank=0; 
    symbfact_min_id(argc,argv,myrank,fill_count,indiv_time,N_supernodes,Process_TEPS);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&fill_count, &global_fill_in, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&N_supernodes, &global_N_supernode, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&indiv_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&indiv_time, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&Process_TEPS, &global_Process_TEPS, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD); 
  if (myrank == 0)
    {
         std::fstream timeLog;
         timeLog.open("Vary_percent_blocksi_pre2.csv",std::fstream::out | std::fstream::app);
        cout<<"Final Number of fill-ins detected: "<<global_fill_in<<endl;
        cout<<"Final Number of Supernodes detected: "<<global_N_supernode<<endl;
        cout<<"Final Maximum time reported among the processes: "<<max_time<<" ms"<<endl;
        cout<<"Final Minimum time reported among the processes: "<<min_time<<" ms"<<endl;
        cout<<"Final TEPS according to slowest process: "<<global_Process_TEPS/((float)(max_time/(float)1000))<<endl;
         timeLog<<"vert_count;N_gpus;"<<"p0;"<<"p1"<<";"<<"#blocks_src"<<";"<<"min_time"<<";"<<"max_time"<<endl;
    timeLog<<argv[2]<<";"<<argv[4]<<";"<<argv[6]<<";"<<argv[7]<<";"<<argv[8]<<";"<<min_time<<";"<<max_time<<endl;
        timeLog.close();
    }
   MPI_Finalize();
    return 0;
}