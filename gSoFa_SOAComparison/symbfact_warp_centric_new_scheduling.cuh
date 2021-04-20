#include "graph.h"
#include "wtime.h"
#include "barrier.cuh"
#include <omp.h>
#include <cuda.h>
#include <math.h> 
#include <cstdio>
#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include <fstream>
#include <cstring>
#include <string>
#include <stdint.h> 
// #include "Compute_fill_ins.cuh" //For byte addressing
#include <vector>
// #include <math.h>

#define _M_LN2  0.693147180559945309417 // Natural log of 2
#define log_2(x) (log(x)/_M_LN2)
 
using namespace std;
typedef int int_t;
typedef unsigned int uint_t;
typedef unsigned long long ull_t;
typedef std::vector<int> vec;
// #define test 1
// #define profile_frontier_sizes 1 
// #define maintain_minimum_max_id_fills 1

#define MAX_VAL UINT_MAX
#define num_t 24
#define MIN_VAL 0
//#define num_blocks 128
#define num_blocks_cat0 128
#define block_size_cat0 128
#define num_blocks_cat1 128
#define block_size_cat1 128
//#define block_size 128
#define chunk_size_0 128
#define chunk_size_1 128
#define frontier_multiple 0.5
#define profile_dumping_loading_time 1
#define profile_memcpy_bandwidth 1
// #define profile_stealing_overhead 1
// #define  print_status 1
#define profile_TEPS 1
#define enable_memadvise 1
// #define disable_maxID_optimization 1

// #define thread_centric 1

// #define lambda 1
// #define allocated_frontier_size 306749905

// #define sequential_supernodal_expansion 1
// #define enable_transpose
// #define pre_processing 1

#define detect_supernodes 1
#define threshold 1
#define debug_fill_count

// #define chunk_scheduling 1
// #define superNode_traversal
#define enable_sort_edge_list

// #define check_termination
// #define workload_stealing 1
// #define RDMA 1

// #define profile_fill_count_row 1
// #define profile_edge_checks 1



#define allocated_frontier_size 306749905 //For N_source: 1024 pre2 dataset allocation (50% worse space complexity possible)


#define overwrite_kernel_config 1
// #define enable_debug 1


////////////-----------------------//////////////
//////////// Only 1 of the following two variables should be defined. Not both///////////////////////////////
// #define enable_fillins_filter_FQ_No_Max_id_update 1
#define enable_fillins_filter_FQ 1 //This optimization performs better than the default
////////////----------------------/////////////////

#ifdef workload_stealing
MPI_Datatype response_type;
MPI_Datatype request_type;
MPI_Datatype token_type;

enum uts_tags { MPIWS_WORKREQUEST = 1, MPIWS_WORKRESPONSE, MPIWS_TDTOKEN, MPIWS_STATS };
enum color{white=0, black, pink, red};

static MPI_Request  wrin_request;  // Incoming steal request
static MPI_Request  wrout_request; // Outbound steal request
static MPI_Request  iw_request;    // Outbound steal request

static MPI_Request  wrin_request_token;  // Incoming token request
static MPI_Request  wrout_request_token; // Outbound token response

long        chunks_recvd;  // Total messages received //Increased when a work (chunk) is received 
long        chunks_sent;   // Total messages sent //Increased when a work (chunk) is sent(released) 
long        ctrl_recvd;    // Total messages received //Increased when a work REQUEST is recieved and when a no work message is received
long        ctrl_sent;     // Total messages sent //increased whrn no work reponse is sent and when requesting a work

static enum color my_color;      // Ring-based termination detection 

struct token
{
    enum color mycolor;
    int N_sent_chunks;
    int N_received_chunks;
};

struct token terminating_token;
// void CreateResponseType(MPI_Datatype& response_type)
void CreateResponseType()
{
    // Reference: https://www.rookiehpc.com/mpi/docs/mpi_type_create_struct.php
    /**
     * @brief Illustrates how to create an indexed MPI datatype.
     * @details This program is meant to be run with 2 processes: a sender and a
     * receiver. These two MPI processes will exchange a message made of a
     * structure representing a person.
     *
     * Structure of a person:
     * - age: int
     * - height: double
     * - name: char[10]
     *
     * How to represent such a structure with an MPI struct:
     *   
     *        +------------------ displacement for
     *        |         block 2: sizeof(int) + sizeof(double)
     *        |                         |
     *        +----- displacement for   |
     *        |    block 2: sizeof(int) |
     *        |            |            |
     *  displacement for   |            |
     *    block 1: 0       |            |
     *        |            |            |
     *        V            V            V
     *        +------------+------------+------------+
     *        |    age     |   height   |    name    |
     *        +------------+------------+------------+
     *         <----------> <----------> <---------->
     *            block 1      block 2      block 3
     *           1 MPI_INT  1 MPI_DOUBLE  10 MPI_CHAR
     **/

    int lengths[4] = {1,1,1,1};
    const MPI_Aint displacements[4] = { 0, sizeof(int), 2*sizeof(int) ,3*sizeof(int) };
    MPI_Datatype types[4] = { MPI_INT, MPI_INT, MPI_INT, MPI_INT};
    MPI_Type_create_struct(4, lengths, displacements, types, &response_type);
    MPI_Type_commit(&response_type);
}

void CreateRequestType()
{
    int lengths[1] = {1};
    const MPI_Aint displacements[1] = { 0 };
    MPI_Datatype types[1] = { MPI_INT};
    MPI_Type_create_struct(1, lengths, displacements, types, &request_type);
    MPI_Type_commit(&request_type);
}

// enum color{white, black, red};
struct request_package
{
    int thief_id;

};
struct response_package
{
    int victim_id;
    int thief_id;
    int work; // work=0 if not work else work =1
    int stolen_chunk_id;

};



void Initialize(int myrank, int num_process)
{
    wrout_request = MPI_REQUEST_NULL;
    iw_request    = MPI_REQUEST_NULL;
    wrin_request = MPI_REQUEST_NULL;
    // Incoming token request

    chunks_recvd =0 ;  // Total workload received
    chunks_sent =0;   // Total workload sent
    ctrl_recvd =0;    // Total messages received
    ctrl_sent=0;     // Total messages sent
    terminating_token.mycolor = black;
    terminating_token.N_sent_chunks =0;
    terminating_token.N_received_chunks =0;
    my_color =  white;
    if (myrank ==0)
    {
        //Process 0 has the initial token
        wrin_request_token = MPI_REQUEST_NULL;
        wrout_request_token = MPI_REQUEST_NULL; // Outbound token response
    }
    else
    {
        //other processes wait to receive a token from the left and pass along the right processes
        int left_process = (myrank == 0) ? num_process - 1 : myrank - 1; // Receive the token from the processor to your left

        MPI_Irecv(&terminating_token, 1, token_type, left_process, MPIWS_TDTOKEN, MPI_COMM_WORLD, &wrin_request_token);
        cout<<"Process "<<myrank<< " is WAITING FOR THE TERMINATING TOKEN FROM PROCESS "<<left_process<<endl;


    }
}

void  ReceiveStealingRequest (int myrank, int& tail_pointer, int& last_stolen_bigchunk,  struct request_package& recv_request,int source_process)
{
    // cout<<"Inside receive stealing request!"<<endl;
    MPI_Status status;

    if (wrin_request == MPI_REQUEST_NULL)
    {

        int receive_request;
        // cout<<"Phase I: Start receiving stealing request!"<<endl;
        // MPI_Irecv(&receive_request, 1, MPI_INT, MPI_ANY_SOURCE, MPIWS_WORKREQUEST, MPI_COMM_WORLD, &wrin_request);
        // cout<<"Phase II: Start receiving stealing request!"<<endl;
        MPI_Irecv(&recv_request, 1, request_type, MPI_ANY_SOURCE, MPIWS_WORKREQUEST, MPI_COMM_WORLD, &wrin_request);
    }

}


void ResponseStealing (int myrank, int& tail_pointer, int& last_stolen_bigchunk,struct request_package& recv_request,int source_process, int& group, int N_gpu, int& N_groups)
{

    MPI_Status status;
    int         flag, index;
    void       *work;

    MPI_Test(&wrin_request, &flag, &status);

    if (flag)
    {
        ctrl_recvd++;// Got a work request
        // cout<<"STEALING PROCESS "<<status.MPI_SOURCE<<endl;
        //  MPI_Irecv(&recv_request, 1, request_type, MPI_ANY_SOURCE, MPIWS_TDTOKEN, MPI_COMM_WORLD, &wrin_request);

        struct response_package send_response;
        send_response.victim_id = myrank;
        send_response.thief_id = recv_request.thief_id;
        if (send_response.thief_id < myrank)
        {
            /* If a node to our left steals from us, our color becomes black */
            my_color = black;
        }
        send_response.work = 12;
        if (tail_pointer > threshold)
        {
            tail_pointer--;
            chunks_sent++;
            send_response.stolen_chunk_id = last_stolen_bigchunk--;
            // N_groups -= N_gpu;
            // group += N_gpu;
        }
        else
        {
            //sending a no work request
            send_response.stolen_chunk_id = -1;
            ctrl_sent++;//Also increased when sending a no work response
        }
#ifdef print_status
        printf("MPI process %d sends victim_id = %d\n\t- thief_id = %d\n\t- work = %d\n\t- stolen_chunk_id = %d\n", myrank, send_response.victim_id, send_response.thief_id , send_response.work,send_response.stolen_chunk_id);
#endif
        MPI_Isend(&send_response, 1, response_type, status.MPI_SOURCE, MPIWS_WORKRESPONSE, MPI_COMM_WORLD, &wrout_request);
        // MPI_Isend(&send_response, 1, response_type, send_response.thief_id, MPIWS_WORKRESPONSE, MPI_COMM_WORLD, &wrout_request);

        // MPI_Send(&send_response, 1, response_type, 0, 123, MPI_COMM_WORLD);
    }
    else
    {
        //cout<<"Stealing Request not received yet!"<<endl;
    }

}



void  ReceiveTerminationToken (int myrank, int& tail_pointer, int& last_stolen_bigchunk,  struct token& terminating_token,int num_process)
{

    int left_process = myrank -1;
    if (myrank == 0) left_process = num_process-1;
    MPI_Status status;
    int         flag, index;
    void       *work;

    if (wrin_request_token == MPI_REQUEST_NULL)
    {
        MPI_Irecv(&terminating_token, 1, token_type, left_process, MPIWS_TDTOKEN, MPI_COMM_WORLD, &wrin_request_token);
        // printf("MPI process %d starts receiving termination token request\n", myrank);
    }

}

void ResponseTerminationToken (int myrank, int& tail_pointer, int& last_stolen_bigchunk, struct token& terminating_token,int num_process)
{

    MPI_Status status;
    int         flag, index;
    void       *work;

    MPI_Test(&wrin_request_token, &flag, &status);
    if (flag)
    {
        index = status.MPI_SOURCE;
        // cout<<"Termination token  received  in process: "<<myrank <<endl;
        int right_process = (myrank+1)%num_process;
        int left_process = myrank -1;
        if (myrank == 0) left_process = num_process-1;
        // MPI_Irecv(&terminating_token, 1, token_type, left_process, MPIWS_TDTOKEN, MPI_COMM_WORLD, &wrin_request_token);
        terminating_token.mycolor = black;
        // printf("MPI process %d asynchnously sends token color = %d\n\t \n", myrank, terminating_token.mycolor);
        MPI_Isend(&terminating_token, 1, token_type, right_process, MPIWS_TDTOKEN, MPI_COMM_WORLD, &wrout_request_token);
        // MPI_Send(&send_response, 1, response_type, 0, 123, MPI_COMM_WORLD);
    }
    // else
    // {
    //     cout<<"Termination token not received yet in process: "<<myrank <<endl;
    // }

}

#endif

struct Super
{
    int_t start;
    int_t end;
};

__host__ __device__ __forceinline__ void swap_ptr_index_vol(volatile int_t* &a, volatile int_t* &b){
    volatile int_t* temp = a;
    a = b;
    b = temp;

}


__host__ __device__ __forceinline__ void swap_ptr_index(int_t* &a, int_t* &b){
    int_t* temp = a;
    a = b;
    b = temp;

}
__host__ __device__ __forceinline__ void  swap_ptr_index_uint_8 (uint8_t* &a, uint8_t* &b){
    uint8_t* temp = a;
    a = b;
    b = temp;
}
__host__ __device__ __forceinline__ void  swap_ptr_index_uint_16 (uint16_t* &a, uint16_t* &b){
    uint16_t* temp = a;
    a = b;
    b = temp;
}
__device__ __forceinline__ int_t Minimum(int_t a, int_t b)
{
    if (a<b) return a;
    else return b;
}

__host__  __forceinline__ int_t Maximum3(int_t a,int_t b,int_t c)
{
    if (a >b) 
    {
        if (a > c) return a;
        else return c;
    }
    else
    {
        if (b > c) return b;
        else return c;
    }
    //    return (a < b)? b:a;
}

__host__ __device__ __forceinline__ int_t Maximum(uint_t a, uint_t b)
{
    return (a < b)? b:a;
}

__device__ __forceinline__ void  syncgroup(int_t group_id,int_t N_blocks_source)
{
    //sync all the "N_blocks_source" falling into group "group_id"
    return;
}


__device__ __forceinline__ void sync_X_block(int group_id, int N_blocks_source,int* d_lock, int N_groups) 
{
    volatile int *lock = d_lock;    

    // Threadfence and syncthreads ensure global writes 
    // thread-0 reports in with its sync counter
    __threadfence();
    __syncthreads();
    //                int group_bid= blockIdx.x & (N_blocks_source-1);//block id in the group
    int group_bid = blockIdx.x % N_blocks_source;
    int block_offset=group_id*N_blocks_source;
    if (group_bid== 0)//First block in the group
    {
        // Report in ourselves
        if (threadIdx.x == 0)
            lock[group_bid+block_offset] = 1;

        __syncthreads();

        // Wait for everyone else to report in
        //NOTE: change for more than 4 blocks
        int stop_block;
        if(group_id==N_groups-1)
        {
            stop_block=gridDim.x;
        }
        else
        {
            stop_block=block_offset+ N_blocks_source;
        }
        for (int peer_block = block_offset+threadIdx.x; 
                peer_block < stop_block; peer_block += blockDim.x)
            while (ThreadLoad(d_lock + peer_block) == 0)
                __threadfence_block();

        __syncthreads();

        // Let everyone know it's safe to proceed
        for (int peer_block = block_offset+threadIdx.x; 
                peer_block < stop_block; peer_block += blockDim.x)
            lock[peer_block] = 0;
    }
    else
    {
        if (threadIdx.x == 0)
        {
            // Report in
            // lock[blockIdx.x] = 1;
            lock[group_bid+block_offset] = 1;


            // Wait for acknowledgment
            //                         while (ThreadLoad (d_lock + blockIdx.x) != 0)
            while (ThreadLoad (d_lock + group_bid+block_offset) != 0)
                __threadfence_block();
            //  while (ThreadLoad (d_lock + group_bid+block_offset) == 1)
            //      __threadfence_block();
        }
        __syncthreads();
    }
}





__global__ void Initialise_cost_array(uint_t* cost_array_d,
        int_t vert_count,uint_t group_MAX_VAL, int_t N_src_group)
{
    int offset=blockDim.x*gridDim.x;
    int total_initialization=vert_count*N_src_group;
    for (int thid=blockDim.x*blockIdx.x+threadIdx.x; thid < total_initialization;thid+=offset)
    {
        cost_array_d[thid]=group_MAX_VAL;
    }
}


__global__ void Compute_fillins_joint_traversal_group_wise_supernode_OptIII (uint_t* cost_array_d,
        int_t* fill_in_d,int_t* frontier,
        int_t* next_frontier,int_t vert_count,
        int_t* csr, int_t* col_st,  int_t* col_ed,
        ull_t* fill_count,int_t gpu_id,int_t N_gpu,
        int_t* src_frontier_d,int_t* next_src_frontier_d,
        int_t* source_d,  int_t* frontier_size,  int_t* next_frontier_size,
        int_t* lock_d, int_t N_groups, int_t* dump,int_t* load,
        int_t N_src_group, int_t group,uint_t max_id_offset, 
        int_t* next_front_d, int_t passed_allocated_frontier_size,
        int_t* my_current_frontier_d,int_t* frontierchecked/*, int_t* offset_next_kernel*/,int_t* swap_GPU_buffers_m,
        int_t* nz_row_U_d, int_t max_supernode_size, int_t N_chunks,int_t* source_flag_d,int_t* my_supernode_d, Super* superObj,
        int_t* validity_supernode_d, int_t* pass_through_d,int_t* fill_count_per_row_d,ull_t* N_edge_checks_per_thread_d,
        int_t* new_col_ed_d, int_t N_GPU_Node, int_t local_gpu_id,ull_t* group80_count_d, ull_t* TEPS_value_perthread_d
        )
{

    // if (group==4) printf("Inside groupwise kernel\n");
    int_t level=0; //test for average frontier size
    int_t thid= blockDim.x*blockIdx.x+threadIdx.x;
    int_t original_thid=thid;    
    int_t dump_local=0;
    // #ifdef supernodes
    int_t warp_id = thid >> 5;
    int_t original_laneId= threadIdx.x & 0x1f;
    int_t N_warps = (blockDim.x * gridDim.x) >> 5;
    int_t N_chunks_group = N_src_group/max_supernode_size;

    int myrank = gpu_id/N_GPU_Node;
    // int_t N_chunks = (ceil)vert_count/(double)max_supernode_size;
    // #endif

#ifdef enable_debug
    if (original_thid==0)  printf("Inside groupwise kernel group: %d\n",group);
    if (original_thid==0) 
    {
        printf("N_src_group:%d\n",N_src_group);
        printf("N_chunks_group:%d\n",N_chunks_group);
        printf("max_supernode_size:%d\n",max_supernode_size);
    }
#endif



    while (thid < N_src_group)
    {
        // int node_offset = myrank * N_src_group * N_GPU_Node;
        // int node_offset = group*N_gpu*N_src_group;
        // source_d[thid] = node_offset + group * N_src_group + thid * N_GPU_Node + local_gpu_id;



        // source_d[thid] = node_offset + gpu_offset + thid;


        int_t intranode_offset = thid;
        pass_through_d[intranode_offset] = 0;  
        nz_row_U_d[intranode_offset]=0; // For supernodes
        thid+=(blockDim.x*gridDim.x);
    }


    sync_X_block(0,gridDim.x,lock_d,1);

    //Assign cost of the sources as min val
    for (thid=original_thid; thid < N_src_group; thid+=(blockDim.x*gridDim.x))
    {
        int_t cost_array_offset = thid*vert_count;//int_t cost_array_offset=source_id*vert_count;
        // int_t fill_array_offset = (thid*N_GPU_Node+local_gpu_id)*vert_count;
        // int_t fill_array_offset = (thid+local_gpu_id*N_src_group)*vert_count;// When allocating combined fill array for GPUs in a node
        int_t fill_array_offset = (thid)*vert_count;// When allocating combined fill array for GPUs in a node
        //cost_array_d[cost_array_offset+source_d[thid]]=MIN_VAL;
        if (source_d[thid] < vert_count)
        {
            cost_array_d[cost_array_offset+source_d[thid]] = max_id_offset; //max_id_offset=MIN_VAL for the current group

            //Done when detecting supernodes
            fill_in_d[fill_array_offset+ source_d[thid]] = source_d[thid];//Done when detecting supernodes
            // fill_in_d[cost_array_offset+ source_d[thid]]=source_d[thid];//Done when detecting supernodes
            //~Done when detecting supernodes
        }
    }


    for (int_t src_id=blockIdx.x ; src_id < N_src_group; src_id +=gridDim.x)
    {
        int_t source=source_d[src_id]; 

        if (source < vert_count)
        {
            for (int_t b_tid=threadIdx.x+col_st[source]; b_tid < col_ed[source]; b_tid+=blockDim.x)
            {
                int_t neighbor=csr[b_tid];
#ifdef profile_edge_checks
                N_edge_checks_per_thread_d[original_thid]++;
#endif
                int_t cost_array_offset=src_id*vert_count;//int_t cost_array_offset=source_id*vert_count;
                // int_t fill_array_offset = src_id*N_GPU_Node*vert_count;
                // int_t fill_array_offset = (src_id*N_GPU_Node+local_gpu_id)*vert_count;

                // int_t fill_array_offset = (src_id+local_gpu_id*N_src_group)*vert_count;// When allocating combined fill array for GPUs in a node
                int_t fill_array_offset = (src_id)*vert_count;// When allocating combined fill array for GPUs in a node


                cost_array_d[cost_array_offset+ neighbor]=max_id_offset;// //max_id_offset=MIN_VAL for the current group
                fill_in_d[fill_array_offset+ neighbor]=source;
                // fill_in_d[cost_array_offset+ neighbor]=source;
                if (neighbor < source) 
                {
                    // start_time_atomic=clock64();
                    int_t front_position=atomicAdd(frontier_size,1);
                    // time_atomic+=(clock64()-start_time_atomic);
                    frontier[front_position]=neighbor;
                    src_frontier_d[front_position]=src_id;//save the source position in the array not the source itself
                    // if (source==1020)  printf("NOTE:groupwise src:%d has neighbor:%d \n",source,neighbor);
                } 
#ifdef detect_supernodes
                else if (neighbor > source)
                    // else
                {
                    // atomicAdd(&nz_row_U_d[local_gpu_id*N_src_group +src_id],1);
                    atomicAdd(&nz_row_U_d[src_id],1);
                    // atomicAdd(&nz_row_U_d[src_id*N_GPU_Node+local_gpu_id],1);
                }          
#endif
            }
        }
    }

    sync_X_block(0,gridDim.x,lock_d,1);
    while(frontier_size[0]!=0)
    {
        my_current_frontier_d[original_thid]=INT_MAX;
        for(thid=original_thid;thid< frontier_size[0];/*thid+=(gridDim.x*blockDim.x)*/)
        {
            int_t front=frontier[thid];
            int_t source_id = src_frontier_d[thid];
            int_t cost_array_offset=source_id * vert_count;

            // int_t fill_array_offset = (source_id*N_GPU_Node+local_gpu_id)*vert_count;
            // int_t fill_array_offset = (source_id+local_gpu_id*N_src_group)*vert_count;// When allocating combined fill array for GPUs in a node
            int_t fill_array_offset = (source_id)*vert_count;// When allocating combined fill array for GPUs in a node
            int_t src=source_d[source_id];

            uint_t cost = Maximum(front+max_id_offset,cost_array_d[cost_array_offset+front]);//Cost1 for the neighbors of the representative that are less than the original front
            // int_t original_front = front;
            // int_t edge_boundary = vert_count;
            if (src < vert_count)
            {  
                //                #ifdef early_exit  
                int degree=  col_ed[front]- col_st[front];
                if (degree > (passed_allocated_frontier_size-ThreadLoad(next_frontier_size)))
                {
                    //exit taking consideration of worst case
                    my_current_frontier_d[original_thid]=thid;
#ifdef enable_debug
                    if (original_thid==0) printf("Earlier exit groupwise original_thid:%d  thid while dumping:%d  front:%d  src:%d\n",original_thid,thid,front,src);
#endif
                    dump[0]=1;
                    dump_local=1;
                    break;
                }  
                // #endif    
#ifdef enable_debug
#ifdef all_frontier_checked
                frontierchecked[thid]=1;
#endif  
#endif
                // int_t super_node = my_supernode_d[front];

                // int_t representative = superObj[super_node].start;
                //Traverse in the detected supernodes
                // if (false)
                int_t end_position;


                end_position = col_ed[front];

                for (int k=col_st[front]; k < end_position; k++)
                {
                    int_t m = csr[k];
                    #ifdef profile_TEPS
                    TEPS_value_perthread_d[original_thid]++;
                    #endif
#ifdef profile_edge_checks
                    N_edge_checks_per_thread_d[original_thid]++;
#endif
                    if (cost_array_d[cost_array_offset+m] > cost)
                    {
                        if (atomicMin(&cost_array_d[cost_array_offset+m],cost) > cost)
                        {

                            if (m < src)
                            {

                                int_t front_position=atomicAdd(next_frontier_size,1);

                                next_frontier[front_position]=m;
                                next_src_frontier_d[front_position]=src_frontier_d[thid];

                            }

                            if ((m + max_id_offset) > cost)
                            {

                                if (atomicMax(&fill_in_d[fill_array_offset+m],src) < src)
                                {
                                    atomicAdd(fill_count,1);
                                    // if ((myrank ==1) && (group ==80)) atomicAdd(group80_count_d,1);
#ifdef profile_fill_count_row
                                    atomicAdd(&fill_count_per_row_d[src],1);
#endif

#ifdef detect_supernodes
                                    if (m > src)
                                    {
                                        // atomicAdd(&nz_row_U_d[local_gpu_id*N_src_group +source_id],1);
                                        atomicAdd(&nz_row_U_d[source_id],1);
                                        // atomicAdd(&nz_row_U_d[source_id*N_GPU_Node+local_gpu_id],1);
                                    }
#endif
#ifdef enable_debug
                                    // printf("source:%d Detected_fill_in:%d\n",src,m);
#endif
                                }
                            }
                        }
                    }

                }

            }
            thid=atomicAdd(next_front_d,1);
        }

        sync_X_block (0,gridDim.x,lock_d,1);
        // #ifdef buffer_cpu_memory
        if (ThreadLoad(dump) ==1)
            // if (dump[0]==1)
        {
            if (dump_local==0) my_current_frontier_d[original_thid]=INT_MAX;
#ifdef enable_debug
            printf ("thread: %d is returned after dump with my_current_frontier_d:%d  frontier_size:%d\n",original_thid,my_current_frontier_d[original_thid],frontier_size[0]);
            if (original_thid==0)
            {
                printf("At kernel exit for dumping. Groupwise: next_front_d:%d  frontier_size:%d\n",next_front_d[0],frontier_size[0]);
                // next_frontier_size[0]=0;
            }
#endif
            return;
            // break;
        }
        // #endif
#ifdef enable_debug
#ifdef all_frontier_checked
        for (int temp_thid=original_thid;temp_thid <  frontier_size[0];temp_thid+=(blockDim.x*gridDim.x))
        {
            if (frontierchecked[temp_thid]!=1)
            {
                printf("groupwise: Error! All frontiers are not checked before swapping\n");
            }
        }
        sync_X_block(0,gridDim.x,lock_d,1);
        //Resetting
        for (int temp_thid=original_thid;temp_thid <  frontier_size[0];temp_thid+=(blockDim.x*gridDim.x))
        {
            frontierchecked[temp_thid]=0;

        }
#endif
#endif
        swap_ptr_index(frontier,next_frontier);
        swap_ptr_index(src_frontier_d,next_src_frontier_d);
        if (original_thid==0)
        {    
#ifdef enable_debug
            printf("frontier size in the last loop: %d\n",frontier_size[0]);
            printf("Number of fill-ins detected till now: %d\n",fill_count[0]);
#endif
            swap_GPU_buffers_m[0] ^=1;
            frontier_size[0]=next_frontier_size[0];
            next_frontier_size[0]=0;
            next_front_d[0] = blockDim.x*gridDim.x;

        }
        level++;
        sync_X_block(0,gridDim.x,lock_d,1); 
    }
    // if (source_d[0] == 4096) 
    // {
    //     int_t intranode_offset = local_gpu_id*N_src_group + 0;
    //     if (original_thid==0) printf("intranode_offset:%d source_index:%d  nz_row_U_d[%d]: %d\n", intranode_offset, 0, intranode_offset, nz_row_U_d[intranode_offset]); 
    // }
}

__global__ void Compute_fillins_joint_traversal_group_wise_supernode_OptIII_warp_centric (uint_t* cost_array_d,
        int_t* fill_in_d,int_t* frontier,
        int_t* next_frontier,int_t vert_count,
        int_t* csr, int_t* col_st,  int_t* col_ed,
        ull_t* fill_count,int_t gpu_id,int_t N_gpu,
        int_t* src_frontier_d,int_t* next_src_frontier_d,
        int_t* source_d,  int_t* frontier_size,  int_t* next_frontier_size,
        int_t* lock_d, int_t N_groups, int_t* dump,int_t* load,
        int_t N_src_group, int_t group,uint_t max_id_offset, 
        int_t* next_front_d, int_t passed_allocated_frontier_size,
        int_t* my_current_frontier_d,int_t* frontierchecked/*, int_t* offset_next_kernel*/,int_t* swap_GPU_buffers_m,
        int_t* nz_row_U_d, int_t max_supernode_size, int_t N_chunks,int_t* source_flag_d,int_t* my_supernode_d, Super* superObj,
        int_t* validity_supernode_d, int_t* pass_through_d,int_t* fill_count_per_row_d,ull_t* N_edge_checks_per_thread_d,
        int_t* new_col_ed_d, int_t N_GPU_Node, int_t local_gpu_id,ull_t* group80_count_d, ull_t* TEPS_value_perthread_d,
        int_t valid_index)
{

    // if (group==4) printf("Inside groupwise kernel\n");
    int_t level=0; //test for average frontier size
    int_t thid= blockDim.x*blockIdx.x+threadIdx.x;
    int_t original_thid=thid;    
    int_t dump_local=0;
    // #ifdef supernodes
    int_t warpId = thid >> 5;
    int_t laneID= threadIdx.x & 0x1f;
    int_t N_warps = (blockDim.x * gridDim.x) >> 5;
    int_t N_chunks_group = N_src_group/max_supernode_size;

    int myrank = gpu_id/N_GPU_Node;
    // int_t N_chunks = (ceil)vert_count/(double)max_supernode_size;
    // #endif

#ifdef enable_debug
    if (original_thid==0)  printf("Inside groupwise kernel group: %d\n",group);
    if (original_thid==0) 
    {
        printf("N_src_group:%d\n",N_src_group);
        printf("N_chunks_group:%d\n",N_chunks_group);
        printf("max_supernode_size:%d\n",max_supernode_size);
    }
#endif



    while (thid < N_src_group)
    {
        // int node_offset = myrank * N_src_group * N_GPU_Node;
        // int node_offset = group*N_gpu*N_src_group;
        // source_d[thid] = node_offset + group * N_src_group + thid * N_GPU_Node + local_gpu_id;



        // source_d[thid] = node_offset + gpu_offset + thid;


        int_t intranode_offset = thid;
        pass_through_d[intranode_offset] = 0;  
        nz_row_U_d[intranode_offset]=0; // For supernodes
        thid+=(blockDim.x*gridDim.x);
    }


    sync_X_block(0,gridDim.x,lock_d,1);

    //Assign cost of the sources as min val
    for (thid=original_thid; thid < N_src_group; thid+=(blockDim.x*gridDim.x))
    {
        int_t cost_array_offset = thid*vert_count;//int_t cost_array_offset=source_id*vert_count;
        // int_t fill_array_offset = (thid*N_GPU_Node+local_gpu_id)*vert_count;
        // int_t fill_array_offset = (thid+local_gpu_id*N_src_group)*vert_count;// When allocating combined fill array for GPUs in a node
        int_t fill_array_offset = (thid)*vert_count;// When allocating combined fill array for GPUs in a node
        //cost_array_d[cost_array_offset+source_d[thid]]=MIN_VAL;
        // if (source_d[thid] < vert_count)
        if (thid < valid_index)
        {
            cost_array_d[cost_array_offset+source_d[thid]] = max_id_offset; //max_id_offset=MIN_VAL for the current group

            //Done when detecting supernodes
            fill_in_d[fill_array_offset+ source_d[thid]] = source_d[thid];//Done when detecting supernodes
            // fill_in_d[cost_array_offset+ source_d[thid]]=source_d[thid];//Done when detecting supernodes
            //~Done when detecting supernodes
        }
    }


    for (int_t src_id=blockIdx.x ; src_id < N_src_group; src_id +=gridDim.x)
    {
        int_t source=source_d[src_id]; 

        // if (source < vert_count)
        if (src_id < valid_index)
        {
            for (int_t b_tid=threadIdx.x+col_st[source]; b_tid < col_ed[source]; b_tid+=blockDim.x)
            {
                int_t neighbor=csr[b_tid];
#ifdef profile_edge_checks
                N_edge_checks_per_thread_d[original_thid]++;
#endif
                int_t cost_array_offset=src_id*vert_count;//int_t cost_array_offset=source_id*vert_count;
                // int_t fill_array_offset = src_id*N_GPU_Node*vert_count;
                // int_t fill_array_offset = (src_id*N_GPU_Node+local_gpu_id)*vert_count;

                // int_t fill_array_offset = (src_id+local_gpu_id*N_src_group)*vert_count;// When allocating combined fill array for GPUs in a node
                int_t fill_array_offset = (src_id)*vert_count;// When allocating combined fill array for GPUs in a node


                cost_array_d[cost_array_offset+ neighbor]=max_id_offset;// //max_id_offset=MIN_VAL for the current group
                fill_in_d[fill_array_offset+ neighbor]=source;
                // fill_in_d[cost_array_offset+ neighbor]=source;
                if (neighbor < source) 
                {
                    // start_time_atomic=clock64();
                    int_t front_position=atomicAdd(frontier_size,1);
                    // time_atomic+=(clock64()-start_time_atomic);
                    frontier[front_position]=neighbor;
                    src_frontier_d[front_position]=src_id;//save the source position in the array not the source itself
                    // if (source==1020)  printf("NOTE:groupwise src:%d has neighbor:%d \n",source,neighbor);
                } 
#ifdef detect_supernodes
                else if (neighbor > source)
                    // else
                {
                    // atomicAdd(&nz_row_U_d[local_gpu_id*N_src_group +src_id],1);
                    atomicAdd(&nz_row_U_d[src_id],1);
                    // atomicAdd(&nz_row_U_d[src_id*N_GPU_Node+local_gpu_id],1);
                }          
#endif
            }
        }
    }

    sync_X_block(0,gridDim.x,lock_d,1);
    while(frontier_size[0]!=0)
    {
        my_current_frontier_d[original_thid]=INT_MAX;
        for(int_t thGroupId = warpId; thGroupId < frontier_size[0];)
        {
            int_t front=frontier[thGroupId];
            int_t source_id = src_frontier_d[thGroupId];
            int_t cost_array_offset=source_id * vert_count;

            // int_t fill_array_offset = (source_id*N_GPU_Node+local_gpu_id)*vert_count;
            // int_t fill_array_offset = (source_id+local_gpu_id*N_src_group)*vert_count;// When allocating combined fill array for GPUs in a node
            int_t fill_array_offset = (source_id)*vert_count;// When allocating combined fill array for GPUs in a node
            int_t src=source_d[source_id];

            uint_t cost = Maximum(front+max_id_offset,cost_array_d[cost_array_offset+front]);//Cost1 for the neighbors of the representative that are less than the original front
            // int_t original_front = front;
            // int_t edge_boundary = vert_count;


            if (src < vert_count)
            {  
                //                #ifdef early_exit  
                int degree=  col_ed[front]- col_st[front];


                // #endif    
#ifdef enable_debug
#ifdef all_frontier_checked
                frontierchecked[thGroupId]=1;
#endif  
#endif
                // int_t super_node = my_supernode_d[front];

                // int_t representative = superObj[super_node].start;
                //Traverse in the detected supernodes
                // if (false)
                int_t end_position;


                end_position = col_ed[front];

                for (int k=col_st[front] + laneID; k < end_position; k+=32)
                {
                    int_t m = csr[k];
                    #ifdef profile_TEPS
                    TEPS_value_perthread_d[original_thid]++;
                    #endif
#ifdef profile_edge_checks
                    N_edge_checks_per_thread_d[original_thid]++;
#endif
                    if (cost_array_d[cost_array_offset+m] > cost)
                    {
                        if (atomicMin(&cost_array_d[cost_array_offset+m],cost) > cost)
                        {

                            if (m < src)
                            {

                                int_t front_position=atomicAdd(next_frontier_size,1);

                                next_frontier[front_position]=m;
                                next_src_frontier_d[front_position]=source_id;

                            }

                            if ((m + max_id_offset) > cost)
                            {

                                if (atomicMax(&fill_in_d[fill_array_offset+m],src) < src)
                                {
                                    atomicAdd(fill_count,1);
                                    // if ((myrank ==1) && (group ==80)) atomicAdd(group80_count_d,1);
#ifdef profile_fill_count_row
                                    atomicAdd(&fill_count_per_row_d[src],1);
#endif

#ifdef detect_supernodes
                                    if (m > src)
                                    {
                                        // atomicAdd(&nz_row_U_d[local_gpu_id*N_src_group +source_id],1);
                                        atomicAdd(&nz_row_U_d[source_id],1);
                                        // atomicAdd(&nz_row_U_d[source_id*N_GPU_Node+local_gpu_id],1);
                                    }
#endif
#ifdef enable_debug
                                    // printf("source:%d Detected_fill_in:%d\n",src,m);
#endif
                                }
                            }
                        }
                    }

                }

            }
            if (laneID==0) thGroupId = atomicAdd(next_front_d,1);

            thGroupId= __shfl_sync(0xffffffff, thGroupId, 0);  
        }

        sync_X_block (0,gridDim.x,lock_d,1);

        swap_ptr_index(frontier,next_frontier);
        swap_ptr_index(src_frontier_d,next_src_frontier_d);
        if (original_thid==0)
        {    
#ifdef enable_debug
            printf("frontier size in the last loop: %d\n",frontier_size[0]);
            printf("Number of fill-ins detected till now: %d\n",fill_count[0]);
#endif
            swap_GPU_buffers_m[0] ^=1;
            frontier_size[0]=next_frontier_size[0];
            next_frontier_size[0]=0;
            next_front_d[0] = N_warps;

        }
        level++;
        sync_X_block(0,gridDim.x,lock_d,1); 
    }

}

int Compute_Src_group(int vert_count)
{
    cout<<"Start finding N_src_per_group"<<endl;
    double temp = 2147483648/(double)(6*vert_count);
    if (temp > vert_count)
    {
        temp=(int)log_2(vert_count/(double)2);
        temp=pow(2,temp);

    }
    else
    {
        temp=(int)log_2(temp);
        temp=pow(2,temp);
    }
    cout<<"Finished finding N_src_per_group"<<endl;
    return (int)temp;
}
void  SWAP_CPU_BUFFERS(int_t* &CPU_buffer_next_level,int_t* &CPU_buffer_next_level_source,int_t& size_CPU_buffer_next_level,
        int_t* &CPU_buffer_current_level,int_t*  &CPU_buffer_current_level_source,int_t& size_CPU_buffer_current_level,int_t* &current_N_buffer_CPU,int_t* &next_N_buffer_CPU,int_t* swap_CPU_buffers_m)
{
    // swap_ptr_index(CPU_buffer_next_level,CPU_buffer_current_level);
    int_t* temp=CPU_buffer_next_level;
    CPU_buffer_next_level=CPU_buffer_current_level;
    CPU_buffer_current_level=temp;

    // swap_ptr_index(CPU_buffer_next_level_source,CPU_buffer_current_level_source);
    temp=CPU_buffer_next_level_source;
    CPU_buffer_next_level_source=CPU_buffer_current_level_source;
    CPU_buffer_current_level_source=temp;

    temp=next_N_buffer_CPU;
    next_N_buffer_CPU=current_N_buffer_CPU;
    current_N_buffer_CPU=temp;

    size_CPU_buffer_current_level=size_CPU_buffer_next_level;
    size_CPU_buffer_next_level=0;
    swap_CPU_buffers_m[0]=0;

}

void SWAP_GPU_BUFFERS(int_t* &frontier_d,int_t* &next_frontier_d,int_t* &src_frontier_d,int_t* &next_src_frontier_d,int_t* &swap_GPU_buffers_m)
{
    int_t* temp=frontier_d;
    frontier_d=next_frontier_d;
    next_frontier_d=temp;

    temp=src_frontier_d;
    src_frontier_d=next_src_frontier_d;
    next_src_frontier_d=temp;
    swap_GPU_buffers_m[0]=0;

}

void Display_CPU_buffers(int_t size, int_t* CPU_buffer_frontier, int_t* CPU_buffer_source,int_t flag_load_dump)
{
    std::string activity;
    if (flag_load_dump==0)
    {
        activity=" dumping  ";
    }
    else
    {
        activity=" loading  ";
    }
    int_t lot=0;
    for (int i=0;i<size;i++)
    {
        if (i%allocated_frontier_size==0) cout<<endl<<"lot:"<<lot++<<activity;
        cout<<i<<":"<<CPU_buffer_frontier[i]<<"/"<<CPU_buffer_source[i]<<"  ";

    }
    cout<<endl;
}

void DetectSuperNodes(int_t* nz_row_U_h,int_t N_src_group,int_t& Nsup, int_t max_supernode_size,int_t N_gpu,
        int_t myrank, Super* &superObj, int_t vert_count,int_t group, int_t* N_non_zero_rows,int_t* my_supernode,int_t* my_representative,int_t* fill_in_h)
{
    //////////********Outer loop iterates through the small chunks of max_supernode_size to detect the supernodes*********///////////
    int_t N_small_chunks_per_GPU = N_src_group/max_supernode_size; //Always a power of 2
    // cout<<"N_small_chunks_per_GPU: "<<N_small_chunks_per_GPU<<endl;
    int_t source_index = 0;
    for (int i=0; i < N_small_chunks_per_GPU; i ++)
    {
        int_t row = (N_src_group * group) + ((i * N_gpu + myrank) * max_supernode_size); //beginning index of the small chunk
        for (int j=0; j < max_supernode_size; j++)
        {
            // row_index ++;
            if (row < vert_count)
            {
                if (j==0) //No relaxation is allowed when difference compared to 1
                {
                    //Begin of new supernode for each smaller chunk at the beginning of small chunk
                    Nsup++;
                    superObj[Nsup].start = row;
                    superObj[Nsup].end = row;
                }
                // else if  (abs(nz_row_U_h[source_index-1] - nz_row_U_h[source_index]) == 1)
                else if  ((nz_row_U_h[source_index-1] - nz_row_U_h[source_index]) == 1)
                {           

                    //Continuation of the supernode
                    //Check if the pattern are also matching for the two rows
                    int_t fill_array_offset_source = source_index * vert_count;
                    int_t fill_array_offset_last_source = (source_index-1) * vert_count;
                    int_t continue_flag=1;
                    for (int_t k= row; k < vert_count; k++)
                    {
                        //Check pattern only on the U part
                        if (fill_in_h[fill_array_offset_source + k] == row)
                        {
                            if (fill_in_h[fill_array_offset_last_source + k] != row-1)
                            {
                                //Pattern don't match
                                Nsup++;
                                superObj[Nsup].start = row;
                                superObj[Nsup].end = row;
                                continue_flag=0;
                                break;
                            } 

                        }
                    }


                    if (continue_flag == 1) superObj[Nsup].end = row;

                }
                else
                {
                    //Or Begin new supernode at some rows other than beginning of small chunk

                    Nsup++;
                    superObj[Nsup].start = row;
                    superObj[Nsup].end = row;
                }
                //For testing
                N_non_zero_rows[row] = nz_row_U_h[source_index];
                my_supernode[row] = Nsup;
                //~for testing
            }


            source_index++;
            row ++;

        }
    }
}


__device__ __forceinline__ int_t ShMem_offset (int_t warp_id, int_t max_supernode_size)
{
    // return ((warp_id*max_supernode_size) +1);
    return (warp_id*(max_supernode_size+1));
}

__global__ void  warpcentric_local_detection  (int_t* nz_row_U_d, int_t N_src_group, int_t max_supernode_size,
        int_t N_gpu, int_t gpu_id, Super* superObj_d,int_t vert_count, int_t group, int_t N_chunks, int_t* Nsup_per_chunk_d,
        int_t* pass_through_d,int_t* Nsup_d,int_t* fill_in_d,int_t* source_d, int_t* my_supernode_d)
{
    extern __shared__ int_t ShMem[];
    int_t thid= blockDim.x*blockIdx.x+threadIdx.x;
    int_t warp_id = thid >> 5;
    int_t warp_id_block = threadIdx.x >> 5;
    int_t original_laneId= threadIdx.x & 0x1f;
    int_t N_warps = (blockDim.x * gridDim.x) >> 5;
    int_t N_chunks_group = N_src_group/max_supernode_size;
    int_t chunk_id = gpu_id + warp_id * N_gpu + group * N_chunks_group; //Interleave the sources among the GPUs 
    int_t upper_chunk_bound =  (group + N_gpu)* N_chunks_group;
    if (upper_chunk_bound > N_chunks) upper_chunk_bound = N_chunks;

    int_t offset_warp = ShMem_offset(warp_id_block,max_supernode_size); // Need to change the definition. Only work for 1 block 
    if (original_laneId==0) ShMem[offset_warp]=0;
    __syncwarp();
    while (chunk_id < upper_chunk_bound)
    {
        // #ifdef supernodes
        //Assign continous source in a chunk of max_supernode_size. The chunks are interleaved among the processes.  
        //A thread warp is responsible for schduling smaller chunks to the GPU       
        for (int_t lane_id = original_laneId; lane_id < max_supernode_size; lane_id += (32))
        {
            // Assign threads in a warp to schdule the sources inside a chunk
            int_t src = lane_id + chunk_id * max_supernode_size;
            int_t pos = lane_id + warp_id * max_supernode_size;


            if (src < vert_count)
            {
                if ((lane_id==0))//remove abs later after confirming the logic//No relaxation is allowed when difference compared to 1
                {
                    //Begin of new supernode for each smaller chunk at the beginning of small chunk
                    //First position allocated for a warp is the #entries in the array whose each entry is the start position of a single thread to detect all the 
                    //items in the specific supernode.
                    int_t index_shared_mem = atomicAdd(&ShMem[offset_warp],1);//Atomically add the #supernodes detected into the shared memory
                    ShMem[offset_warp+index_shared_mem+1] = pos; //pos is the source_index of the source of the beginning of the supernode

                }
                else if ((nz_row_U_d[pos-1] - nz_row_U_d[pos]) != 1)
                {
                    int_t index_shared_mem = atomicAdd(&ShMem[offset_warp],1);//Atomically add the #supernodes detected into the shared memory
                    ShMem[offset_warp+index_shared_mem+1] = pos; //pos is the source_index of the source of the beginning of the supernode
                    // printf("index_shared_mem:%d  offset_warp:%d   pos:%d\n",index_shared_mem,offset_warp,pos);
                }
                else
                {
                    //Satisfies the first condition of non-zero count
                    //    pass_through_d[pos-1] = 1;
                    pass_through_d[pos] = 1;
                }
            }
        }
        __syncwarp();

        int_t Nsup_small_chunk = ShMem[offset_warp];


        for (int_t lane_id = original_laneId; lane_id < Nsup_small_chunk; lane_id += 32)
        {
            // 1 thread assign the start of supernode to all the rows in the supernode.

            //Each thread starts a new supernode at this instant
            int_t supernode_index = atomicAdd(Nsup_d,1);
            int_t local_supernode_count = 1;
            int_t start_index_supernode = ShMem[offset_warp+lane_id+1];
            int_t first_row_supernode = source_d[start_index_supernode];
            superObj_d[supernode_index].start = first_row_supernode;
            superObj_d[supernode_index].end = first_row_supernode;
            //    printf("first_row_supernode: %d\n",first_row_supernode);

            my_supernode_d[first_row_supernode] = supernode_index;
            int_t begin = start_index_supernode + 1;

            while (pass_through_d[begin] == 1)
            {
                //Check with the second filter
                //    int_t current_src_id = ShMem[begin];
                int_t current_row_id = begin;
                int_t current_row =   source_d[current_row_id];
                int_t offset_begin = current_row_id*vert_count;

                if (fill_in_d[offset_begin+ first_row_supernode] == current_row)
                {
                    //Inside the second filter that tests if Urj is non-zero for supernode starting at r and current rwo j
                    //Extend the supernode
                    superObj_d[supernode_index].end = current_row;

                }
                else
                {
                    //Introduce a new supernode
                    supernode_index = atomicAdd(Nsup_d,1);
                    superObj_d[supernode_index].start = current_row;
                    superObj_d[supernode_index].end = current_row;

                }
                my_supernode_d[current_row] = supernode_index;
                begin++;

            }

        }
        __syncwarp();
        if (original_laneId == 0) 
        {
            // ShMem[offset(warp_id)] = 0;
            ShMem[offset_warp]=0;
            Nsup_per_chunk_d[chunk_id] =  Nsup_small_chunk; //Writing the number for the prefix sum in the next kernel
        }
        __syncwarp();

        // source_d[thid]=gpu_id + thid * N_gpu + group * N_src_group; //Interleave the sources among the GPUs
        // #endif
        chunk_id += (N_warps*N_gpu);
        warp_id +=  N_warps;
    }
} 

__device__ __forceinline__ int_t binary_search(int_t first_row_supernode,int_t* array,int_t l,int_t r)
{
    if (first_row_supernode > array[r])
        return r;
    while (l <= r) 
    { 
        int_t m = l + (r - l) / 2; 
        int_t val =  array[m];
        if (first_row_supernode < val) 
        {
            r = m-1;
        }
        else if (first_row_supernode > val)
        {
            l = m + 1; 
        }
        else
        {
            return m;
        }
    } 
    return r;
}
__device__ __forceinline__ int_t Compute_r_id(int_t src,int_t N_src_group,int_t N_GPU_Node)
{

    int_t col_id = src / (N_src_group * N_GPU_Node);
    int_t base_source = src - col_id * N_src_group * N_GPU_Node;
    return ((base_source % N_GPU_Node)*N_src_group + base_source/N_GPU_Node);
    // return r_id;
}

__device__ __forceinline__ int_t Compute_r_id_local(int_t src,int_t N_src_group,int_t N_GPU_Node,int_t& g_id)
{

    int_t col_id = src / (N_src_group * N_GPU_Node);
    int_t base_source = src - col_id * N_src_group * N_GPU_Node;
    g_id=base_source % N_GPU_Node;
    return base_source/N_GPU_Node;
    // return ((base_source % N_GPU_Node)*N_src_group + base_source/N_GPU_Node);
    // return r_id;
}

struct aux_device
{
    int gpu_id; //gpu id

    int_t* fill_in_d;

    // int_t** fill_in_d_P;
    int_t* fill_in_d_P0;
    int_t* fill_in_d_P1;
    int_t* fill_in_d_P2;
    int_t* fill_in_d_P3;
    int_t* fill_in_d_P4;
    int_t* fill_in_d_P5;
    #ifdef profile_TEPS
    ull_t TEPS_value;
    ull_t* TEPS_value_perthread_d; 
    ull_t* TEPS_value_perthread;        
    #endif
    // fill_in_d_P
    int_t* nz_row_U_d;
    int_t* pass_through_d;

    int_t* col_ed_d;
    int_t* csr_d;
    int_t* col_st_d;
    int_t* count;

    uint_t max_id_offset;
    uint_t group_MAX_VAL;
    uint_t count_group_loop;

    int* next_source_d;
    int_t* frontier_size_d;
    int_t* temp_next_frontier_size_d;
    uint_t* cost_array_d;
    int_t* fill_in_last_row_h;
    int_t last_row_U_count;
    int* frontier_d;
    int* next_frontier_d;
    ull_t* fill_count_d;
    ull_t* group80_count_d;
    ull_t group80_count;
    ull_t fill_count;
    int_t* src_frontier_d;
    int_t* next_src_frontier_d;
    int* source_d;
    int_t* my_current_frontier_d;
    int* lock_d;
    int_t* next_front_d;
    int_t* CPU_buffer_next_level;//= (int*) malloc (N_src_group*vert_count*sizeof(int_t));
    int_t* CPU_buffer_next_level_source;//= (int*) malloc (N_src_group*vert_count*sizeof(int_t));
    int_t size_CPU_buffer_next_level;
    int_t* CPU_buffer_current_level;
    int_t* CPU_buffer_current_level_source;//= (int*) malloc (N_src_group*vert_count*sizeof(int_t));
    int_t size_CPU_buffer_current_level;
    int_t* next_N_buffer_CPU;
    int_t* current_N_buffer_CPU;
    int_t* N_non_zero_rows;
    int_t* my_supernode;
    int_t* my_representative;
    int_t* my_supernode_d;
    int_t* swap_CPU_buffers_m;
    int_t* swap_GPU_buffers_m;
    int_t* dump_m;
    int_t* load_m;
    int_t* current_buffer_m;
    int_t* next_buffer_m;
    int_t buffer_flag_host;
    int_t buffer_flag_getfronter_host;
    int_t* offset_next_kernel;
    int_t* offset_kernel;
    int_t N_dumping_cpu_memory;
    int_t N_reading_cpu_memory;
    int_t* next_frontier_size_d;

    // int_t* pass_through_d;
    int_t* source_flag;
    int_t* source_flag_d;
    Super* super_obj;
    int_t* nz_row_U_h;
    int_t* Nsup_d;
    Super* superObj ;
    Super* superObj_d;
    int_t* validity_supernode_d;
    int_t* validity_supernode_h; 
    int_t* Nsup_per_chunk_d;
    int_t* fill_count_per_row_d;
    // int_t* new_col_ed_d;
    int_t* fill_count_per_row;
    ull_t* N_edge_checks_per_thread_d;
    ull_t* N_edge_checks_per_thread;



};


// __global__ void  warpcentric_local_detection_test  (int_t* nz_row_U_d_g0,int_t* nz_row_U_d_g1,int_t* nz_row_U_d_g2,int_t* nz_row_U_d_g3, int_t* nz_row_U_d_g4, int_t* nz_row_U_d_g5, 
//         int_t N_src_group, int_t max_supernode_size,
//         int_t N_gpu, int_t gpu_id, Super* superObj_d,int_t vert_count, int_t group, int_t N_chunks, int_t* Nsup_per_chunk_d,
//         int_t* Nsup_d,
//         int_t* fill_in_d_g0, int_t* fill_in_d_g1, int_t* fill_in_d_g2, int_t* fill_in_d_g3,int_t* fill_in_d_g4,int_t* fill_in_d_g5,int_t* source_d, 
//         int_t* my_supernode_d,int_t* new_col_ed_d,int_t* col_ed_d,int_t* csr_d,int_t* col_st_d,int_t local_gpu_id,
//         int_t N_GPU_Node,int_t* count,int_t* my_pass_through_d)
// {
//     extern __shared__ int_t ShMem[];
//     int_t thid= blockDim.x*blockIdx.x+threadIdx.x;
//     int_t warp_id = thid >> 5;
//     int_t warp_id_block = threadIdx.x >> 5;
//     int_t original_laneId= threadIdx.x & 0x1f;
//     int_t N_warps = (blockDim.x * gridDim.x) >> 5;
//     int_t N_chunks_group = N_src_group/max_supernode_size;
//     int_t chunk_id = gpu_id + warp_id * N_gpu + group * N_chunks_group; //Interleave the sources among the GPUs 
//     int_t upper_chunk_bound =  (group + N_gpu) * N_chunks_group;
//     if (upper_chunk_bound > N_chunks) upper_chunk_bound = N_chunks;

//     int_t offset_warp = ShMem_offset(warp_id_block,max_supernode_size); // Max_supernode_size is allocated per warp in the shared memory. Need to change the definition. Only work for 1 block 
//     if (original_laneId==0) ShMem[offset_warp]=0;
//     __syncwarp();
//     while (chunk_id < upper_chunk_bound)
//     {
//         //1 warp is responsible to find the supernodes in a chunk
//         // #ifdef supernodes
//         //Assign continous source in a chunk of max_supernode_size. The chunks are interleaved among the processes.  
//         //A thread warp is responsible for schduling smaller chunks to the GPU       
//         for (int_t lane_id = original_laneId; lane_id < max_supernode_size; lane_id += (32))
//         {
//             // Assign threads in a warp to schdule the sources inside a chunk
//             int_t src = lane_id + chunk_id * max_supernode_size;// It is the exact source 

//             // int_t pos = lane_id + warp_id * max_supernode_size;//It is source index
//             int_t pos = src % N_src_group;//It is source index                

//             int_t g_id;
//             int_t r_id = Compute_r_id_local( src, N_src_group, N_GPU_Node,g_id);
//             int_t* nz_row_U_d;// = Get_memoryPointer_nz(g_id,nz_row_U_d_g0,);
//             // int_t* pass_through_d;
//             switch(g_id)
//             {
//                 case 0:
//                     nz_row_U_d= nz_row_U_d_g0;

//                     break;
//                 case 1:
//                     nz_row_U_d= nz_row_U_d_g1;

//                     break;
//                 case 2:
//                     nz_row_U_d= nz_row_U_d_g2;

//                     break;
//                 case 3:
//                     nz_row_U_d= nz_row_U_d_g3;

//                     break;
//                 case 4:
//                     nz_row_U_d= nz_row_U_d_g4;

//                     break;
//                 case 5:
//                     nz_row_U_d= nz_row_U_d_g5;

//                     break;
//                 default:
//                     nz_row_U_d= nz_row_U_d_g0; 
//                     // pass_through_d=pass_through_d_g0;       
//             }
//                  if (src < vert_count)
//             {
//                 if ((lane_id==0))//remove abs later after confirming the logic//No relaxation is allowed when difference compared to 1
//                 {
//                     //Begin of new supernode for each smaller chunk at the beginning of small chunk
//                     //First position allocated for a warp is the #entries in the array whose each entry is the start position of a single thread to detect all the 
//                     //items in the specific supernode.
//                     int_t index_shared_mem = atomicAdd(&ShMem[offset_warp],1);//Atomically add the #supernodes detected into the shared memory
//                     // ShMem[offset_warp+index_shared_mem+1] = pos; //pos is the source_index of the source of the beginning of the supernode
//                     ShMem[offset_warp+index_shared_mem+1] = src; //pos is the source_index of the source of the beginning of the supernode

//                 }
//                 // else if ((nz_row_U_d[pos-1] - nz_row_U_d[pos]) != 1)
//                 else 
//                 {
//                     int_t previous_src = src-1;
//                     int_t last_src_r_id= Compute_r_id_local( previous_src, N_src_group, N_GPU_Node,g_id);
//                     int_t* last_nz_row_U_d;
//                     switch(g_id)
//                     {
//                         case 0:
//                             last_nz_row_U_d= nz_row_U_d_g0;
//                             // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
//                             break;
//                         case 1:
//                             last_nz_row_U_d= nz_row_U_d_g1;
//                             // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
//                             break;
//                         case 2:
//                             last_nz_row_U_d= nz_row_U_d_g2;
//                             // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
//                             break;
//                         case 3:
//                             last_nz_row_U_d= nz_row_U_d_g3;
//                             // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
//                             break;
//                         case 4:
//                             last_nz_row_U_d= nz_row_U_d_g4;
//                             // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
//                             break;
//                         case 5:
//                             last_nz_row_U_d= nz_row_U_d_g5;
//                             // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
//                             break;
//                         default:
//                             last_nz_row_U_d= nz_row_U_d_g0;        
//                     }

//                     if ((last_nz_row_U_d[last_src_r_id] - nz_row_U_d[r_id]) != 1)
//                     {
//                         int_t index_shared_mem = atomicAdd(&ShMem[offset_warp],1);//Atomically add the #supernodes detected into the shared memory
//                         // ShMem[offset_warp+index_shared_mem+1] = pos; ////Put source instead of postion.....pos is the source_index of the source of the beginning of the supernode
//                         ShMem[offset_warp+index_shared_mem+1] = src; ////Put source instead of postion.....pos is the source_index of the source of the beginning of the supernode
//                         // printf("index_shared_mem:%d  offset_warp:%d   pos:%d\n",index_shared_mem,offset_warp,pos);
//                     }
//                     else
//                     {
//                         //Satisfies the first condition of non-zero count
//                         //    pass_through_d[pos-1] = 1;
//                         // pass_through_d[pos] = 1;
//                         // printf("Pass through detected!\n");
//                         // pass_through_d[r_id] = 1;
//                         my_pass_through_d[src%N_src_group] = 1;
//                         // printf("Passthrough count:%d\n",atomicAdd(count,1)+1);
//                     }
//                 }
//             }
//         }
//         __syncwarp();

//         int_t Nsup_small_chunk = ShMem[offset_warp];


//         for (int_t lane_id = original_laneId; lane_id < Nsup_small_chunk; lane_id += 32)
//         {
//             // 1 thread assign the start of supernode to all the rows in the supernode.

//             //Each thread starts a new supernode at this instant
//             int_t supernode_index = atomicAdd(Nsup_d,1);

//             int_t local_supernode_count = 1;

//             // int_t start_index_supernode = ShMem[offset_warp+lane_id+1];

//             // int_t first_row_supernode = source_d[start_index_supernode];
//             int_t first_row_supernode = ShMem[offset_warp+lane_id+1];

//             // int_t col_id = first_row_supernode / (N_src_group * N_GPU_Node);
//             // int_t base_source = first_row_supernode - col_id * N_src_group * N_GPU_Node;
//             // int_t start_index_supernode = (base_source % N_GPU_Node)*N_src_group + base_source/N_GPU_Node;

//             // int_t start_index_supernode = Compute_r_id( first_row_supernode, N_src_group, N_GPU_Node);
//             // int_t start_index_supernode = first_row_supernode%N_src_group;

//             superObj_d[supernode_index].start = first_row_supernode;
//             superObj_d[supernode_index].end = first_row_supernode;

//             //    printf("first_row_supernode: %d\n",first_row_supernode);

//             // my_supernode_d[first_row_supernode] = supernode_index;
//             int_t next_row= first_row_supernode+1;
//             // int_t begin = start_index_supernode + 1;
//             //  col_id = next_row / (N_src_group * N_GPU_Node);
//             //  base_source = next_row - col_id * N_src_group * N_GPU_Node;
//             //  int_t begin = (base_source % N_GPU_Node)*N_src_group + base_source/N_GPU_Node;
//             int_t g_id;
//             int_t next_row_id = Compute_r_id_local( next_row, N_src_group, N_GPU_Node,g_id);
//             int_t* fill_in_d;
//             // int_t* pass_through_d;
//             switch(g_id)
//             {
//                 case 0:
//                     fill_in_d= fill_in_d_g0;
//                     // pass_through_d= pass_through_d_g0;
//                     // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
//                     break;
//                 case 1:
//                     fill_in_d= fill_in_d_g1;
//                     // pass_through_d= pass_through_d_g1;
//                     // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
//                     break;
//                 case 2:
//                     fill_in_d= fill_in_d_g2;
//                     // pass_through_d= pass_through_d_g2;
//                     // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
//                     break;
//                 case 3:
//                     fill_in_d= fill_in_d_g3;
//                     // pass_through_d= pass_through_d_g3;
//                     // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
//                     break;
//                 case 4:
//                     fill_in_d= fill_in_d_g4;
//                     // pass_through_d= pass_through_d_g3;
//                     // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
//                     break;
//                 case 5:
//                     fill_in_d= fill_in_d_g5;
//                     // pass_through_d= pass_through_d_g3;
//                     // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
//                     break;
//                 default:
//                     fill_in_d= fill_in_d_g0;  
//                     // pass_through_d= pass_through_d_g0;    
//             }
//             // int_t next_row_id = Compute_r_id( next_row, N_src_group, N_GPU_Node);
//             // my_pass_through_d
//             int_t pass_through_local_index = next_row % N_src_group;
//             // while (pass_through_d[next_row_id] == 1)
//             while (my_pass_through_d[pass_through_local_index] == 1)
//             {
//                 // printf("Entering the second filter! \n");
//                 //Check with the second filter
//                 //    int_t current_src_id = ShMem[begin];

//                 int_t current_row_id = next_row_id; //The GPU has its own dedicated source_d so no mapping is done to current_row_id



//                 // int_t current_row =   source_d[current_row_id];
//                 int_t current_row =   next_row;

//                 int_t offset_current_row_id = current_row_id*vert_count;


//                 if (fill_in_d[offset_current_row_id + first_row_supernode] == current_row)
//                 {
//                     //Inside the second filter that tests if Urj is non-zero for supernode starting at r and current rwo j
//                     //Extend the supernode
//                     // printf("Supernode extended! \n");
//                     superObj_d[supernode_index].end = current_row;                    
//                     // col_ed_d[current_row] = 1 + col_st_d[current_row]+ binary_search(first_row_supernode,&csr_d[col_st_d[current_row]],0,col_ed_d[current_row]-col_st_d[current_row]-1);

//                 }
//                 else
//                 {
//                     //Introduce a new supernode
//                     // printf("Supernode Added! \n");
//                     supernode_index = atomicAdd(Nsup_d,1);
//                     // superObj_d[supernode_index].start = current_row;
//                     // superObj_d[supernode_index].end = current_row;


//                 }
//                 next_row++;
//                 // my_supernode_d[current_row] = supernode_index;
//                 if (next_row <vert_count)
//                 {
//                     next_row_id= Compute_r_id_local(next_row,N_src_group,N_GPU_Node,g_id);
//                     pass_through_local_index = next_row%N_src_group;

//                     switch(g_id)
//                     {
//                         case 0:
//                             fill_in_d= fill_in_d_g0;
//                             // pass_through_d= pass_through_d_g0;
//                             // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
//                             break;
//                         case 1:
//                             fill_in_d= fill_in_d_g1;
//                             // pass_through_d= pass_through_d_g1;
//                             // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
//                             break;
//                         case 2:
//                             fill_in_d= fill_in_d_g2;
//                             // pass_through_d= pass_through_d_g2;
//                             // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
//                             break;
//                         case 3:
//                             fill_in_d= fill_in_d_g3;
//                             // pass_through_d= pass_through_d_g3;
//                             // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
//                             break;
//                         case 4:
//                             fill_in_d= fill_in_d_g4;
//                             // pass_through_d= pass_through_d_g3;
//                             // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
//                             break;
//                         case 5:
//                             fill_in_d= fill_in_d_g5;
//                             // pass_through_d= pass_through_d_g3;
//                             // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
//                             break;
//                         default:
//                             fill_in_d= fill_in_d_g0;  
//                             // pass_through_d= pass_through_d_g0;    
//                     }
//                 } 
//                 else
//                 {
//                     break;
//                 }
//                 // if (group == 2)
//                 // {
//                 //     printf("Iniside passthroug while loop:currentrow:%d courrent_rid:%d Nextrow:%d next_rid:%d!\n",current_row, current_row_id,current_row+1,next_row_id);
//                 // }
//                 // begin++;

//             }

//         }
//         __syncwarp();
//         if (original_laneId == 0) 
//         {
//             // ShMem[offset(warp_id)] = 0;
//             ShMem[offset_warp]=0;
//             Nsup_per_chunk_d[chunk_id] =  Nsup_small_chunk; //Writing the number for the prefix sum in the next kernel
//         }
//         __syncwarp();

//         // source_d[thid]=gpu_id + thid * N_gpu + group * N_src_group; //Interleave the sources among the GPUs
//         // #endif
//         chunk_id += (N_warps*N_gpu);
//         warp_id +=  N_warps;
//     }
// } 


__device__ __forceinline__ int_t Compute_r_id_resecheduled(int local_chunk_id, int max_supernode_size, int j, int N_GPU_Node, int& owner_gpu, int& index_in_C)
{
    //Note: index_in_C is index of source list of concurrent sources the GPU is working on

    index_in_C = local_chunk_id * max_supernode_size + j;
    owner_gpu = index_in_C%N_GPU_Node; //GPU that has information for the source src
    // int pos = (ptr*(local_chunk_id+1))/N_GPU_Node;//position of the respective source in the owner GPU
    return (index_in_C/N_GPU_Node);//position of the respective source in the owner GPU

}



__global__ void  warpcentric_local_detection_test  (int_t* nz_row_U_d_g0,int_t* nz_row_U_d_g1,int_t* nz_row_U_d_g2,int_t* nz_row_U_d_g3, int_t* nz_row_U_d_g4, int_t* nz_row_U_d_g5, 
        int_t N_src_group, int_t max_supernode_size,
        int_t N_gpu, int_t global_gpu_id, Super* superObj_d,int_t vert_count, int_t group, int_t N_chunks, int_t* Nsup_per_chunk_d,
        int_t* Nsup_d,
        int_t* fill_in_d_g0, int_t* fill_in_d_g1, int_t* fill_in_d_g2, int_t* fill_in_d_g3,int_t* fill_in_d_g4,int_t* fill_in_d_g5,int_t* source_d, 
        int_t* my_supernode_d,int_t* new_col_ed_d,int_t* col_ed_d,int_t* csr_d,int_t* col_st_d,int_t local_gpu_id,
        int_t N_GPU_Node,int_t* count,int_t* my_pass_through_d,int_t* my_chunks, int num_curr_chunks_per_node)
{
    extern __shared__ int_t ShMem[];
    int_t thid= blockDim.x*blockIdx.x+threadIdx.x;
    int_t original_thid = thid;
    int_t warp_id = thid >> 5;
    int_t warp_id_block = threadIdx.x >> 5;
    int_t original_laneId= threadIdx.x & 0x1f;
    int_t N_warps = (blockDim.x * gridDim.x) >> 5;
    int_t N_chunks_group = N_src_group/max_supernode_size;

    int_t chunk_id;//global chunk_id

    // int_t offset_warp = ShMem_offset(warp_id_block,max_supernode_size); // Max_supernode_size is allocated per warp in the shared memory. Need to change the definition. Only work for 1 block 

    int_t local_chunk_id_to_GPU = 0;
    for (int i=local_gpu_id; i<num_curr_chunks_per_node; i+=N_GPU_Node)
        // for (int i=local_gpu_id; i<num_curr_chunks_per_node; i+=N_GPU_Node)
    {
        if (original_thid==0) ShMem[0]=0;
        // __syncwarp();
        __syncthreads();// Only 1 thread block is used

        chunk_id = my_chunks[i];
        
        // if (original_thid ==0) printf("GPU: %d works on chunk: %d\n",local_gpu_id,chunk_id);
        int local_chunk_id = i;
        //Each GPU in a node processes 1 Chunk of size (max_supenode_size) to process 
        int starting_source_index_C = local_chunk_id_to_GPU * max_supernode_size;

        //Detect supernodes in chunk_id

        if (chunk_id < N_chunks)
        {
            //The chunk is valid. Detect supernode in the chunk
            //Let 1 GPU detect supernode in 1 chunk unlike warp centric.
            int starting_source_chunk = chunk_id * max_supernode_size;
            for (int source_offset = original_thid; source_offset< max_supernode_size; source_offset+=(blockDim.x*gridDim.x))
            {
                int src = chunk_id * max_supernode_size + source_offset;//Each thread grabs a source or row in the chunk of max_supernode_size
                int j = src - starting_source_chunk;
                int_t owner_gpu; 
                int_t index_in_C; //Note: index_in_C is index of source list of concurrent sources the GPU is working on
                int r_id = Compute_r_id_resecheduled(local_chunk_id, max_supernode_size, j, N_GPU_Node, owner_gpu, index_in_C);
                // if ((chunk_id == 4) || (chunk_id == 0))                   
                // {
                //     printf("src: %d, owner_gpu: %d, row_index: %d\n",src, owner_gpu, r_id);
                // }
                int_t* nz_row_U_d;// = Get_memoryPointer_nz(g_id,nz_row_U_d_g0,);

                //First condition checking for T3 detection
                switch(owner_gpu)
                {
                    case 0:
                        nz_row_U_d= nz_row_U_d_g0;

                        break;
                    case 1:
                        nz_row_U_d= nz_row_U_d_g1;

                        break;
                    case 2:
                        nz_row_U_d= nz_row_U_d_g2;

                        break;
                    case 3:
                        nz_row_U_d= nz_row_U_d_g3;

                        break;
                    case 4:
                        nz_row_U_d= nz_row_U_d_g4;

                        break;
                    case 5:
                        nz_row_U_d= nz_row_U_d_g5;

                        break;
                    default:
                        nz_row_U_d= nz_row_U_d_g0; 
                        // pass_through_d=pass_through_d_g0;       
                }
                if (src < vert_count)
                {
                    if ((original_thid==0))//remove abs later after confirming the logic//No relaxation is allowed when difference compared to 1
                    {
                        //Begin of new supernode for each smaller chunk at the beginning of small chunk
                        //First position allocated for a warp is the #entries in the array whose each entry is the start position of a single thread to detect all the 
                        //items in the specific supernode.
                        int_t index_shared_mem = atomicAdd(&ShMem[0],1);//Atomically add the #supernodes detected into the shared memory
                        // ShMem[offset_warp+index_shared_mem+1] = pos; //pos is the source_index of the source of the beginning of the supernode
                        ShMem[index_shared_mem+1] = src; //pos is the source_index of the source of the beginning of the supernode

                    }
                    // else if ((nz_row_U_d[pos-1] - nz_row_U_d[pos]) != 1)
                    else 
                    {
                        int_t previous_src = src-1;
                        // int_t last_src_r_id= Compute_r_id_local( previous_src, N_src_group, N_GPU_Node,g_id);
                        j = previous_src - starting_source_chunk;
                        int_t previous_src_index_in_C;
                        int_t last_src_r_id = Compute_r_id_resecheduled(local_chunk_id, max_supernode_size, j, N_GPU_Node, owner_gpu, previous_src_index_in_C);
                        int_t* last_nz_row_U_d;
                        switch(owner_gpu)
                        {
                            case 0:
                                last_nz_row_U_d= nz_row_U_d_g0;
                                // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                                break;
                            case 1:
                                last_nz_row_U_d= nz_row_U_d_g1;
                                // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                                break;
                            case 2:
                                last_nz_row_U_d= nz_row_U_d_g2;
                                // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                                break;
                            case 3:
                                last_nz_row_U_d= nz_row_U_d_g3;
                                // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                                break;
                            case 4:
                                last_nz_row_U_d= nz_row_U_d_g4;
                                // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                                break;
                            case 5:
                                last_nz_row_U_d= nz_row_U_d_g5;
                                // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                                break;
                            default:
                                last_nz_row_U_d= nz_row_U_d_g0;        
                        }

                        if ((last_nz_row_U_d[last_src_r_id] - nz_row_U_d[r_id]) != 1)
                        {
                            int_t index_shared_mem = atomicAdd(&ShMem[0],1);//Atomically add the #supernodes detected into the shared memory
                            // ShMem[offset_warp+index_shared_mem+1] = pos; ////Put source instead of postion.....pos is the source_index of the source of the beginning of the supernode
                            ShMem[index_shared_mem+1] = src; ////Put source instead of postion.....pos is the source_index of the source of the beginning of the supernode
                            // printf("index_shared_mem:%d  offset_warp:%d   pos:%d\n",index_shared_mem,offset_warp,pos);
                        }
                        else
                        {
                            //Satisfies the first condition of non-zero count
                            // printf("Pass through detected!\n");
                            // my_pass_through_d[src%N_src_group] = 1;
                            // my_pass_through_d[index_in_C%N_src_group] = 1;
                            int diff = src-starting_source_chunk;
                            int pos_in_C = starting_source_index_C + diff;
                            my_pass_through_d[pos_in_C%N_src_group] = 1;
                            // my_pass_through_d[r_id%N_src_group] = 1;
                            // printf("Passthrough count:%d\n",atomicAdd(count,1)+1);
                        }
                    }
                }

            }

            // for (int source = thid + chunk_id*max_supernode_size; thid <)
        }
        //Synchronize all working threads
        // sync_X_block(0,gridDim.x,lock_d,1);
        __syncthreads();// Only 1 thread block is used
        //Entering the second phase (second condition) for supernode detection
        int_t Nsup_small_chunk = ShMem[0];
        if (chunk_id < N_chunks)
        {
            for (int thid = original_thid; thid < Nsup_small_chunk;  thid+=(blockDim.x*gridDim.x))
            {
                // 1 thread assign the start of supernode to all the rows in the supernode.

                //Each thread starts a new supernode at this instant
                int_t supernode_index = atomicAdd(Nsup_d,1);

                int_t local_supernode_count = 1;
                int_t first_row_supernode = ShMem[thid+1];
                superObj_d[supernode_index].start = first_row_supernode;
                superObj_d[supernode_index].end = first_row_supernode;
                int_t next_row= first_row_supernode+1;
                int starting_source_chunk = chunk_id * max_supernode_size;
                int j = next_row - starting_source_chunk;


                int_t owner_gpu;
                int_t next_row_index_in_C;
                int_t next_row_id = Compute_r_id_resecheduled(local_chunk_id, max_supernode_size, j, N_GPU_Node, owner_gpu, next_row_index_in_C);

                int_t* fill_in_d;
                switch(owner_gpu)
                {
                    case 0:
                        fill_in_d= fill_in_d_g0;
                        // pass_through_d= pass_through_d_g0;
                        // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                        break;
                    case 1:
                        fill_in_d= fill_in_d_g1;
                        // pass_through_d= pass_through_d_g1;
                        // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                        break;
                    case 2:
                        fill_in_d= fill_in_d_g2;
                        // pass_through_d= pass_through_d_g2;
                        // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                        break;
                    case 3:
                        fill_in_d= fill_in_d_g3;
                        // pass_through_d= pass_through_d_g3;
                        // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                        break;
                    case 4:
                        fill_in_d= fill_in_d_g4;
                        // pass_through_d= pass_through_d_g3;
                        // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                        break;
                    case 5:
                        fill_in_d= fill_in_d_g5;
                        // pass_through_d= pass_through_d_g3;
                        // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                        break;
                    default:
                        fill_in_d= fill_in_d_g0;  
                        // pass_through_d= pass_through_d_g0;    
                }
                // int_t pass_through_local_index = next_row % N_src_group;
                

                int diff = next_row-starting_source_chunk;
                int pos_in_C = starting_source_index_C + diff;
                int_t pass_through_local_index = pos_in_C % N_src_group;
                // int_t pass_through_local_index = next_row_index_in_C % N_src_group;
         

                // int_t pass_through_local_index = next_row_id % N_src_group;
                while (my_pass_through_d[pass_through_local_index] == 1)
                {
                    int_t current_row_id = next_row_id; //The GPU has its own dedicated source_d so no mapping is done to current_row_id
                    int_t current_row =   next_row;

                    int_t offset_current_row_id = current_row_id*vert_count;
                    if (fill_in_d[offset_current_row_id + first_row_supernode] == current_row)
                    {
                        //Inside the second filter that tests if Urj is non-zero for supernode starting at r and current rwo j
                        //Extend the supernode
                        // printf("Supernode extended! \n");
                        superObj_d[supernode_index].end = current_row;                    
                        // col_ed_d[current_row] = 1 + col_st_d[current_row]+ binary_search(first_row_supernode,&csr_d[col_st_d[current_row]],0,col_ed_d[current_row]-col_st_d[current_row]-1);

                    }
                    else
                    {
                        //Introduce a new supernode
                        // printf("Supernode Added! \n");
                        supernode_index = atomicAdd(Nsup_d,1);
                        // superObj_d[supernode_index].start = current_row;
                        // superObj_d[supernode_index].end = current_row;


                    }
                    next_row++;
                    if (next_row <vert_count)
                    {
                        j = next_row - starting_source_chunk;

                        next_row_id = Compute_r_id_resecheduled(local_chunk_id, max_supernode_size, j, N_GPU_Node, owner_gpu, next_row_index_in_C);

                        // next_row_id= Compute_r_id_local(next_row,N_src_group,N_GPU_Node,g_id);
                        // pass_through_local_index = next_row%N_src_group;
                        diff = next_row-starting_source_chunk;
                        pos_in_C = starting_source_index_C + diff;

                        // pass_through_local_index = next_row_index_in_C%N_src_group;
                        pass_through_local_index = pos_in_C%N_src_group;
                        // pass_through_local_index = next_row_id%N_src_group;

                        switch(owner_gpu)
                        {
                            case 0:
                                fill_in_d= fill_in_d_g0;
                                // pass_through_d= pass_through_d_g0;
                                // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                                break;
                            case 1:
                                fill_in_d= fill_in_d_g1;
                                // pass_through_d= pass_through_d_g1;
                                // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                                break;
                            case 2:
                                fill_in_d= fill_in_d_g2;
                                // pass_through_d= pass_through_d_g2;
                                // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                                break;
                            case 3:
                                fill_in_d= fill_in_d_g3;
                                // pass_through_d= pass_through_d_g3;
                                // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                                break;
                            case 4:
                                fill_in_d= fill_in_d_g4;
                                // pass_through_d= pass_through_d_g3;
                                // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                                break;
                            case 5:
                                fill_in_d= fill_in_d_g5;
                                // pass_through_d= pass_through_d_g3;
                                // printf("GPU:%d accessing GPU:%d\n",gpu_id,dest_gpi_Id);
                                break;
                            default:
                                fill_in_d= fill_in_d_g0;  
                                // pass_through_d= pass_through_d_g0;    
                        }
                    } 
                    else
                    {
                        break;
                    }
                }
            }
        }
        __syncthreads();
        if (original_thid ==0)
        {
            ShMem[0]=0;
            Nsup_per_chunk_d[chunk_id] =  Nsup_small_chunk; //Writing the number for the prefix sum in the next kernel
        }
        __syncthreads();
        local_chunk_id_to_GPU++;
    }

}






void DetectSuperNodes_parallel(int_t* nz_row_U_d, int_t N_src_group,int_t max_supernode_size, int_t N_gpu, int_t gpu_id, 
        Super* &superObj_d,int_t vert_count, int_t group,int_t N_chunks,int_t* Nsup_per_chunk_d,int_t* pass_through_d,
        int_t* Nsup_d, int_t* fill_in_d,int_t* source_d,int_t* &my_supernode_d)
{
    //***************Warp centric Supernode detection************//////////////
    //***************1 warp assignined for 1 small chunk to detect the local supernodes************//////////////
    int_t N_warp = (128*128) >> 5; 
    int_t N_warp_per_block = 128 >> 5;

    int_t N_entries_ShMem = N_warp_per_block * (max_supernode_size+1); // +1 is for maintaining the count of supernodes per warp
    // cout<<"N_entries_ShMem: "<< N_entries_ShMem <<endl;
    // cout<<"Size of shared memory per block: "<< (N_entries_ShMem * sizeof(int_t) / (float)1024)<<endl;
    // cout<<"Before warpcentric_local_detection kernel"<<endl;
    warpcentric_local_detection<<<128,128, N_entries_ShMem * sizeof(int_t)>>> (nz_row_U_d,N_src_group,max_supernode_size, N_gpu,
            gpu_id, superObj_d,vert_count,group,N_chunks,Nsup_per_chunk_d,pass_through_d,Nsup_d,fill_in_d,source_d,my_supernode_d);
    H_ERR(cudaDeviceSynchronize());
    int_t* updated_supernode = new int_t [vert_count];
    H_ERR(cudaMemcpy(updated_supernode,my_supernode_d,sizeof(int_t)*vert_count,cudaMemcpyDeviceToHost));
    int_t Nsup;




    // cout<<"After warpcentric_local_detection kernel"<<endl;

    //***************Prefix sum to compute the offsets************//////////////
    //    prefix_sum <<<128,128>>> (Nsup_warp, offset_per_warp);
    //    H_ERR(cudaDeviceSynchronize());

    //***************Reassigning proper index (continous at process/GPU level) to the supernodes************//////////////
    // warpcentric_reassign_index <<<128,128>>> (offset_per_warp, superObj_d);
    // H_ERR(cudaDeviceSynchronize());

}
aux_device *dev_mem;
// void DetectSuperNodes_parallel_test(int_t* nz_row_U_d_g0,int_t* nz_row_U_d_g1,int_t* nz_row_U_d_g2,int_t* nz_row_U_d_g3, int_t N_src_group,int_t max_supernode_size, int_t N_gpu, int_t gpu_id, 
//         Super* &superObj_d,int_t vert_count, int_t group,int_t N_chunks,int_t* Nsup_per_chunk_d,int_t* pass_through_d_g0,int_t* pass_through_d_g1,int_t* pass_through_d_g2,int_t* pass_through_d_g3,
//         int_t* Nsup_d, int_t* fill_in_d_g0,int_t* fill_in_d_g1,int_t* fill_in_d_g2,int_t* fill_in_d_g3,int_t* source_d,int_t* &my_supernode_d,int_t* new_col_ed_d,int_t* col_ed_d,int_t* csr_d, int_t* col_st_d, 
//         cudaStream_t streamObj,int_t local_gpu_id,int_t N_GPU_Node,int_t* count,int_t* my_pass_through_d)
void DetectSuperNodes_parallel_test( aux_device *dev_mem, int_t N_src_group,int_t max_supernode_size, int_t N_gpu, int_t gpu_id, 
        Super* &superObj_d,int_t vert_count, int_t group,int_t N_chunks,int_t* Nsup_per_chunk_d,
        int_t* Nsup_d,int_t* source_d,int_t* &my_supernode_d,int_t* new_col_ed_d,int_t* col_ed_d,int_t* csr_d, int_t* col_st_d, 
        cudaStream_t streamObj,int_t local_gpu_id,int_t N_GPU_Node,int_t* count,int_t* my_pass_through_d,int_t* my_chunks, int num_curr_chunks_per_node)
{
    //***************Warp centric Supernode detection************//////////////
    //***************1 warp assignined for 1 small chunk to detect the local supernodes************//////////////
    int_t N_warp = (1*128) >> 5; 
    int_t N_warp_per_block = 128 >> 5;
    // if (group > 2)
    // {
    //     printf("Detect supernide parallel!\n");
    // }
    // int_t N_entries_ShMem = N_warp_per_block * (max_supernode_size+1); // +1 is for maintaining the count of supernodes per warp
    // int_t N_entries_ShMem = (N_src_group+1); // +1 is for maintaining the count of supernodes per warp
    int_t N_entries_ShMem = (max_supernode_size+1); // +1 is for maintaining the count of supernodes per warp

// cout<<"max_supernode_size: "<<max_supernode_size<<" num_curr_chunks_per_node: "<<num_curr_chunks_per_node<<endl;
// cout<<"my_chunks: {"<<endl;
// for (int i=0;i <num_curr_chunks_per_node; i++)
// {
//     cout<< my_chunks[i]<<", ";
// }
// cout<<" }"<<endl;
    // warpcentric_local_detection_test<<<128,128, N_entries_ShMem * sizeof(int_t),streamObj>>> (nz_row_U_d_g0,nz_row_U_d_g1,nz_row_U_d_g2,nz_row_U_d_g3,N_src_group,max_supernode_size, N_gpu,
    //         gpu_id, superObj_d,vert_count,group,N_chunks,Nsup_per_chunk_d,
    //         pass_through_d_g0, pass_through_d_g1, pass_through_d_g2, pass_through_d_g3, Nsup_d,
    //         fill_in_d_g0, fill_in_d_g1, fill_in_d_g2, fill_in_d_g3, source_d,my_supernode_d,new_col_ed_d,
    //         col_ed_d,csr_d,col_st_d,local_gpu_id, N_GPU_Node,count,my_pass_through_d);
    warpcentric_local_detection_test<<<1,128, N_entries_ShMem * sizeof(int_t),streamObj>>> (dev_mem[0].nz_row_U_d,dev_mem[1].nz_row_U_d,dev_mem[2].nz_row_U_d,dev_mem[3].nz_row_U_d,dev_mem[4].nz_row_U_d,dev_mem[5].nz_row_U_d,
            N_src_group,max_supernode_size, N_gpu,
            gpu_id, superObj_d,vert_count,group,N_chunks,Nsup_per_chunk_d,
            Nsup_d,
            dev_mem[0].fill_in_d, dev_mem[1].fill_in_d, dev_mem[2].fill_in_d, dev_mem[3].fill_in_d, dev_mem[4].fill_in_d,dev_mem[5].fill_in_d,source_d,my_supernode_d,new_col_ed_d,
            col_ed_d,csr_d,col_st_d,local_gpu_id, N_GPU_Node,count,my_pass_through_d,my_chunks,  num_curr_chunks_per_node);

}

int compare (const void * a, const void * b)
{
    if ( *(int*)a <  *(int*)b ) return -1;
    if ( *(int*)a == *(int*)b ) return 0;
    if ( *(int*)a >  *(int*)b ) return 1;
}

void SortEdgeList (int_t* &csr,int_t* &col_beg, int_t* &col_ed,int_t vert_count, int_t edge_count)
{
    int_t* new_csr= (int*) malloc (edge_count*sizeof(int_t));
    int_t* new_col_beg= (int*) malloc (vert_count*sizeof(int_t));
    int_t* new_col_ed= (int*) malloc (vert_count*sizeof(int_t));

    for (int_t i =0;i < vert_count; i++)
    {
        // std::vector<int> temp_vector;
        int_t begin = col_beg[i];
        int_t end = col_ed[i];
        qsort (&csr[begin], end-begin, sizeof(int), compare);
    }
}

void Transpose_Matrix(int_t* &csr,int_t* &col_beg, int_t* &col_ed,int_t vert_count, int_t edge_count)
{
    int_t* new_csr= (int*) malloc (edge_count*sizeof(int_t));
    int_t* new_col_beg= (int*) malloc (vert_count*sizeof(int_t));
    int_t* new_col_ed= (int*) malloc (vert_count*sizeof(int_t));

    int_t* edge_count_per_vertex = (int*) malloc (vert_count*sizeof(int_t));
    memset(edge_count_per_vertex, 0, vert_count*sizeof(int_t));
    vec* vectors_graph = (vec*) malloc (vert_count*sizeof(vec));
    // std::<std::vector<int> > vectors_graph;

    for (int_t i =0;i < vert_count; i++)
    {
        // std::vector<int> temp_vector;
        int_t begin = col_beg[i];
        int_t end = col_ed[i];
        while (begin < end)
        {
            int_t neighbor= csr[begin];
            vectors_graph[neighbor].push_back(i);
            begin++;
        }
    }
    //Writing the transposed matrix in the arrays
    int_t csr_count=0;
    for (int_t i =0; i< vert_count; i++)
    {
        if (i==0) 
        {
            new_col_beg[i] = 0;
        }
        else
        {
            new_col_beg[i] = new_col_ed[i-1];
        }
        new_col_ed[i] = new_col_beg[i] + vectors_graph[i].size();
        int_t begin = new_col_beg[i];
        int_t end = new_col_ed[i];
        int_t vector_begin=0;
        while (begin < end)
        {
            new_csr[begin] =  vectors_graph[i][vector_begin];           
            begin++;
            vector_begin++;
        }
    }

    swap_ptr_index(new_csr,csr);
    swap_ptr_index(new_col_beg,col_beg);
    swap_ptr_index(new_col_ed,col_ed);
}


void SynchronizeAllDevices(cudaStream_t* stream,int_t N_gpu)
{
    for (int i = 0; i < N_gpu; i++)
    {
        //ensures every GPU finishes their work
        H_ERR(cudaSetDevice(i));
        H_ERR(cudaStreamSynchronize(stream[i]));
        //	H_ERR(cudaDeviceSynchronize());
    }
}


// Allocate_Initialize (dev_mem[i], vert_count, edge_count, csr, col_st, col_ed, BLKS_NUM, blockSize,i,N_src_group,real_allocation, N_chunks);

void Allocate_Initialize (struct aux_device& device_obj,int_t vert_count,int_t edge_count,int_t* csr,int_t* col_st,int_t* col_ed,
        int_t BLKS_NUM,int_t blockSize,int index, int_t next_front,int_t N_src_group,int_t real_allocation,int_t N_chunks)
{
    device_obj.gpu_id=index;

#ifdef workload_stealing
    cout<<"Allocating space for fill_in initialization debug"<<endl;
    H_ERR(cudaMallocManaged((void**) &device_obj.fill_in_d_P0,sizeof(int_t)*N_src_group*vert_count)); 
    H_ERR(cudaMemset(device_obj.fill_in_d_P0, 0, sizeof(int_t)*N_src_group*vert_count));

    H_ERR(cudaMallocManaged((void**) &device_obj.fill_in_d_P1,sizeof(int_t)*N_src_group*vert_count)); 
    H_ERR(cudaMemset(device_obj.fill_in_d_P1, 0, sizeof(int_t)*N_src_group*vert_count));

    H_ERR(cudaMallocManaged((void**) &device_obj.fill_in_d_P2,sizeof(int_t)*N_src_group*vert_count)); 
    H_ERR(cudaMemset(device_obj.fill_in_d_P2, 0, sizeof(int_t)*N_src_group*vert_count));

    H_ERR(cudaMallocManaged((void**) &device_obj.fill_in_d_P3,sizeof(int_t)*N_src_group*vert_count)); 
    H_ERR(cudaMemset(device_obj.fill_in_d_P3, 0, sizeof(int_t)*N_src_group*vert_count));

    H_ERR(cudaMallocManaged((void**) &device_obj.fill_in_d_P4,sizeof(int_t)*N_src_group*vert_count)); 
    H_ERR(cudaMemset(device_obj.fill_in_d_P4, 0, sizeof(int_t)*N_src_group*vert_count));

    H_ERR(cudaMallocManaged((void**) &device_obj.fill_in_d_P5,sizeof(int_t)*N_src_group*vert_count)); 
    H_ERR(cudaMemset(device_obj.fill_in_d_P5, 0, sizeof(int_t)*N_src_group*vert_count));
#endif

#ifdef profile_TEPS
device_obj.TEPS_value =0;
device_obj.TEPS_value_perthread= (ull_t*) malloc (BLKS_NUM*blockSize*sizeof(ull_t));
H_ERR(cudaMalloc((void**) &device_obj.TEPS_value_perthread_d,sizeof( ull_t)*BLKS_NUM*blockSize)); //size of lock_d is num of blocks
H_ERR(cudaMemset(device_obj.TEPS_value_perthread_d,0,sizeof( ull_t)*BLKS_NUM*blockSize));
#endif
    H_ERR(cudaMallocManaged((void**) &device_obj.fill_in_d,sizeof(int_t)*N_src_group*vert_count)); 
    H_ERR(cudaMemset(device_obj.fill_in_d, 0, sizeof(int_t)*N_src_group*vert_count));

    H_ERR(cudaMallocManaged((void**) &device_obj.nz_row_U_d,sizeof(int_t)*N_src_group)); 
    H_ERR(cudaMemset(device_obj.nz_row_U_d, 0, sizeof(int_t)*N_src_group));
    H_ERR(cudaMallocManaged((void**) &device_obj.pass_through_d,sizeof(int_t)*N_src_group)); 
    H_ERR(cudaMemset(device_obj.pass_through_d, 0, sizeof(int_t)*N_src_group));


    device_obj.max_id_offset = MAX_VAL-vert_count;//vert_count*group;
    device_obj.group_MAX_VAL = device_obj.max_id_offset + vert_count;
    device_obj.count_group_loop=0;


    H_ERR(cudaMallocManaged((void**) &device_obj.count,sizeof(int_t))); 
    device_obj.count[0] =0;
    // H_ERR(cudaMallocManaged((void**) &device_obj.csr_d,sizeof(int_t)*edge_count)); 
    H_ERR(cudaMalloc((void**) &device_obj.csr_d,sizeof(int_t)*edge_count)); 


    // H_ERR(cudaMallocManaged((void**) &device_obj.col_st_d,sizeof(int_t)*vert_count)); 
    H_ERR(cudaMalloc((void**) &device_obj.col_st_d,sizeof(int_t)*vert_count)); 



    // H_ERR(cudaMallocManaged((void**) &device_obj.col_ed_d,sizeof(int_t)*vert_count)); 
    H_ERR(cudaMalloc((void**) &device_obj.col_ed_d,sizeof(int_t)*vert_count)); 


    H_ERR(cudaMemcpy(device_obj.csr_d, csr,sizeof(int_t)*edge_count,cudaMemcpyHostToDevice));

    H_ERR(cudaMemcpy(device_obj.col_st_d, col_st,sizeof(int_t)*vert_count,cudaMemcpyHostToDevice));

    H_ERR(cudaMemcpy(device_obj.col_ed_d, col_ed,sizeof(int_t)*vert_count,cudaMemcpyHostToDevice));




    H_ERR(cudaMalloc((void**) &device_obj.next_source_d,sizeof(int_t))); 
    H_ERR(cudaMalloc((void**) &device_obj.frontier_size_d,sizeof(int_t))); 
    H_ERR(cudaMemset(device_obj.frontier_size_d, 0, sizeof(int_t)));

    H_ERR(cudaMalloc((void**) &device_obj.temp_next_frontier_size_d,sizeof(int_t))); 
    H_ERR(cudaMemset(device_obj.temp_next_frontier_size_d, 0, sizeof(int_t)));
    H_ERR(cudaMalloc((void**) &device_obj.cost_array_d,sizeof(uint_t)*N_src_group*vert_count)); 
    device_obj.fill_in_last_row_h =  (int_t*) malloc (vert_count*sizeof(int_t));
    device_obj.last_row_U_count = 0;
    H_ERR(cudaMalloc((void**) &device_obj.frontier_d,sizeof(int_t)*real_allocation)); 
    H_ERR(cudaMalloc((void**) &device_obj.next_frontier_d,sizeof(int_t)*real_allocation)); 
    H_ERR(cudaMalloc((void**) &device_obj.fill_count_d,sizeof(ull_t))); 
    H_ERR(cudaMemset(device_obj.fill_count_d, 0, sizeof(ull_t)));
    device_obj.fill_count=0;

    H_ERR(cudaMalloc((void**) &device_obj.group80_count_d,sizeof(ull_t))); 
    H_ERR(cudaMemset(device_obj.group80_count_d, 0, sizeof(ull_t)));
    device_obj.group80_count=0;


    H_ERR(cudaMalloc((void**) &device_obj.src_frontier_d,sizeof(int_t)*real_allocation));  //stores the code (mapping) of source not the source itself
    H_ERR(cudaMalloc((void**) &device_obj.next_src_frontier_d,sizeof(int_t)*real_allocation)); 
    H_ERR(cudaMalloc((void**) &device_obj.source_d,sizeof(int_t)*N_src_group));
    H_ERR(cudaMalloc((void**) &device_obj.my_current_frontier_d,sizeof(int_t)*BLKS_NUM*blockSize));  
    H_ERR(cudaMalloc((void**) &device_obj.lock_d,sizeof(int)*BLKS_NUM)); //size of lock_d is num of blocks
    H_ERR(cudaMemset(device_obj.lock_d, 0, sizeof(int)*BLKS_NUM));
    H_ERR(cudaMalloc((void**) &device_obj.next_front_d,sizeof( int_t))); //size of lock_d is num of blocks
    H_ERR(cudaMemcpy(device_obj.next_front_d,&next_front,sizeof(int_t),cudaMemcpyHostToDevice));
    H_ERR(cudaMallocHost((void**)&device_obj.CPU_buffer_next_level,N_src_group*vert_count*sizeof(int_t)));
    H_ERR(cudaMallocHost((void**)&device_obj.CPU_buffer_next_level_source,N_src_group*vert_count*sizeof(int_t)));
    device_obj.size_CPU_buffer_next_level=0;
    H_ERR(cudaMallocHost((void**)&device_obj.CPU_buffer_current_level,N_src_group*vert_count*sizeof(int_t)));
    H_ERR(cudaMallocHost((void**)&device_obj.CPU_buffer_current_level_source,N_src_group*vert_count*sizeof(int_t)));
    device_obj.size_CPU_buffer_current_level=0;
    device_obj.next_N_buffer_CPU=(int*) malloc (10000*sizeof(int_t));
    device_obj.current_N_buffer_CPU=(int*) malloc (10000*sizeof(int_t));
    device_obj.N_non_zero_rows = (int*) malloc (vert_count*sizeof(int_t));
    device_obj.my_supernode = (int*) malloc (vert_count*sizeof(int_t));
    // for (int i=0; i<vert_count;i++)
    // {
    //     device_obj.my_supernode[i]=i;
    // }
    device_obj.my_representative = (int*) malloc (vert_count*sizeof(int_t));//representative in supernodes optmizations
    H_ERR(cudaMallocHost((void**)&device_obj.my_supernode_d,vert_count*sizeof(int_t)));
    H_ERR(cudaMemcpy(device_obj.my_supernode_d,device_obj.my_supernode,sizeof(int_t)* vert_count,cudaMemcpyHostToDevice));
    H_ERR(cudaMallocManaged((void**) &device_obj.swap_CPU_buffers_m,sizeof( int_t))); 
    H_ERR(cudaMemset(device_obj.swap_CPU_buffers_m,0,sizeof( int_t)));
    H_ERR(cudaMallocManaged((void**) &device_obj.swap_GPU_buffers_m,sizeof( int_t))); 
    H_ERR(cudaMemset(device_obj.swap_GPU_buffers_m,0,sizeof( int_t)));
    H_ERR(cudaMallocManaged((void**) &device_obj.next_buffer_m,sizeof( int_t))); 
    H_ERR(cudaMallocManaged((void**) &device_obj.current_buffer_m,sizeof( int_t))); 
    device_obj.current_buffer_m[0]=0;
    device_obj.next_buffer_m[0]=0;
    H_ERR(cudaMallocManaged((void**) &device_obj.dump_m,sizeof( int_t)));
    H_ERR(cudaMemset(device_obj.dump_m,0,sizeof( int_t)));
    H_ERR(cudaMallocManaged((void**) &device_obj.load_m,sizeof( int_t))); 
    H_ERR(cudaMemset(device_obj.load_m,0,sizeof( int_t)));
    device_obj.buffer_flag_host=0;
    device_obj.buffer_flag_getfronter_host=0;
    H_ERR(cudaMallocManaged((void**) &device_obj.offset_next_kernel,sizeof( int_t))); 
    device_obj.offset_next_kernel[0]= 0;
    H_ERR(cudaMallocManaged((void**) &device_obj.offset_kernel,sizeof( int_t))); 
    device_obj.offset_kernel[0]= 0;
    device_obj.N_dumping_cpu_memory=0;
    device_obj.N_reading_cpu_memory=0;
    H_ERR(cudaMalloc((void**) &device_obj.next_frontier_size_d,sizeof(int_t)*N_src_group)); 
    H_ERR(cudaMemset(device_obj.next_frontier_size_d, 0, sizeof(int_t)*N_src_group));




    // H_ERR(cudaMalloc((void**) &device_obj.pass_through_d,sizeof(int_t)*vert_count)); 
    // H_ERR(cudaMemset(device_obj.pass_through_d, 0, sizeof(int_t)*vert_count));
    H_ERR(cudaMalloc((void**) &device_obj.source_flag_d,sizeof(int_t)*vert_count)); 
    H_ERR(cudaMemset(device_obj.source_flag_d, 0, sizeof(int_t)*vert_count));
    //  H_ERR(cudaMemcpy(device_obj.source_flag,device_obj.source_flag_d,sizeof(int_t)* vert_count,cudaMemcpyDeviceToHost));
    device_obj.super_obj = new Super [vert_count];
    device_obj.source_flag= (int*) malloc (vert_count*sizeof(int_t));
    H_ERR(cudaMemcpy(device_obj.source_flag,device_obj.source_flag_d,sizeof(int_t)* vert_count,cudaMemcpyDeviceToHost));
    H_ERR(cudaMallocHost((void**)&device_obj.nz_row_U_h,vert_count*sizeof(int_t)));
    H_ERR(cudaMalloc((void**)&device_obj.Nsup_d,sizeof(int_t)));
    H_ERR(cudaMemset(device_obj.Nsup_d, 0, sizeof(int_t)));
    device_obj.superObj = (Super*) malloc (vert_count*sizeof(Super));
    H_ERR(cudaMalloc((void**) &device_obj.superObj_d,sizeof(Super)*vert_count)); 
    // for (int i=0; i<vert_count;i++)
    // {
    //     device_obj.superObj[i].end=i;
    //     device_obj.my_supernode[i]=i;
    // }
    H_ERR(cudaMalloc((void**) &device_obj.validity_supernode_d,sizeof(int_t)*vert_count));
    H_ERR(cudaMemset(device_obj.validity_supernode_d, 0, sizeof(int_t)*vert_count));
    H_ERR(cudaMallocHost((void**)&device_obj.validity_supernode_h,vert_count*sizeof(int_t)));
    memset(device_obj.validity_supernode_h, 0, vert_count*sizeof(int_t));
    H_ERR(cudaMemcpy(device_obj.superObj_d,device_obj.superObj,sizeof(Super)* vert_count,cudaMemcpyHostToDevice));
    H_ERR(cudaMalloc((void**) &device_obj.Nsup_per_chunk_d,sizeof(int_t)*N_chunks)); 
    H_ERR(cudaMemset(device_obj.Nsup_per_chunk_d, 0, sizeof(int_t)*N_chunks));
    H_ERR(cudaMalloc((void**) &device_obj.fill_count_per_row_d,sizeof(int_t)*vert_count)); 
    H_ERR(cudaMemset(device_obj.fill_count_per_row_d, 0, sizeof(int_t)*vert_count));
    //  H_ERR(cudaMalloc((void**) &device_obj.new_col_ed_d,sizeof(int_t)*vert_count)); 
    device_obj.fill_count_per_row = (int*) malloc (vert_count*sizeof(int_t));//representative in supernodes optmizations
    H_ERR(cudaMalloc((void**) &device_obj.N_edge_checks_per_thread_d,sizeof(ull_t)*BLKS_NUM*blockSize)); 
    H_ERR(cudaMemset(device_obj.N_edge_checks_per_thread_d, 0, sizeof(ull_t)*BLKS_NUM*blockSize));
    device_obj.N_edge_checks_per_thread = (ull_t*) malloc (BLKS_NUM*blockSize*sizeof(ull_t));



}
#ifdef workload_stealing
int RequestStealing (int myrank, int victim)
{
    struct request_package send_request;
    send_request.thief_id = myrank;
    printf("MPI process %d sends stealing request with thief id = %d\n", myrank, send_request.thief_id);
    MPI_Send(&send_request, 1, request_type, victim, 1000, MPI_COMM_WORLD);

    // MPI_Request request;
    // MPI_Status wait_status;
    // MPI_Isend(&send_request, 1, request_type, victim, 1000, MPI_COMM_WORLD, &request);
    // MPI_Wait(&request, &wait_status);


    struct response_package recv_response;
    MPI_Status status;
    MPI_Recv(&recv_response, 1, response_type, victim, 123, MPI_COMM_WORLD, &status);
    printf("MPI process %d receives victim_id = %d\n\t- thief_id = %d\n\t- work = %d\n\t- stolen_chunk_id = %d\n", myrank, recv_response.victim_id, recv_response.thief_id , recv_response.work,recv_response.stolen_chunk_id);
    return recv_response.stolen_chunk_id;

}
#endif

int Get_Big_chunk(int myrank,int& tail_pointer, int& last_stolen_bigchunk)
{
    int return_bigChunk = -1;
    if (tail_pointer >= threshold)
    {
        //allow the work to be stolen
        return_bigChunk = last_stolen_bigchunk--;// = last_stolen_bigchunk-1;
        tail_pointer--;
    }
    return return_bigChunk;
}



#ifdef workload_stealing
int get_group_id(int N_gpu, int stolen_chunk_id)
{
    //Note: Stolen chunk id is 1 based indexing
    // return ((stolen_chunk_id-1)*N_gpu);
    return ((stolen_chunk_id)*N_gpu);
}


void CreateTokenType()
{
    int lengths[4] = {1,1,1};
    const MPI_Aint displacements[3] = { 0, sizeof(int), 2*sizeof(int)  };
    MPI_Datatype types[3] = { MPI_INT, MPI_INT, MPI_INT};
    MPI_Type_create_struct(3, lengths, displacements, types, &token_type);
    MPI_Type_commit(&token_type);
}
#endif

void symbfact_min_id(int args,char** argv,int myrank,ull_t& fill_count, double& time, int& N_supernodes,ull_t& Process_TEPS) 
//int main(int args, char **argv)
{
    std::cout<<"Input: ./exe beg csr weight #Processes chunk_size percent_cat0 percent_cat1 N_blocks_source_cat2 N_GPU_Node N_src_group;\n";
    if(args!=11){std::cout<<"Wrong input\n";exit(1);}

    const char *beg_file=argv[1];
    const char *end_file=argv[2];
    const char *csr_file=argv[3];
    int num_process=atoi(argv[4]);
    // int chunk_size=atoi(argv[5]);
    int_t max_supernode_size=atoi(argv[5]);// Should be power of 2
    int percent_cat0=atoi(argv[6]);
    int percent_cat2=atoi(argv[7]);
    int  N_blocks_source_cat2 = atoi(argv[8]);
    int N_GPU_Node=atoi(argv[9]);
    int N_gpu = num_process*N_GPU_Node;
    cout<<"N_gpu: "<<N_gpu<<endl;
    cout<<"N_GPU_Node: "<<N_GPU_Node<<endl;
#ifdef workload_stealing
    CreateResponseType();
    CreateRequestType();
    CreateTokenType();
    Initialize(myrank,num_process);
#endif
    //int N_src_group=atoi(argv[10]);

    printf("My rank:%d\n",myrank);
#ifdef lambda
    //2 process in a node with 4 GPU per process
    H_ERR(cudaSetDevice(myrank * N_GPU_Node));
#else
    H_ERR(cudaSetDevice(myrank % N_GPU_Node));
#endif
    int device;
    H_ERR(cudaGetDevice(&device));
    cout<<"rank "<<myrank<<"has local GPU:"<<device<<endl;
    cout<<"N_blocks_source_cat2: "<<N_blocks_source_cat2<<endl; 
    FILE* fptr;
    if ((fptr = fopen(csr_file,"r")) == NULL)
    {
        printf("Error! opening csr file");
        // Program exits if the file pointer returns NULL.
        exit(1);
    }
    printf("Reading CSR \n");
    int_t edge_count;
    int_t vert_count;
    fscanf(fptr,"%d", &edge_count);
    printf("Number of edges=%d\n", edge_count);

    fscanf(fptr,"%d", &vert_count);
    printf("Number of vertices=%d\n", vert_count);


    int_t* csr= (int*) malloc (edge_count*sizeof(int_t));
    for (int_t i=0;i<edge_count;i++)
    {
        fscanf(fptr,"%d", &csr[i]);
    }
    fclose(fptr); 
    printf("Reading col_begin \n");
    if ((fptr = fopen(beg_file,"r")) == NULL)
    {
        printf("Error! opening col_beg file");
        // Program exits if the file pointer returns NULL.
        exit(1);
    }
    int_t* col_st= (int*) malloc (vert_count*sizeof(int_t));
    for (int_t i=0;i<edge_count;i++)
    {
        fscanf(fptr,"%d", &col_st[i]);
    }
    fclose(fptr); 
    if ((fptr = fopen(end_file,"r")) == NULL)
    {
        printf("Error! opening col end file");
        // Program exits if the file pointer returns NULL.
        exit(1);
    }
    printf("Reading col_end \n");

    int_t* col_ed= (int*) malloc (vert_count*sizeof(int_t));
    for (int_t i=0;i<edge_count;i++)
    {
        fscanf(fptr,"%d", &col_ed[i]);
    }
    fclose(fptr); 


#ifdef enable_transpose  
    cout<<"Transpose ENABLED!"<<endl;
    cout<<"Graph before transpose"<<endl;
    for (int_t i =11930;i < 11940; i++)
    {
        // std::vector<int> temp_vector;
        int_t begin = col_st[i];
        int_t end = col_ed[i];
        cout<<"vertex "<<i<<" neighbors: ";
        while (begin < end)
        {
            int_t neighbor= csr[begin];
            cout << neighbor<<" ";
            begin++;
        }
        cout<<endl;
    }


    Transpose_Matrix(csr,col_st,col_ed,vert_count,edge_count);

    //After Transpose
    cout<<"Graph after transpose"<<endl;
    for (int_t i =11930;i < 11940; i++)
    {
        // std::vector<int> temp_vector;
        int_t begin = col_st[i];
        int_t end = col_ed[i];
        cout<<"vertex "<<i<<" neighbors: ";
        while (begin < end)
        {
            int_t neighbor = csr[begin];
            cout << neighbor<<" ";
            begin++;
        }
        cout<<endl;
    }
#else
    cout<<"Transpose DISABLED!"<<endl;
#endif

#ifdef enable_sort_edge_list
    cout<<"Sorting ENABLED!"<<endl;
    cout<<"Graph before sorting"<<endl;
    for (int_t i =5000;i < 5010; i++)
    {
        // std::vector<int> temp_vector;
        int_t begin = col_st[i];
        int_t end = col_ed[i];
        cout<<"vertex "<<i<<" neighbors: ";
        while (begin < end)
        {
            int_t neighbor= csr[begin];
            cout << neighbor<<" ";
            begin++;
        }
        cout<<endl;
    }
    SortEdgeList(csr,col_st,col_ed,vert_count,edge_count);

    cout<<"Graph after sorting"<<endl;
    for (int_t i =5000;i < 5010; i++)
    {
        // std::vector<int> temp_vector;
        int_t begin = col_st[i];
        int_t end = col_ed[i];
        cout<<"vertex "<<i<<" neighbors: ";
        while (begin < end)
        {
            int_t neighbor = csr[begin];
            cout << neighbor<<" ";
            begin++;
        }
        cout<<endl;
    }
#else
    cout<<"Sorting DISABLED!"<<endl;
#endif

    // int_t sources_per_process=vert_count/num_process;
    int_t N_chunks= (int) ceil (vert_count/(float)max_supernode_size);

    int N_src_group = Compute_Src_group(vert_count);
    // N_src_group = 128;
    // N_src_group=atoi(argv[10]);

// if (N_src_group * N_GPU_Node*num_process > vert_count)
// {
  
//     N_src_group = ceil((vert_count)/(float)(N_GPU_Node*num_process));
//     N_src_group=(int)log_2(N_src_group);
//     N_src_group=pow(2,N_src_group);
// }
    cout<<"N_src_group: "<<N_src_group<<endl;

    aux_device *dev_mem;
    dev_mem = (aux_device *)malloc(N_GPU_Node*sizeof(aux_device));

    int_t real_allocation=allocated_frontier_size*1.1;
    cout<<"Allocated frontier size: "<<allocated_frontier_size<<endl;
    cout<<"Real allocated frontier size: "<<real_allocation<<endl;

    int div_factor=ceil(log_2(N_src_group)/(double)8);
    cout<<"div_factor: "<<div_factor<<endl;
    int cfg_blockSize=128; 
    int BLKS_NUM,blockSize;
    cudaOccupancyMaxPotentialBlockSize( &BLKS_NUM, &blockSize, Compute_fillins_joint_traversal_group_wise_supernode_OptIII, 0, 0);
    H_ERR(cudaDeviceSynchronize());
    BLKS_NUM = (blockSize * BLKS_NUM)/cfg_blockSize;
    int exp=log2((float)BLKS_NUM);
    // int exp=std::log2((float)BLKS_NUM);
    BLKS_NUM=pow(2,exp);
    blockSize = cfg_blockSize;

    // BLKS_NUM =1;
    // blockSize =32;

    cout<<"Detected GridDim: "<<BLKS_NUM<<endl;
    if (blockSize < 32) cout<<"Blocksize is less than 32. Causes problem in source scheduling in supernode implementation";

    int_t N_groups=(ceil) (vert_count/(float)N_src_group);
    // Barrier global_barrier(BLKS_NUM); 
    cout<<"BLKS_NUM: "<<BLKS_NUM<<endl;
    cout<<"blockSize: "<<blockSize<<endl;
#ifdef thread_centric
    int_t next_front=BLKS_NUM*blockSize;
#else
    int_t next_front= (BLKS_NUM*blockSize) >> 5;
#endif
    //Initialize and allocate memory for GPU 0
    double allocation_start_time = wtime();
    cudaStream_t* stream = (cudaStream_t*) malloc ((N_GPU_Node)*sizeof(cudaStream_t));



    int_t **fill_2d;
    int_t** nz_2d ;
    fill_2d = new int*[N_GPU_Node];
    nz_2d=  new int*[N_GPU_Node];

    double streamcreate_start;
    double streamcreate_time=0;
    for (int i=0;i<N_GPU_Node;i++)
    {
        // cout<<"selecting GPU: "<<i<<endl;
        streamcreate_start = wtime();
#ifdef lambda
        H_ERR(cudaSetDevice(i+myrank*N_GPU_Node)); 
#else
        H_ERR(cudaSetDevice(i)); 
#endif
        H_ERR( cudaStreamCreate(&stream[i]));	
        streamcreate_time += wtime() - streamcreate_start;
        Allocate_Initialize (dev_mem[i], vert_count, edge_count, csr, col_st, col_ed, BLKS_NUM, blockSize,i,next_front,N_src_group,real_allocation, N_chunks);
        fill_2d[i] = dev_mem[i].fill_in_d;
        nz_2d[i] = dev_mem[i].nz_row_U_d;
    }

    int_t N_section=3;



    // int_t* frontier_size_d;








    //Only required for sequential supernodal detection
    int_t* fill_in_h =  (int_t*) malloc (N_src_group*vert_count*sizeof(int_t));
    //~Only required for sequential supernodal detection



    //not used
    int* frontierchecked;
    ull_t group_80_count_total =0 ;
    //not used


    //Not used
    int_t * new_col_ed_d;
    H_ERR(cudaMallocManaged((void**) &new_col_ed_d,sizeof(int_t)*vert_count)); 
    //~Not used

    cout<<"Running the merge kernel"<<endl;


    // uint_t max_id_offset = MAX_VAL-vert_count;//vert_count*group;
    // uint_t group_MAX_VAL = max_id_offset + vert_count;
    // int_t reinitialize=1;
    uint_t group_loops=MAX_VAL/vert_count;
    // uint_t count_group_loop=0;


    double start_time;
    double dumping_loading_time=0;
    ull_t last_fill_count=0;
    int_t temp_allocated_frontier_size=allocated_frontier_size;
    ull_t size_copied=0;
    double start_only_Copy=0;
    double time_only_Copy=0;

    int_t N_small_chunks =  (int) ceil (vert_count/(float)max_supernode_size);

    int_t Nsup =-1;

    int_t member_count=max_supernode_size; //count of members in the current supernode
    // ull_t* N_edge_checks_per_thread = (ull_t*) malloc (BLKS_NUM*blockSize*sizeof(ull_t));
    double T3_parallel_detection =0;
    double T3_parallel_detection_temp=0;
    std::fstream nz_count_U_file;
    nz_count_U_file.open("nz_count_U_file.dat",std::fstream::out);

    double allocation_time = wtime()-allocation_start_time;

    // 
    double* start_all=new double[4];
    double* end_all=new double[4];
    bool test =true;
    double time_response_polling =0;
    double time_stealing_response =0;
    double time_stealing_request =0;
    double initialization_overhead =0;

    #ifdef detect_supernodes
    #ifdef enable_memadvise
    for (int i=0;i<N_GPU_Node;i++)
    {
        ///Advise: Set Accessed by
        /////**** This advice is recommended in scenarios where data locality is not important, but avoiding faults is.
        /////***** Consider for example a system containing multiple GPUs with peer-to-peer access enabled, where the
        /////**** * * data located on one GPU is occasionally accessed by peer GPUs. In such scenarios, migrating data
        /////**** * * data located on one GPU is occasionally accessed by peer GPUs. In such scenarios, migrating data
        /////**** * * over to the other GPUs is not as important because the accesses are infrequent and the overhead of
        /////**** * * migration may be too high. But preventing faults can still help improve performance, and so having
        /////**** * * a mapping set up in advance is useful.


        H_ERR(cudaMemAdvise(dev_mem[i].fill_in_d, sizeof(int)*N_src_group*vert_count, cudaMemAdviseSetPreferredLocation, i));// Advising for cudaMemAdviseSetAccessedBy
        H_ERR(cudaMemAdvise(dev_mem[i].nz_row_U_d, sizeof(int)*N_src_group, cudaMemAdviseSetPreferredLocation, i));// Advising for cudaMemAdviseSetAccessedBy   
        H_ERR(cudaMemAdvise(dev_mem[i].pass_through_d, sizeof(int)*N_src_group, cudaMemAdviseSetPreferredLocation, i));// Advising for cudaMemAdviseSetAccessedBy   

        for (int j =1;j<N_GPU_Node; j++)
        {
            H_ERR(cudaMemAdvise(dev_mem[i].fill_in_d, sizeof(int)*N_src_group*vert_count, cudaMemAdviseSetAccessedBy, (i+j)%N_GPU_Node));// Advising for cudaMemAdviseSetAccessedBy
            H_ERR(cudaMemAdvise(dev_mem[i].nz_row_U_d, sizeof(int)*N_src_group, cudaMemAdviseSetAccessedBy, (i+j)%N_GPU_Node));// Advising for cudaMemAdviseSetAccessedBy   
            H_ERR(cudaMemAdvise(dev_mem[i].pass_through_d, sizeof(int)*N_src_group, cudaMemAdviseSetAccessedBy, (i+j)%N_GPU_Node));// Advising for cudaMemAdviseSetAccessedBy   
        }
        // H_ERR(cudaMemAdvise(dev_mem[i].fill_in_d, sizeof(int)*N_src_group*vert_count, cudaMemAdviseSetAccessedBy, (i+1)%N_GPU_Node));// Advising for cudaMemAdviseSetAccessedBy
        // H_ERR(cudaMemAdvise(dev_mem[i].fill_in_d, sizeof(int)*N_src_group*vert_count, cudaMemAdviseSetAccessedBy, (i+2)%N_GPU_Node));// Advising for cudaMemAdviseSetAccessedBy
        // H_ERR(cudaMemAdvise(dev_mem[i].fill_in_d, sizeof(int)*N_src_group*vert_count, cudaMemAdviseSetAccessedBy, (i+3)%N_GPU_Node));// Advising for cudaMemAdviseSetAccessedBy

        // H_ERR(cudaMemAdvise(dev_mem[i].nz_row_U_d, sizeof(int)*N_src_group, cudaMemAdviseSetAccessedBy, (i+1)%N_GPU_Node));// Advising for cudaMemAdviseSetAccessedBy   
        // H_ERR(cudaMemAdvise(dev_mem[i].nz_row_U_d, sizeof(int)*N_src_group, cudaMemAdviseSetAccessedBy, (i+2)%N_GPU_Node));// Advising for cudaMemAdviseSetAccessedBy   
        // H_ERR(cudaMemAdvise(dev_mem[i].nz_row_U_d, sizeof(int)*N_src_group, cudaMemAdviseSetAccessedBy, (i+3)%N_GPU_Node));// Advising for cudaMemAdviseSetAccessedBy   

        // H_ERR(cudaMemAdvise(dev_mem[i].pass_through_d, sizeof(int)*N_src_group, cudaMemAdviseSetAccessedBy, (i+1)%N_GPU_Node));// Advising for cudaMemAdviseSetAccessedBy   
        // H_ERR(cudaMemAdvise(dev_mem[i].pass_through_d, sizeof(int)*N_src_group, cudaMemAdviseSetAccessedBy, (i+2)%N_GPU_Node));// Advising for cudaMemAdviseSetAccessedBy   
        // H_ERR(cudaMemAdvise(dev_mem[i].pass_through_d, sizeof(int)*N_src_group, cudaMemAdviseSetAccessedBy, (i+3)%N_GPU_Node));// Advising for cudaMemAdviseSetAccessedBy   

    }
    

    #endif
    #endif

    int size_big_chunk = N_src_group*N_GPU_Node;
    cout<<"size_big_chunk: "<<size_big_chunk<<endl;
    int N_big_chunks = (ceil) (vert_count/(float)(size_big_chunk));

    int N_src_process = N_GPU_Node * N_src_group;
    int N_src_loop = num_process * N_src_process;
    // cout<<"N_big_chunks: "<<N_big_chunks<<endl;

    // int total_big_chunks_per_node = N_big_chunks/num_process;
    int total_big_chunks_per_node = (ceil)(vert_count/(float)N_src_loop);


    // if (myrank == (num_process-1)) total_big_chunks_per_node = total_big_chunks_per_node + (N_big_chunks - total_big_chunks_per_node*num_process);
    // #ifdef print_status
    cout<<"My total_big_chunks_per_node: "<<total_big_chunks_per_node<<endl;
    // #endif
    int* tail_pointer = new int[num_process];
    // int tail_pointer= total_big_chunks_per_node;
    int last_stolen_bigchunk= total_big_chunks_per_node;
    int Total_N_head_pointer = 0;
    int* head_pointer = new int[num_process];
    for (int i=0;i<num_process;i++)
    {
        head_pointer[i] = -1;
        tail_pointer[i] = total_big_chunks_per_node-1;
    }
    // int head_pointer =0;

#ifdef RDMA

    MPI_Win win; 
    MPI_Win_create(tail_pointer, num_process * sizeof(int), sizeof(int), MPI_INFO_NULL,MPI_COMM_WORLD, &win);
    // int head_pointer=0;
    MPI_Win win_processed; 
    // MPI_Win_create(big_chunks_area, 2*sizeof(int), sizeof(int), MPI_INFO_NULL,MPI_COMM_WORLD, &win);
    MPI_Win_create(head_pointer, num_process * sizeof(int), sizeof(int), MPI_INFO_NULL,MPI_COMM_WORLD, &win_processed);

    cout<<"Windows Created!"<<endl;
#endif
    bool break_process = false;

    //New source scheduling
    int chunk_size = max_supernode_size;
    int num_nodes = num_process;
    int total_num_chunks = ceil(vert_count/(float)chunk_size);
    int num_curr_chunks_per_node = (N_src_group * N_GPU_Node)/chunk_size;

    int total_num_chunks_per_node = ceil(total_num_chunks/(float)num_nodes);
    if (num_curr_chunks_per_node > total_num_chunks_per_node) num_curr_chunks_per_node = total_num_chunks_per_node;
    int* mysrc = new int[N_GPU_Node*N_src_group];
    int* counter = new int[N_GPU_Node];
    int iteration = 0;
    int* my_chunks = new int[num_curr_chunks_per_node];
    //~New source scheduling

    start_time=wtime();

    for (int i=0;i<N_GPU_Node;i++)
    {
#ifdef lambda
        H_ERR(cudaSetDevice(i+myrank*N_GPU_Node)); 
#else
        H_ERR(cudaSetDevice(i)); 
#endif
        Initialise_cost_array<<<128,128,0,stream[i]>>>( dev_mem[i].cost_array_d,vert_count,dev_mem[i].group_MAX_VAL,N_src_group);
        // H_ERR(cudaDeviceSynchronize());
    } 
    int_t group=0;
    int source=0;
    // int_t group =80;
    // for (int_t group=0; group< N_groups;group+=N_gpu)//N_gpu=num_process

    // for (int_t group=0; group< N_groups;group+=N_gpu)//N_gpu=num_process

// int iteration =20;
    int number_iterations = ((total_num_chunks_per_node + num_curr_chunks_per_node)/num_curr_chunks_per_node);
    for (int local_chunk_count=0; local_chunk_count < (total_num_chunks_per_node + num_curr_chunks_per_node); local_chunk_count+=num_curr_chunks_per_node)//To ensure all the chunks are processsed
    
    {	
        // int i=0;
        int node_offset = myrank * N_src_group * N_GPU_Node;
        int starting_source = node_offset + group*N_src_group;

        // source_d[thid]=gpu_id + thid * N_gpu + group * N_src_group;
        #ifdef workload_stealing
        // if (tail_pointer == 0)
        double start_response_lock = wtime();
        // MPI_Win_lock(MPI_LOCK_EXCLUSIVE,myrank,0,win);    //head to tail pointer
        // MPI_Win_lock(MPI_LOCK_EXCLUSIVE,myrank,0,win_processed); //
        // head_pointer[myrank]++;
        int old_value = __sync_fetch_and_add (&head_pointer[myrank], 1);

        if (old_value +1 > total_big_chunks_per_node) 
        {
            break;// Reached at the end of the chunk list
        }
        else
        {
            //computing the right range of chunk to be worked on
            group = (old_value+1) * N_gpu;
            cout<<"Process: "<<myrank<<" is processing its chunk: "<<old_value+1<<endl;
        }


        time_stealing_response += (wtime()-start_response_lock);
        double start_chunk_time = wtime();
        // if (break_process)
        // {
        //     cout<<"Process: "<<myrank<<" breaking the loop!"<<endl;
        //     break;
        // }
#endif
        // cout<<"At iteration: "<<iteration<<endl;
        // cout<<"Assigning the chunks to the process/node: "<<myrank<<endl;
        // cout<<"my_chunks: {";

        int starting_chunk = iteration * num_nodes * num_curr_chunks_per_node + myrank;
        my_chunks[0] = starting_chunk;
        // cout<< my_chunks[0]<<",";
        for (int j=1; j<num_curr_chunks_per_node; j++)
        {
            my_chunks[j] = my_chunks[j-1] + num_nodes;
            // cout<< my_chunks[j]<<",";
        }
        // cout<<"}"<<endl;
        // if (starting_chunk > N_chunks)
        // {
        //     continue;
        // }
        // cout<<"Computing sources for the GPUs!"<<endl;
        memset(counter, 0, N_GPU_Node*sizeof(int));
        int ptr = 0;
        for (int iter=0; iter < num_curr_chunks_per_node; iter++)
        {
            //Assigning 1 chunk in 1 iter
            int source_begin = my_chunks[iter] * chunk_size;
            for (int j = 0; j < chunk_size; j++)
            {
                int source = source_begin+j;
                int local_gpu_id = ptr % N_GPU_Node;
                mysrc[local_gpu_id*N_src_group+counter[local_gpu_id]] = source;
                // if (myrank ==0)
                //   std::cout<<"src "<<source <<" --> Process "<<myrank<<"\n";
                counter[local_gpu_id]++;
                ptr ++;
            }
        }



        // cout<<"Process: "<<myrank<<" Starting source: "<<starting_source<<" Ending source: "<< starting_source +  N_src_group * N_gpu <<"group: "<<group<< endl;
        for (int i=0;i<N_GPU_Node;i++)
        {
#ifdef lambda
            H_ERR(cudaSetDevice(i+myrank*N_GPU_Node)); 
#else
            H_ERR(cudaSetDevice(i)); 
#endif
            dev_mem[i].swap_GPU_buffers_m[0] = 0;
            // myrank =0;
            int myGlobalrank = myrank * N_GPU_Node + i;
            // int myGlobalrank =  N_GPU_Node +i;


            H_ERR(cudaMemcpy(dev_mem[i].source_d, &mysrc[i*N_src_group] , N_src_group*sizeof(int_t),cudaMemcpyHostToDevice));//Copying the respective sources from the CPU to GPU. New source scheduling

            int last_heighest_source=0;
            int valid_index =0;
            for (int k=0;k<N_src_group;k++)
            {
                // printf("%d\t",mysrc[i*N_src_group+k]);
                if (last_heighest_source <  mysrc[i*N_src_group+k])
                {
                    last_heighest_source=mysrc[i*N_src_group+k];
                    if (mysrc[i*N_src_group+k] < vert_count)
                    {
                        valid_index = k+1;
                    }
                    else
                    {
                        // printf("valid index of GPU:%d is %d\n",i,valid_index);
                    }
                }
                // if (last_heighest_source > mysrc[i*N_src_group+k])
                // {
                //     // printf("Found decreasing source at GPU: %d\n",i);

                //     // printf("valid index of GPU:%d is %d\n",i,valid_index);


                // }

            }

#ifdef thread_centric
            Compute_fillins_joint_traversal_group_wise_supernode_OptIII <<<BLKS_NUM,blockSize,0, stream[i]>>>  (dev_mem[i].cost_array_d,dev_mem[i].fill_in_d,dev_mem[i].frontier_d,
                    dev_mem[i].next_frontier_d,vert_count,dev_mem[i].csr_d,dev_mem[i].col_st_d,dev_mem[i].col_ed_d,dev_mem[i].fill_count_d,myGlobalrank,N_gpu,
                    dev_mem[i].src_frontier_d,dev_mem[i].next_src_frontier_d,dev_mem[i].source_d, dev_mem[i].frontier_size_d, dev_mem[i].next_frontier_size_d,
                    dev_mem[i].lock_d,N_groups,
                    dev_mem[i].dump_m,dev_mem[i].load_m,
                    N_src_group,group,dev_mem[i].max_id_offset,dev_mem[i].next_front_d,temp_allocated_frontier_size,
                    dev_mem[i].my_current_frontier_d,frontierchecked,dev_mem[i].swap_GPU_buffers_m,
                    dev_mem[i].nz_row_U_d, max_supernode_size,  N_small_chunks,dev_mem[i].source_flag_d, dev_mem[i].my_supernode_d, dev_mem[i].superObj_d,dev_mem[i].validity_supernode_d,
                    dev_mem[i].pass_through_d,dev_mem[i].fill_count_per_row_d, dev_mem[i].N_edge_checks_per_thread_d,new_col_ed_d,N_GPU_Node,i,dev_mem[i].group80_count_d,dev_mem[i].TEPS_value_perthread_d);
#else
            Compute_fillins_joint_traversal_group_wise_supernode_OptIII_warp_centric<<<BLKS_NUM,blockSize,0, stream[i]>>>  (dev_mem[i].cost_array_d,dev_mem[i].fill_in_d,dev_mem[i].frontier_d,
                    dev_mem[i].next_frontier_d,vert_count,dev_mem[i].csr_d,dev_mem[i].col_st_d,dev_mem[i].col_ed_d,dev_mem[i].fill_count_d,myGlobalrank,N_gpu,
                    dev_mem[i].src_frontier_d,dev_mem[i].next_src_frontier_d,dev_mem[i].source_d, dev_mem[i].frontier_size_d, dev_mem[i].next_frontier_size_d,
                    dev_mem[i].lock_d,N_groups,
                    dev_mem[i].dump_m,dev_mem[i].load_m,
                    N_src_group,group,dev_mem[i].max_id_offset,dev_mem[i].next_front_d,temp_allocated_frontier_size,
                    dev_mem[i].my_current_frontier_d,frontierchecked,dev_mem[i].swap_GPU_buffers_m,
                    dev_mem[i].nz_row_U_d, max_supernode_size,  N_small_chunks,dev_mem[i].source_flag_d, dev_mem[i].my_supernode_d, dev_mem[i].superObj_d,dev_mem[i].validity_supernode_d,
                    dev_mem[i].pass_through_d,dev_mem[i].fill_count_per_row_d, dev_mem[i].N_edge_checks_per_thread_d,new_col_ed_d,N_GPU_Node,i,dev_mem[i].group80_count_d, dev_mem[i].TEPS_value_perthread_d,valid_index);
#endif



        }




        // SynchronizeAllDevices(stream,N_GPU_Node);

        // if (source_d[thid] == 4096) printf(" nz_row_U_d[4096]: %d\n", nz_row_U_d[intranode_offset]); 

        //////////Make the max_id_offset and related variable per GPU so that Synchronize all devices is not required/////////////
        for (int i=0;i<N_GPU_Node;i++)
        {          
            dev_mem[i].max_id_offset-=vert_count;
            dev_mem[i].count_group_loop++;
            #ifdef disable_maxID_optimization
            if (true)
            #else
            if (dev_mem[i].count_group_loop >= group_loops)
            #endif
                // if ((max_id_offset-vert_count) < 0)
            {
                dev_mem[i].max_id_offset=MAX_VAL-vert_count;           
                dev_mem[i].group_MAX_VAL = dev_mem[i].max_id_offset + vert_count;

#ifdef lambda
                H_ERR(cudaSetDevice(i+myrank*N_GPU_Node)); 
#else
                H_ERR(cudaSetDevice(i)); 
#endif
                Initialise_cost_array<<<128,128,0,stream[i]>>>(dev_mem[i].cost_array_d,vert_count,dev_mem[i].group_MAX_VAL,N_src_group);
                // Initialise_cost_array<<<128,128>>>(dev_mem[i].cost_array_d,vert_count,group_MAX_VAL,N_src_group);

                // H_ERR(cudaDeviceSynchronize());
                dev_mem[i].count_group_loop=0;
            }
            else
            {
                dev_mem[i].group_MAX_VAL = dev_mem[i].max_id_offset + vert_count;
            }
        }
        SynchronizeAllDevices(stream,N_GPU_Node);
#ifdef debug_fill_count
        ull_t fill_count_group = 0;
        for (int i=0;i<N_GPU_Node;i++)
        {
            ull_t indiv_gpu_fillcount =0 ;
#ifdef lambda
            H_ERR(cudaSetDevice(i+myrank*N_GPU_Node)); 
#else
            H_ERR(cudaSetDevice(i)); 
#endif
            H_ERR(cudaMemcpy(&indiv_gpu_fillcount,dev_mem[i].fill_count_d,sizeof(ull_t),cudaMemcpyDeviceToHost));
            fill_count_group += indiv_gpu_fillcount;
        }
#ifdef RDMA
        // cout<<"Process: "<<myrank<<" Fill count corresponding chunk: "<<head_pointer[myrank]<<" fill_count: "<< fill_count_group - last_fill_count<<endl;
#else
        // cout<<"Process: "<<myrank<<" Fill count corresponding chunk: "<<head_pointer[myrank]++<<" fill_count: "<< fill_count_group - last_fill_count<<endl;
#endif

#ifdef print_status
        // cout<<"Process: "<<myrank<<" Fill count corresponding group: "<<group<<" fill_count: "<< fill_count_group - last_fill_count<<endl;
#endif
        last_fill_count = fill_count_group;
#endif

        //******Check the supernodal expansion***********//
#ifdef detect_supernodes


        T3_parallel_detection_temp = wtime();




        for (int i=0;i<N_GPU_Node;i++)
        {
#ifdef lambda
            H_ERR(cudaSetDevice(i+myrank*N_GPU_Node)); 
#else
            H_ERR(cudaSetDevice(i)); 
#endif


            int myGlobalrank = myrank*N_GPU_Node + i;

            DetectSuperNodes_parallel_test (dev_mem,N_src_group, max_supernode_size, N_gpu,
                    myGlobalrank, dev_mem[i].superObj_d, vert_count,group, N_chunks, dev_mem[i].Nsup_per_chunk_d,dev_mem[i].Nsup_d,
                    dev_mem[i].source_d, dev_mem[i].my_supernode_d, new_col_ed_d,
                    dev_mem[i].col_ed_d, dev_mem[i].csr_d, dev_mem[i].col_st_d,stream[i],i, N_GPU_Node,dev_mem[i].count, dev_mem[i].pass_through_d,my_chunks,  num_curr_chunks_per_node);
            // SynchronizeAllDevices(stream,N_GPU_Node);

        }
        SynchronizeAllDevices(stream,N_GPU_Node);
        T3_parallel_detection += (wtime()-T3_parallel_detection_temp);
        // H_ERR(cudaMemcpy(&Nsup,Nsup_d,sizeof(int_t),cudaMemcpyDeviceToHost));
        // cout<<"#supernodes detected till now: "<<Nsup<<endl;
        // #endif
#endif

        // tail_pointer--;
#ifdef print_status
        // cout<<"Process: "<<myrank<<" Finished group: "<<group<<" Computed source range: ("<<starting_source<<" - "<<starting_source +  N_src_group * N_GPU_Node<<")"<<endl;
        // cout<<"Process: "<<myrank<<" Finished chunk: "<<head_pointer[myrank]<<endl;
#endif
        Total_N_head_pointer++;
        // head_pointer++;
#ifdef workload_stealing
        cout<<"Process: "<<myrank<<" time for chunk: "<<old_value+1<<" is: "<<(wtime()-start_chunk_time)*1000<<" ms\n";
#endif
        iteration++;
    }
    // tail_pointer =0;


    SynchronizeAllDevices(stream,N_GPU_Node);
    // cout<<"Process: "<<myrank<<" Finished static work! in time: "<<(wtime()-start_time)*1000<<" ms \n";
#ifdef workload_stealing
    //Stealing work from other working processes
    // int victim_process = (myrank+1)%num_process;
    int no_work = 0;//no_work is increased by 1 when a victim process
    int dest_process = myrank; 
    // int dest_process;
    // if (myrank == 4) 
    // {
    //     dest_process=  myrank + 1; 
    // }
    // else
    // {
    //     dest_process = 4;
    // }
    // int dest_process = myrank + 1; 
    bool move_next_process = false;
    // bool break_loop =false;
    while (no_work < num_process-1) 
        // while (no_work < 1) 
    {
        // struct request_package send_request;
        // struct request_package recv_request;
        int group=0;
#ifdef profile_stealing_overhead
        double polling_start_time = wtime();
#endif

#ifdef profile_stealing_overhead
        time_response_polling+=(wtime()-polling_start_time);
#endif
        //Check if the victim process (dest_process) itself is requesting work from other process

        //~Check if the victim process (dest_process) itself is requesting work from other process
        MPI_Status status;
        // send_request.thief_id = myrank;
        // dest_process = (dest_process +1) % num_process;


        dest_process = (myrank + 1 + no_work) % num_process;

#ifdef print_status
        // printf("MPI process %d sends stealing request to process:%d with thief id = %d myRemaingin bug_chunks:%d \n", myrank, dest_process, send_request.thief_id,tail_pointer);
#endif
        // int target_displacement =  sizeof(int); //Displacement from start of window to beginning of target buffer
        ////////////
        int loc=0;

        double start_time_stealing_request = wtime();

        MPI_Win_lock(MPI_LOCK_EXCLUSIVE,dest_process,0,win_processed);	
        int result=1;

        MPI_Fetch_and_op(&result, &head_pointer[dest_process],
                MPI_INT, dest_process, //target rank =1
                dest_process, MPI_SUM, win_processed); 



        MPI_Win_unlock(dest_process, win_processed);            
        // MPI_Win_flush_all(win);
        // MPI_Win_flush_local(myrank, win);
        int stolen_chunk_id = head_pointer[dest_process] +1;

        cout<<"Process: "<<myrank<<" Received chunk_id: "<<stolen_chunk_id<<" from process: "<<dest_process<<endl;
        // if (tail_pointer[dest_process] > dest_processed)
        if (stolen_chunk_id < total_big_chunks_per_node)
        {
            cout<<"Process: "<<myrank<<" is processing chunk: "<<stolen_chunk_id<<" of Process: "<<dest_process<<endl;
        }
        else
        {
            cout<<"Process: "<<myrank<<" skipped the received chunk and Moving to the next process!"<<endl;
            move_next_process = true;
        }
        // MPI_Win_unlock(dest_process, win_processed);
        // MPI_Win_unlock(dest_process,win);
        time_stealing_request += (wtime()-start_time_stealing_request);
        if (move_next_process) 
        {
            ctrl_recvd++;//Received "No Work" message
#ifdef print_status
            // cout<<"Process "<<myrank<<" Moving to the next process for work request!"<<endl;
#endif
            no_work++;// move to the next process when working on multiple processes

            move_next_process = false;
            continue;
        }
        // int stolen_chunk_id = tail_pointer[dest_process];


        // if (recv_response.stolen_chunk_id !=-1)
        if (stolen_chunk_id >= 0)
        {
            Total_N_head_pointer++;
            chunks_recvd++;
            int victim_rank =  dest_process;
            //Process the stolen chunk
            int stolen_group_id = get_group_id(N_gpu, stolen_chunk_id);
            // ProcessStolenChunk(stolen_group_id, victim_rank);
            double start_chunk_time = wtime();
            // cout<<"Process: "<<myrank<<" stole chunk: "<<stolen_chunk_id<<" of process: "<<dest_process<<endl;
            double start_initialization_overhead = wtime();
            for (int i=0;i<N_GPU_Node;i++)
            {          
                // H_ERR(cudaMemset( dev_mem[i].fill_in_d, 0, sizeof(int_t)*N_src_group*vert_count));
                dev_mem[i].max_id_offset-=vert_count;
                dev_mem[i].count_group_loop++;
                if (dev_mem[i].count_group_loop >= group_loops)
                    // if (true)
                {
                    dev_mem[i].max_id_offset=MAX_VAL-vert_count;           
                    dev_mem[i].group_MAX_VAL = dev_mem[i].max_id_offset + vert_count;

#ifdef lambda
                    H_ERR(cudaSetDevice(i+myrank*N_GPU_Node)); 
#else
                    H_ERR(cudaSetDevice(i)); 
#endif
                    Initialise_cost_array<<<128,128,0,stream[i]>>>(dev_mem[i].cost_array_d,vert_count,dev_mem[i].group_MAX_VAL,N_src_group);
                    // Initialise_cost_array<<<128,128>>>(dev_mem[i].cost_array_d,vert_count,group_MAX_VAL,N_src_group);

                    // H_ERR(cudaDeviceSynchronize());
                    dev_mem[i].count_group_loop=0;
                }
                else
                {
                    dev_mem[i].group_MAX_VAL = dev_mem[i].max_id_offset + vert_count;
                }
            }
            SynchronizeAllDevices(stream,N_GPU_Node);
            initialization_overhead += (wtime()-start_initialization_overhead);


            // int starting_source = stolen_group_id*N_src_group;
            int node_offset = victim_rank * N_src_group * N_GPU_Node;
            int starting_source = node_offset + stolen_group_id*N_src_group;
#ifdef print_status
            // cout<<"Process (Thief): "<<myrank<<" Starting source: "<<starting_source<<" Ending source: "<< starting_source +  N_src_group * N_gpu << "stolen group: "<<stolen_group_id<<" victim process: "<<victim_rank<<" Skipped source range: ("<<starting_source<<" - "<<starting_source +  N_src_group * N_GPU_Node<<")"<<endl;
#endif


            for (int i=0;i<N_GPU_Node;i++)
            {
#ifdef lambda
                H_ERR(cudaSetDevice(i+ myrank * N_GPU_Node)); 
#else
                H_ERR(cudaSetDevice(i)); 
#endif
                dev_mem[i].swap_GPU_buffers_m[0]=0;
                int myGlobalrank = victim_rank * N_GPU_Node + i;
                // Compute_fillins_joint_traversal_group_wise_supernode_OptIII <<<BLKS_NUM,blockSize,0, stream[i]>>>  (dev_mem[i].cost_array_d,dev_mem[i].fill_in_d,dev_mem[i].frontier_d,
                //         dev_mem[i].next_frontier_d,vert_count,dev_mem[i].csr_d,dev_mem[i].col_st_d,dev_mem[i].col_ed_d,dev_mem[i].fill_count_d,myGlobalrank,N_gpu,
                //         dev_mem[i].src_frontier_d,dev_mem[i].next_src_frontier_d,dev_mem[i].source_d, dev_mem[i].frontier_size_d, dev_mem[i].next_frontier_size_d,
                //         dev_mem[i].lock_d,N_groups,
                //         dev_mem[i].dump_m,dev_mem[i].load_m,
                //         N_src_group, stolen_group_id, dev_mem[i].max_id_offset,dev_mem[i].next_front_d,temp_allocated_frontier_size,
                //         dev_mem[i].my_current_frontier_d,frontierchecked,dev_mem[i].swap_GPU_buffers_m,
                //         dev_mem[i].nz_row_U_d, max_supernode_size,  N_small_chunks,dev_mem[i].source_flag_d, dev_mem[i].my_supernode_d, dev_mem[i].superObj_d,dev_mem[i].validity_supernode_d,
                //         dev_mem[i].pass_through_d,dev_mem[i].fill_count_per_row_d, dev_mem[i].N_edge_checks_per_thread_d,new_col_ed_d,N_GPU_Node,i, dev_mem[i].group80_count_d);
                int_t* fill_in_d_temp;
                switch(victim_rank)
                {
                    case 0:
                        fill_in_d_temp = dev_mem[i].fill_in_d_P0;
                        break;
                    case 1:
                        fill_in_d_temp = dev_mem[i].fill_in_d_P1;
                        break;
                    case 2:
                        fill_in_d_temp = dev_mem[i].fill_in_d_P2;
                        break;
                    case 3:
                        fill_in_d_temp = dev_mem[i].fill_in_d_P3;
                        break;
                    case 4:
                        fill_in_d_temp = dev_mem[i].fill_in_d_P4;
                        break;
                    case 5:
                        fill_in_d_temp = dev_mem[i].fill_in_d_P5;
                        break;
                }

                Compute_fillins_joint_traversal_group_wise_supernode_OptIII <<<BLKS_NUM,blockSize,0, stream[i]>>>  (dev_mem[i].cost_array_d,fill_in_d_temp,dev_mem[i].frontier_d,
                        dev_mem[i].next_frontier_d,vert_count,dev_mem[i].csr_d,dev_mem[i].col_st_d,dev_mem[i].col_ed_d,dev_mem[i].fill_count_d,myGlobalrank,N_gpu,
                        dev_mem[i].src_frontier_d,dev_mem[i].next_src_frontier_d,dev_mem[i].source_d, dev_mem[i].frontier_size_d, dev_mem[i].next_frontier_size_d,
                        dev_mem[i].lock_d,N_groups,
                        dev_mem[i].dump_m,dev_mem[i].load_m,
                        N_src_group, stolen_group_id, dev_mem[i].max_id_offset,dev_mem[i].next_front_d,temp_allocated_frontier_size,
                        dev_mem[i].my_current_frontier_d,frontierchecked,dev_mem[i].swap_GPU_buffers_m,
                        dev_mem[i].nz_row_U_d, max_supernode_size,  N_small_chunks,dev_mem[i].source_flag_d, dev_mem[i].my_supernode_d, dev_mem[i].superObj_d,dev_mem[i].validity_supernode_d,
                        dev_mem[i].pass_through_d,dev_mem[i].fill_count_per_row_d, dev_mem[i].N_edge_checks_per_thread_d,new_col_ed_d,N_GPU_Node,i, dev_mem[i].group80_count_d);

            }


            SynchronizeAllDevices(stream,N_GPU_Node);

#ifdef debug_fill_count
            ull_t fill_count_group = 0;

            for (int i=0;i<N_GPU_Node;i++)
            {
                ull_t indiv_gpu_fillcount =0 ;
#ifdef lambda
                H_ERR(cudaSetDevice(i+myrank*N_GPU_Node)); 
#else
                H_ERR(cudaSetDevice(i)); 
#endif
                H_ERR(cudaMemcpy(&indiv_gpu_fillcount,dev_mem[i].fill_count_d,sizeof(ull_t),cudaMemcpyDeviceToHost));
                // if (stolen_group_id==80)
                // {
                // 	ull_t group80_count;
                // 	H_ERR(cudaMemcpy(&group80_count,dev_mem[i].group80_count_d,sizeof(ull_t),cudaMemcpyDeviceToHost));
                // 	group_80_count_total += group80_count;
                // 	cout<<"group80_count: "<<group80_count<<endl;
                // }

                fill_count_group += indiv_gpu_fillcount;
            }
            // cout<<"Process: "<<myrank<<" (stolen) Fill count corresponding chunk: "<<stolen_chunk_id<<"from victim process: "<<dest_process<<" fill_count: "<< fill_count_group - last_fill_count<<endl;
#ifdef print_status
            // cout<<"Process (stolen): "<<myrank<<" Fill count corresponding (stolen) group: "<<stolen_group_id<<" fill_count: "<< fill_count_group - last_fill_count<<endl;
#endif
            last_fill_count = fill_count_group;
#endif

#ifdef detect_supernodes
            T3_parallel_detection_temp = wtime();
            for (int i=0;i<N_GPU_Node;i++)         
            {
#ifdef lambda
                H_ERR(cudaSetDevice(i+myrank*N_GPU_Node)); 
#else
                H_ERR(cudaSetDevice(i)); 
#endif
                int myGlobalrank = victim_rank*N_GPU_Node + i;
                DetectSuperNodes_parallel_test (dev_mem,N_src_group, max_supernode_size, N_gpu,
                        myGlobalrank, dev_mem[i].superObj_d, vert_count, stolen_group_id,  N_chunks, dev_mem[i].Nsup_per_chunk_d,dev_mem[i].Nsup_d,
                        dev_mem[i].source_d, dev_mem[i].my_supernode_d, new_col_ed_d,
                        dev_mem[i].col_ed_d, dev_mem[i].csr_d, dev_mem[i].col_st_d,stream[i],i, N_GPU_Node,dev_mem[i].count, dev_mem[i].pass_through_d,my_chunks,  num_curr_chunks_per_node);
            }
            SynchronizeAllDevices(stream,N_GPU_Node);
            T3_parallel_detection += (wtime()-T3_parallel_detection_temp);
#endif
#ifdef print_status
            // cout<<"Process: "<<myrank<<" Finished group (stolen): "<<stolen_group_id<<" stolen chunk_id: "<<stolen_chunk_id<<endl;
#endif
            cout<<"Process: "<<myrank<<" time for chunk: "<<stolen_chunk_id<<" of process: "<<dest_process <<" is:"<<(wtime()-start_chunk_time)*1000<<" ms\n";

        }
        else
        {    
            ctrl_recvd++;//Received "No Work" message
#ifdef print_status
            cout<<"Process "<<myrank<<" Moving to the next process for work request!"<<endl;
#endif
            no_work++;// move to the next process when working on multiple processes
        }
    }
    //~Stealing work from other working processes
#endif

    // #endif
#ifdef workload_stealing
    // MPI_Barrier(MPI_COMM_WORLD); 
#endif
    time +=((wtime()-start_time)*1000);
    cout<<"merge traversal complete!"<<endl;
    #ifdef profile_TEPS
    ull_t Total_EDGE_checks_node =0;
    ull_t* Temp = new ull_t[blockSize*BLKS_NUM];
    for (int i=0; i<N_GPU_Node; i++)
    {
        H_ERR(cudaSetDevice(i));
        H_ERR(cudaMemcpy(Temp,dev_mem[i].TEPS_value_perthread_d,blockSize*BLKS_NUM*sizeof(ull_t),cudaMemcpyDeviceToHost));
    // TEPS_value_perthread_d[original_thid]++;
    for (int j=0; j< blockSize*BLKS_NUM; j++)
    {
        Total_EDGE_checks_node += Temp[j];
    }
    }
    #endif
    // H_ERR(cudaMemcpy(source_flag,source_flag_d,sizeof(int_t)* vert_count,cudaMemcpyDeviceToHost));
    // Super* super_obj = new Super [vert_count];

    // H_ERR(cudaMemcpy(super_obj,superObj_d,sizeof(int_t)*Nsup,cudaMemcpyDeviceToHost));
    //  for (int_t i=0;i< Nsup; i++ )
    //  {
    //      cout<<i<<": super_obj["<<i<<"].start: "<< super_obj[i].start<<"   super_obj["<<i<<"].end: "<< super_obj[i].start<<endl;

    //  }
    int_t source_count =0;
    // H_ERR(cudaMemcpy(my_supernode,my_supernode_d,sizeof(int_t)*vert_count,cudaMemcpyDeviceToHost));
    // for (int_t i=0;i<vert_count;i++)
    // {
    //     if (source_flag[i] == 1) source_count++;
    //     cout<<i<<" my_supernode: "<< my_supernode[i]<<endl;
    // }
    nz_count_U_file.close();
    int_t N_supernode_detected=0;
    int_t pass_through_detected=0;
    for (int i=0;i<N_GPU_Node;i++)
    {
#ifdef lambda
        H_ERR(cudaSetDevice(i+myrank*N_GPU_Node)); 
#else
        H_ERR(cudaSetDevice(i)); 
#endif
        ull_t fill_count_temp;
        int_t temp_supernode = 0;
        int_t temp_pass_through_count=0;
        H_ERR(cudaMemcpy(&fill_count_temp,dev_mem[i].fill_count_d,sizeof(ull_t),cudaMemcpyDeviceToHost));
        H_ERR(cudaMemcpy(&temp_supernode,dev_mem[i].Nsup_d,sizeof(int_t),cudaMemcpyDeviceToHost));
        H_ERR(cudaMemcpy(&temp_pass_through_count,dev_mem[i].count,sizeof(int_t),cudaMemcpyDeviceToHost));
        // cout<<"Number of supernode detected by GPU: "<<i<<" :"<<temp_supernode<<endl;
        pass_through_detected += temp_pass_through_count;
        N_supernode_detected += temp_supernode;
        // H_ERR(cudaMemcpy(&fill_count_temp,dev_mem[i].fill_count_d,sizeof(ull_t),cudaMemcpyDeviceToHost));
        fill_count += fill_count_temp;
    }

    // DisplaySuperNodes( Nsup,superObj,N_non_zero_rows);
    // DisplayMySuperNodes( my_supernode, vert_count);
    // H_ERR(cudaMemcpy(&Nsup,Nsup_d,sizeof(int_t),cudaMemcpyDeviceToHost));
    N_supernodes = N_supernode_detected + 1;
    cout<<"Process: "<<myrank<<" Number of supernodes detected: "<<N_supernodes<<endl;
    // cout<<"Process: "<<myrank<<" Total pass through detected: "<<pass_through_detected<< endl;

    cout<<"Process: "<<myrank<<" Number of fill-ins detected: "<<fill_count<<endl;
 
   cout<<"Process: "<<myrank<<" time for fill-in detection: "<<time <<endl;
   #ifdef profile_TEPS
   cout<<"Process: "<<myrank<<" Total Taversed Edges: "<<Total_EDGE_checks_node <<endl;
   Process_TEPS = Total_EDGE_checks_node;
    // Process_TEPS = ;
   cout<<"Process: "<<myrank<<" TTEPS: "<< (Total_EDGE_checks_node/ (float)(time/(float)1000)) <<endl;    
   #endif
    cout<<"Process: "<<myrank<<" Time for allocations and stream creations: "<<allocation_time*1000 <<" ms"<<endl; 
    cout<<"Process: "<<myrank<<" Time for only stream creation: "<< streamcreate_time * 1000 <<" ms"<<endl;
    cout<<"Process: "<<myrank<<" Time for only supernodes detection: "<<T3_parallel_detection*1000<<" ms"<<endl;
    // cout<<"Total number of edge checks: "<< Total_edge_checks <<endl;
    // cout<<"N_dumping_cpu_memory: "<<N_dumping_cpu_memory<<endl;
    // cout<<"Process: "<<myrank<<" CPU time with dumping and loading time: "<<dumping_loading_time*1000<<" ms"<<endl;
    // cout<<"Process: "<<myrank<<" Only dumping and loading time: "<<time_only_Copy*1000<<" ms"<<endl;
    // cout<<"Process: "<<myrank<<" Achieved throughput of dumping and loading: "<< size_copied*sizeof(int)/((double)(1024*1024*1024))/(double)time_only_Copy<<" GB/s"<<endl;
    cout<<"Process: "<<myrank<<" time_response_polling: "<<time_response_polling*1000<<" ms"<<endl;
    cout<<"Process: "<<myrank<<" Total_N_head_pointer: "<<Total_N_head_pointer<<endl;
    cout<<"Process: "<<myrank<<" time_stealing_response: "<<time_stealing_response*1000<<" ms"<< endl;
    cout<<"Process: "<<myrank<<" time_stealing_request: "<<time_stealing_request*1000 <<" ms"<< endl;
    cout<<"Process: "<<myrank<<" initialization_overhead: "<<initialization_overhead*1000 <<" ms"<< endl;
    // if (myrank==0) cout<<" group_80_count_total: "<< group_80_count_total<<endl;
    return;
}

// 
