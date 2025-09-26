#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "syclcheck.cpp"
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "plrange.dp.hpp"

#ifdef DEBUG_PRINT
const sycl::property_list prop_list = sycl::property_list{sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()};
#else
const sycl::property_list prop_list = sycl::property_list{sycl::property::queue::in_order()};
#endif

/* 

CUDA/HIP kernel for range selection using forward chaining

*/

/* kernels begin */

int *d_vec;

inline int64_t range_binary_search(const int32_t* ax, const int32_t* rev, int64_t i, int64_t st_end,
                                   int d_max_dist_x){
    int64_t st_high = st_end, st_low=i;
    while (st_high != st_low) {
        int64_t mid = (st_high + st_low -1) / 2+1;
        if (rev[i] != rev[mid] || ax[mid] > ax[i] + d_max_dist_x) {
            st_high = mid -1;
        } else {
            st_low = mid;
        }
    }
    return st_high;
}


/**
 * Forward Range Selection Kernel using global memory and binary range search. 
 * cut reads into segements where successor range = 0. 
*/
void range_selection_kernel_binary(const int32_t* ax, const int32_t* rev, size_t *start_idx_arr, size_t *read_end_idx_arr, 
    int32_t *range, size_t* cut, size_t* cut_start_idx, size_t total_n, size_t anchor_per_block,
    int d_max_dist_x, int d_max_iter, int d_cut_check_anchors){
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int tid = item_ct1.get_local_id(2);
    int bid = item_ct1.get_group(2);

    size_t start_idx = start_idx_arr[bid];
    size_t read_end_idx = read_end_idx_arr[bid];
    size_t end_idx = start_idx + anchor_per_block;
    end_idx = end_idx > read_end_idx ? read_end_idx : end_idx;
    size_t cut_idx = cut_start_idx[bid];
    if(tid == 0 && (bid == 0 || read_end_idx_arr[bid-1] != read_end_idx)){
        cut[cut_idx] = start_idx;
    }
    cut_idx++;
    int range_op[3] = {16, 512, 5000};  // Range Options
    range_op[2] = d_max_iter;
    for (size_t i = start_idx + tid; i < end_idx;
         i += item_ct1.get_local_range(2)) {
        size_t st_max = i + d_max_iter;
        st_max = st_max < read_end_idx ? st_max : read_end_idx -1;
        size_t st;
        for (int j=0; j<3; ++j){
            st = i + range_op[j];
            st = st <= st_max ? st : st_max;
            assert(st < total_n);
            assert(i < total_n);
            if (st > i && (rev[st] != rev[i] || ax[st] > ax[i] + d_max_dist_x)){
                break;
            }
        }
        st = range_binary_search(ax, rev, i, st, d_max_dist_x);
        range[i] = st - i;

        if (tid >= item_ct1.get_local_range(2) - d_cut_check_anchors &&
            item_ct1.get_local_range(2) - tid + i <= end_idx) {
            if (st == i) cut[cut_idx] = i+1;
        }
        cut_idx++;
    }
}

/**
 * Forward Range Selection Kernel using global memory and linear range search.
 * cut reads into segements where successor range = 0.
 */
void range_selection_kernel_naive(const int32_t* ax, const int32_t* rev, size_t *start_idx_arr, size_t *read_end_idx_arr, 
    int32_t *range, size_t* cut, size_t* cut_start_idx, size_t total_n, range_kernel_config_t config,
    int d_max_dist_x, int d_max_iter, int d_cut_check_anchors){
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int tid = item_ct1.get_local_id(2);
    int bid = item_ct1.get_group(2);

    size_t start_idx = start_idx_arr[bid];
    size_t read_end_idx = read_end_idx_arr[bid];
    size_t end_idx = start_idx + config.anchor_per_block;
    end_idx = end_idx > read_end_idx ? read_end_idx : end_idx;
    assert(end_idx == (bid + 1 <
                       sycl::ext::oneapi::this_work_item::get_nd_item<3>()
                           .get_group_range(2))
               ? start_idx_arr[bid + 1]
               : total_n);
    // if(end_idx_ref != end_idx){
    //     if (tid == 0){
    //         int grimdim = gridDim.x;
    //     printf("start idx %d anchor_per_block %d read_end_idx %d, next start idx %d gridDim %d bid %d\n", 
    //     start_idx, config.anchor_per_block, read_end_idx, start_idx_arr[bid+1], grimdim, bid);
    //     }
    // }
    // __syncthreads();
    
    size_t cut_idx = cut_start_idx[bid];
    if(tid == 0 && (bid == 0 || read_end_idx_arr[bid-1] != read_end_idx)){
        cut[cut_idx] = start_idx;
    }
    cut_idx++;
    for (size_t i = start_idx + tid; i < end_idx;
         i += item_ct1.get_local_range(2)) {
        size_t st = i + d_max_iter;
        st = i + d_max_iter < read_end_idx ? st : read_end_idx -1;
        assert(st < total_n);
        assert(i < total_n);
        while (st > i && 
                (rev[i] != rev[st] // NOTE: different prefix cannot become predecessor 
                || ax[st] > ax[i] + d_max_dist_x)) { // NOTE: same prefix compare the value
            --st;
        }
        range[i] = st - i;

        if (tid >= item_ct1.get_local_range(2) - d_cut_check_anchors &&
            item_ct1.get_local_range(2) - tid + i <= end_idx) {
            if (st == i) cut[cut_idx] = i+1;
        }
        cut_idx++;
    }
}

// __global__ void range_selection_kernel(const int64_t* ax, size_t *start_idx_arr, size_t *read_end_idx_arr, int32_t *range){
//     int tid = threadIdx.x;
//     int bid = blockIdx.x;

//     size_t start_idx = start_idx_arr[bid];
//     size_t read_end_idx = read_end_idx_arr[bid];
//     size_t end_idx = start_idx + MAX_ANCHOR_PER_BLOCK;
//     end_idx = end_idx > read_end_idx ? read_end_idx : end_idx;

//     size_t load_anchor_idx = 100;
//     size_t load_smem_idx;
//     size_t cal_idx = start_idx + threadIdx.x;
//     int32_t cal_smem = tid;
//     __shared__ int64_t smem[NUM_ANCHOR_IN_SMEM];

//     /* prefetch anchors */
//     load_smem_idx = tid;
//     load_anchor_idx = start_idx + tid;
//     // if (tid == 20) printf("load_smem_idx %d, load_anchor_idx %lu\n", load_smem_idx, load_anchor_idx);
//     for (int i = 0; i < PREFETCH_ANCHORS_RANGE/NUM_THREADS_RANGE && load_anchor_idx < read_end_idx; ++i){
//         // if (tid == 20) printf("load_smem_idx %d, load_anchor_idx %lu\n", load_smem_idx, load_anchor_idx);
//         smem[load_smem_idx] = ax[load_anchor_idx];
//         load_smem_idx += NUM_THREADS_RANGE;
//         load_anchor_idx += NUM_THREADS_RANGE;
//     }

//     int iter = (NUM_ANCHOR_IN_SMEM - PREFETCH_ANCHORS_RANGE)/NUM_THREADS_RANGE; // iterations before another load is needed
//     while (cal_idx < end_idx) { // tail threads may skip this loop
//         /* load anchors */
//         load_smem_idx = load_smem_idx >= NUM_ANCHOR_IN_SMEM ? load_smem_idx - NUM_ANCHOR_IN_SMEM : load_smem_idx;
//         for (int i = 0; i < iter && load_anchor_idx < end_idx + PREFETCH_ANCHORS_RANGE; ++i){
//             // if (tid == 20) printf("load it load_smem_idx %d, load_anchor_idx %lu\n", load_smem_idx, load_anchor_idx);
//             smem[load_smem_idx] = ax[load_anchor_idx];
//             load_smem_idx += NUM_THREADS_RANGE;
//             load_anchor_idx += NUM_THREADS_RANGE;
//             load_smem_idx = load_smem_idx >= NUM_ANCHOR_IN_SMEM ? load_smem_idx - NUM_ANCHOR_IN_SMEM : load_smem_idx;
//         }

//         __syncthreads();
        
//         /* calculate sucessor range */
//         for (int i = 0; i < iter && cal_idx < end_idx; ++i){
//             int64_t anchor = smem[cal_smem];

//             size_t st = cal_idx + PREFETCH_ANCHORS_RANGE < read_end_idx ? cal_idx + PREFETCH_ANCHORS_RANGE : read_end_idx-1;
//             int32_t st_smem = cal_smem + st - cal_idx;
//             st_smem = st_smem >= NUM_ANCHOR_IN_SMEM ? st_smem - NUM_ANCHOR_IN_SMEM : st_smem;
//             // if (tid == 20) printf("cal idx %lu, cal_mem %d, st %lu, st_smem %d\n", cal_idx, cal_smem, st,st_smem);

//             // if (tid == 20) printf("anchor.x %d, smem[st_smem] %d, anchor.x+MAX_DIST_X%d\n", anchor, smem[st_smem], anchor+MAX_DIST_X);

//             while (st > cal_idx && 
//                         (anchor>> 32 != smem[st_smem] >> 32 ||
//                             smem[st_smem] > anchor + d_max_dist_x
//                         )
//                     ){
//                 // if (bid == 25)
//                 // printf("while 0 bid %d tid %d cal_idx %d\n", bid, tid, cal_idx);
//                 --st;
//                 if (st_smem == 0) st_smem = NUM_ANCHOR_IN_SMEM-1;
//                 else --st_smem;
//             }
            
//             /* NOTE: fallback: succussor is not prefetched */
//             if (st >= PREFETCH_ANCHORS_RANGE + cal_idx){
//                 st = cal_idx + MAX_ITER < read_end_idx ? i + MAX_ITER : read_end_idx-1;
//                 while(
//                     anchor >> 32 != ax[st] >> 32 || 
//                     ax[st] > anchor + d_max_dist_x // check from global memory
//                 ){
//                     --st;
//                     // if (bid == 25)
//                     // printf("while 1 bid %d tid %d\n", bid, tid);
//                 }

//             }
//             range[cal_idx] = st - cal_idx;
//             cal_smem += NUM_THREADS_RANGE;
//             cal_smem = cal_smem >= NUM_ANCHOR_IN_SMEM ? cal_smem - NUM_ANCHOR_IN_SMEM : cal_smem;
//             cal_idx += NUM_THREADS_RANGE;
//             // if (bid == 25)
//             // printf("for loop i %d bid %d tid %d\n", i, bid, tid);
//         }
//         // if (bid == 25)
//         // printf("outer while bid %d tid %d\n", bid, tid);
//         __syncthreads();

//     }
    
// }

/* kernels end */

#ifdef __cplusplus
extern "C" {
#endif

/* host functions begin */
range_kernel_config_t range_kernel_config;

void plrange_upload_misc(Misc misc) try {
  sycl::queue q(prop_list);
  sycl::queue &q_ct1 = q; 

  d_vec = sycl::malloc_device<int>(3, q_ct1);

  q_ct1.memcpy(&d_vec[0], &misc.max_dist_x, sizeof(int));
  q_ct1.memcpy(&d_vec[1], &misc.max_iter, sizeof(int));
  q_ct1.memcpy(&d_vec[2], &range_kernel_config.cut_check_anchors, sizeof(int));
  q_ct1.wait_and_throw();
} catch (const sycl::exception &e) {
  fprintf(stderr, "Error in %s:%i %s(): %s.\n", __FILE__, __LINE__, __func__, e.what());
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void plrange_free_misc() try {
  sycl::queue q(prop_list);
  sycl::queue &q_ct1 = q; 

  sycl::free(d_vec, q_ct1);
  q_ct1.wait_and_throw();
} catch (const sycl::exception &e) {
  fprintf(stderr, "Error in %s:%i %s(): %s.\n", __FILE__, __LINE__, __func__, e.what());
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void plrange_async_range_selection(deviceMemPtr *dev_mem,
                                   sycl::queue **stream,
                                   sycl::event *start_event_short) {
    size_t total_n = dev_mem->total_n, cut_num = dev_mem->num_cut;
    int griddim = dev_mem->griddim;
    sycl::range<3> DimBlock(range_kernel_config.blockdim, 1, 1);
    sycl::range<3> DimGrid(griddim, 1, 1);

    // Run kernel
    /*
    DPCT1049:0: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
  {
    *start_event_short = (*stream)->submit([&](sycl::handler &cgh) {
      auto d_max_dist_x_ptr_ct1 = &d_vec[0];
      auto d_max_iter_ptr_ct1 = &d_vec[1];
      auto d_cut_check_anchors_ptr_ct1 = &d_vec[2];

      const int32_t *dev_mem_d_ax_ct0 = dev_mem->d_ax;
      const int32_t *dev_mem_d_xrev_ct1 = dev_mem->d_xrev;
      auto dev_mem_d_start_idx_ct2 = dev_mem->d_start_idx;
      auto dev_mem_d_read_end_idx_ct3 = dev_mem->d_read_end_idx;
      auto dev_mem_d_range_ct4 = dev_mem->d_range;
      auto dev_mem_d_cut_ct5 = dev_mem->d_cut;
      auto dev_mem_d_cut_start_idx_ct6 = dev_mem->d_cut_start_idx;

      auto tmp_anchors_per_block = range_kernel_config.anchor_per_block;

      cgh.parallel_for(
          sycl::nd_range<3>(DimGrid * DimBlock, DimBlock),
          [=](sycl::nd_item<3> item_ct1) {
            range_selection_kernel_binary(
                dev_mem_d_ax_ct0, dev_mem_d_xrev_ct1, dev_mem_d_start_idx_ct2,
                dev_mem_d_read_end_idx_ct3, dev_mem_d_range_ct4,
                dev_mem_d_cut_ct5, dev_mem_d_cut_start_idx_ct6, total_n,
                tmp_anchors_per_block, *d_max_dist_x_ptr_ct1, *d_max_iter_ptr_ct1,
                *d_cut_check_anchors_ptr_ct1);
          });
    });
  }
  sycl_check(**stream);
#ifdef DEBUG_PRINT
    // fprintf(stderr, "[Info] %s (%s:%d): Batch total_n %lu, Range Kernel Launched, grid %d cut %d\n", __func__, __FILE__, __LINE__, total_n, DimGrid.x, cut_num);
#endif
}

void plrange_sync_range_selection(deviceMemPtr *dev_mem, Misc misc) {
    size_t total_n = dev_mem->total_n, cut_num = dev_mem->num_cut;
    int griddim = dev_mem->griddim;
    sycl::range<3> DimBlock(range_kernel_config.blockdim, 1, 1);
    sycl::range<3> DimGrid(griddim, 1, 1);

    plrange_upload_misc(misc);

    // Run kernel
#ifdef DEBUG_PRINT
        fprintf(stderr, "[Info] %s (%s:%d): Grim Dim: %zu Cut: %zu Anchors: %zu\n", __func__, __FILE__, __LINE__, DimGrid[2],
                cut_num, total_n);
#endif
    /*
    DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
  {
    dpct::get_in_order_queue().submit([&](sycl::handler &cgh) {
      auto d_max_dist_x_ptr_ct1 = &d_vec[0];
      auto d_max_iter_ptr_ct1 = &d_vec[1];
      auto d_cut_check_anchors_ptr_ct1 = &d_vec[2];

      const int32_t *dev_mem_d_ax_ct0 = dev_mem->d_ax;
      const int32_t *dev_mem_d_xrev_ct1 = dev_mem->d_xrev;
      auto dev_mem_d_start_idx_ct2 = dev_mem->d_start_idx;
      auto dev_mem_d_read_end_idx_ct3 = dev_mem->d_read_end_idx;
      auto dev_mem_d_range_ct4 = dev_mem->d_range;
      auto dev_mem_d_cut_ct5 = dev_mem->d_cut;
      auto dev_mem_d_cut_start_idx_ct6 = dev_mem->d_cut_start_idx;

      auto tmp_anchors_per_block = range_kernel_config.anchor_per_block;

      cgh.parallel_for(
          sycl::nd_range<3>(DimGrid * DimBlock, DimBlock),
          [=](sycl::nd_item<3> item_ct1) {
            range_selection_kernel_binary(
                dev_mem_d_ax_ct0, dev_mem_d_xrev_ct1, dev_mem_d_start_idx_ct2,
                dev_mem_d_read_end_idx_ct3, dev_mem_d_range_ct4,
                dev_mem_d_cut_ct5, dev_mem_d_cut_start_idx_ct6, total_n,
                tmp_anchors_per_block, *d_max_dist_x_ptr_ct1, *d_max_iter_ptr_ct1,
                *d_cut_check_anchors_ptr_ct1);
          });
    });
  }

  try {
    dpct::get_current_device().queues_wait_and_throw();
  } catch (const sycl::exception &e) {
    fprintf(stderr, "Error in %s:%i %s(): %s.\n", __FILE__, __LINE__, __func__, e.what());
    fflush(stderr);
    exit(EXIT_FAILURE);
  }

#ifdef DEBUG_PRINT
    fprintf(stderr, "[Info] %s: range calculation success\n", __func__);
#endif
}

#ifdef __cplusplus
}
#endif

/* host functions end */
