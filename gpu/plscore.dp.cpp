#ifdef DEBUG_PRINT
#define DPCT_PROFILING_ENABLED
#endif

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "syclcheck.cpp"
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "plscore.dp.hpp"

/* 

Parallel chaining helper functions with CUDA

*/

inline dpct::constant_memory<Misc, 0> misc;
inline dpct::constant_memory<int, 0> long_seg_cutoff;
inline dpct::constant_memory<int, 0> mid_seg_cutoff;
inline dpct::global_memory<unsigned, 0> curr_long_segid;

/* arithmetic functions begin */

// __device__ static inline float cuda_mg_log2(float x) // NB: this doesn't work when x<2
// {
// 	union { float f; uint32_t i; } z = { x };
// 	float log_2 = ((z.i >> 23) & 255) - 128;
// 	z.i &= ~(255 << 23);
// 	z.i += 127 << 23;
// 	log_2 += (-0.34484843f * z.f + 2.02466578f) * z.f - 0.67487759f;
// 	return log_2;
// }

static inline float cuda_mg_log2(int32_t x) // NB: this doesn't work when x<2
{
    return 31 - sycl::clz((int)x);
}

int32_t original_comput_sc(const int32_t ai_x, const int32_t ai_y, const int32_t aj_x, const int32_t aj_y,
                                const int8_t sidi,  const int8_t sidj,
                                int32_t max_dist_x, int32_t max_dist_y,
                                int32_t bw, float chn_pen_gap,
                                float chn_pen_skip, int is_cdna, int n_seg) {
    int32_t dq = ai_y - aj_y, dr, dd, dg, q_span, sc;
    if (dq <= 0 || dq > max_dist_x) return INT32_MIN;
    dr = ai_x - aj_x;
    if (sidi == sidj && (dr == 0 || dq > max_dist_y)) return INT32_MIN;
    dd = dr > dq ? dr - dq : dq - dr;
    if (sidi == sidj && dd > bw) return INT32_MIN;
    if (n_seg > 1 && !is_cdna && sidi == sidj && dr > max_dist_y)
        return INT32_MIN;  // nseg = 1 by default
    dg = dr < dq ? dr : dq;
    q_span = MM_QSPAN;
    sc = q_span < dg ? q_span : dg;
    if (dd || dg > q_span) {
        float lin_pen, log_pen;
        lin_pen = chn_pen_gap * (float)dd + chn_pen_skip * (float)dg;
        log_pen =
            dd >= 1 ? cuda_mg_log2(dd + 1) : 0.0f;  // mg_log2() only works for dd>=2
        if (is_cdna || sidi != sidj) {
            if (sidi != sidj && dr == 0)
                ++sc;  // possibly due to overlapping paired ends; give a minor
                       // bonus
            else if (dr > dq || sidi != sidj)
                sc -=
                    (int)(lin_pen < log_pen ? lin_pen
                                            : log_pen);  // deletion or jump
                                                         // between paired ends
            else
                sc -= (int)(lin_pen + .5f * log_pen);
        } else
            sc -= (int)(lin_pen + .5f * log_pen);
    }
    return sc;
}

inline int32_t comput_sc(const int32_t ai_x, const int32_t ai_y, const int32_t aj_x, const int32_t aj_y,
                                const int32_t sidi,  const int32_t sidj,
                                const int32_t max_dist_x, const int32_t max_dist_y,
                                const int32_t bw, const float chn_pen_gap,
                                const float chn_pen_skip, const int is_cdna, const int n_seg) {
    const bool is_same_sid = sidi == sidj;
    const int32_t dq = ai_y - aj_y, dr = ai_x - aj_x;
    /*
    DPCT1017:16: The sycl::abs_diff call is used instead of the __sad call.
    These two calls do not provide exactly the same functionality. Check the
    potential precision and/or performance issues for the generated code.
    */
    const int32_t dd = sycl::abs_diff(dr, dq) + 0;

    if (dq <= 0 || dq > max_dist_x ||
        (is_same_sid && (dr == 0 || 
                        dq > max_dist_y || 
                        dd > bw || 
                        (n_seg > 1 && !is_cdna && dr > max_dist_y))))
        return INT32_MIN;

    const int32_t dg = dr < dq ? dr : dq;
    int32_t sc = MM_QSPAN < dg ? MM_QSPAN : dg;
    
    if (dd || dg > MM_QSPAN) {
        int32_t log_pen = dd >= 1 ? (31 - sycl::clz(dd + 1)) : 0;
        int32_t lin_pen = chn_pen_gap * (float)dd + chn_pen_skip * (float)dg;
        // Initial conditions for modifying score based on penalties
        bool minorBonus = is_cdna && !is_same_sid && dr == 0;
        bool majorAdjustment = (is_cdna && dg == dq) || !is_same_sid;
        sc += minorBonus;
        sc -= (!minorBonus && majorAdjustment) * (int)(lin_pen < log_pen ? lin_pen : log_pen);
        sc -= (!minorBonus && !majorAdjustment) * (int)(lin_pen + 0.5f * log_pen);
    }
    return sc;
}


/* arithmetic functions end */

inline void compute_sc_seg_one_wf(const int32_t* anchors_x, const int32_t* anchors_y, const int8_t* sid, const int32_t* range, 
                    const size_t start_idx, const size_t end_idx,
                    int32_t* f, uint16_t* p
, Misc misc){
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const Misc blk_misc = misc;
    int tid = item_ct1.get_local_id(2);
    // init f and p
    for (size_t i = start_idx + tid; i < end_idx;
         i += item_ct1.get_local_range(2)) {
        f[i] = MM_QSPAN;
        p[i] = 0;
    }
        sycl::group_barrier(
            sycl::ext::oneapi::this_work_item::
                get_sub_group()); // NOTE: single warp, no need to sync
    for (size_t i=start_idx; i < end_idx; i++) {
        int32_t range_i = range[i];
        for (int32_t j = tid; j < range_i; j += item_ct1.get_local_range(2)) {
            int32_t sc = comput_sc(
                                anchors_x[i+j+1], 
                                anchors_y[i+j+1], 
                                anchors_x[i], 
                                anchors_y[i],
                                sid [i+j+1],
                                sid [i],
                                blk_misc.max_dist_x, blk_misc.max_dist_y, blk_misc.bw, blk_misc.chn_pen_gap, 
                                blk_misc.chn_pen_skip, blk_misc.is_cdna, blk_misc.n_seg);
            if (sc == INT32_MIN) continue;
            sc += f[i];
            if (sc >= f[i+j+1] && sc != MM_QSPAN) {
                f[i+j+1] = sc;
                p[i+j+1] = j+1;

            }
        }
        sycl::group_barrier(
            sycl::ext::oneapi::this_work_item::
                get_sub_group()); // NOTE: single warp, no need to sync
    }
    
}


inline void compute_sc_seg_multi_wf(const int32_t* anchors_x, const int32_t* anchors_y, const int8_t* sid, const int32_t* range, 
                    const size_t start_idx, const size_t end_idx,
                    int32_t* f, uint16_t* p
, Misc misc){
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const Misc blk_misc = misc;
    int tid = item_ct1.get_local_id(2);
    int bid = item_ct1.get_group(2);
    // init f and p
    for (size_t i = start_idx + tid; i < end_idx;
         i += item_ct1.get_local_range(2)) {
        f[i] = MM_QSPAN;
        p[i] = 0;
    }
    /*
    DPCT1065:17: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    for (size_t i=start_idx; i < end_idx; i++) {
        int32_t range_i = range[i];
        for (int32_t j = tid; j < range_i; j += item_ct1.get_local_range(2)) {
            int32_t sc = comput_sc(
                                anchors_x[i+j+1], 
                                anchors_y[i+j+1], 
                                anchors_x[i], 
                                anchors_y[i],
                                sid [i+j+1],
                                sid [i],
                                blk_misc.max_dist_x, blk_misc.max_dist_y, blk_misc.bw, blk_misc.chn_pen_gap, 
                                blk_misc.chn_pen_skip, blk_misc.is_cdna, blk_misc.n_seg);
            if (sc == INT32_MIN) continue;
            sc += f[i];
            if (sc >= f[i+j+1] && sc != MM_QSPAN) {
                f[i+j+1] = sc;
                p[i+j+1] = j+1;

            }
        }
        /*
        DPCT1118:2: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:18: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
    
}

#define NUM_ANCHORS_PREFETCH 1024

// inline __device__ void compute_sc_seg_shared(const int64_t* anchors_x, const int64_t* anchors_y, int32_t* range, 
//                     size_t start_idx, size_t end_idx,
//                     int32_t* f, uint16_t* p
// ){
//     Misc blk_misc = misc;
//     int tid = threadIdx.x;
//     int bid = blockIdx.x;
//     // init f and p
//     for (size_t i=start_idx+tid; i < end_idx; i += blockDim.x) {
//         f[i] = anchors_y[i] >> 32 & 0xff;
//         p[i] = 0;
//     }
//     __syncthreads();
//     // assert(range[end_idx-1] == 0);
//     __shared__ int64_t anchors_x_shared[NUM_ANCHORS_PREFETCH];

//     __shared__ int64_t anchors_y_shared[NUM_ANCHORS_PREFETCH];
//     size_t prefetch_end_idx = 0;
//     unsigned int prefetch_smem_offset = 0;
//     for (size_t i = start_idx; i < end_idx; i++) {
//         int32_t range_i = range[i];
//         // if (range_i + i >= end_idx)
//         //     printf("range_i %d i %lu start_idx %lu, end_idx %lu\n", range_i, i, start_idx, end_idx);
//         // assert(range_i + i < end_idx);
//         for (int32_t j = tid; j < range_i; j += blockDim.x) {
//             int32_t sc = comput_sc(
//                                 anchors_x[i+j+1], 
//                                 anchors_y[i+j+1], 
//                                 anchors_x[i], 
//                                 anchors_y[i],
//                                 blk_misc.max_dist_x, blk_misc.max_dist_y, blk_misc.bw, blk_misc.chn_pen_gap, 
//                                 blk_misc.chn_pen_skip, blk_misc.is_cdna, blk_misc.n_seg);
//             if (sc == INT32_MIN) continue;
//             sc += f[i];
//             if (sc >= f[i+j+1] && sc != (anchors_y[i+j+1]>>32 & 0xff)) {
//                 f[i+j+1] = sc;
//                 p[i+j+1] = j+1;

//             }
//         }
//         __syncthreads();
//     }
// }

// inline __device__ void compute_sc_long_seg_one_wf(const int64_t* anchors_x, const int64_t* anchors_y, int32_t* range, 
//                     size_t start_idx, size_t end_idx,
//                     int32_t* f, uint16_t* p
// ){
//     Misc blk_misc = misc;
//     int tid = threadIdx.x;
//     // int bid = blockIdx.x;
//     // NOTE: smallest alignd offset that is greater than start_idx
//     //      anchor_offset = tid;
//     //      while (anchor_offset <= start_idx) anchor_offset += blockDim.x;
//     int anchor_offset = tid + (start_idx - tid + blockDim.x) / blockDim.x * blockDim.x;
//     // init f and p
//     for (size_t i=anchor_offset; i < end_idx; i += blockDim.x) {
//         f[i] = anchors_y[i] >> 32 & 0xff;
//         p[i] = 0;
//     }
//     // int64_t local_anchors[10];
//     int64_t anchor_x = anchors_x[anchor_offset];
//     int64_t anchor_y = anchors_y[anchor_offset];
//     __syncthreads();
//     // assert(range[end_idx-1] == 0);
//     for (size_t i=start_idx; i < end_idx; i++) {
//         int32_t range_i = range[i];
//         // if (range_i + i >= end_idx)
//         //     printf("range_i %d i %lu start_idx %lu, end_idx %lu\n", range_i, i, start_idx, end_idx);
//         // assert(range_i + i < end_idx);
//         // for (int32_t j = tid; j < range_i; j += blockDim.x) {
//         for (unsigned j = anchor_offset; j < i+range_i+1; j += blockDim.x) {
//             anchor_x = anchors_x[j];
//             anchor_y = anchors_y[j];
//             int32_t sc = comput_sc(
//                                 anchor_x, 
//                                 anchor_y, 
//                                 anchors_x[i], 
//                                 anchors_y[i],
//                                 blk_misc.max_dist_x, blk_misc.max_dist_y, blk_misc.bw, blk_misc.chn_pen_gap, 
//                                 blk_misc.chn_pen_skip, blk_misc.is_cdna, blk_misc.n_seg);
//             if (sc == INT32_MIN) continue;
//             sc += f[i];
//             if (sc >= f[j] && sc != (anchors_y[j]>>32 & 0xff)) {
//                 f[j] = sc;
//                 p[j] = j+1;
//             }
//         }
//         anchor_offset += (anchor_offset <= i+1) * blockDim.x; // update anchor offset
//         __syncthreads();
//     }
    
// }



/* kernels begin */


template <size_t short_block_size>

void score_generation_short(
                                /* Input: Anchor & Range Inputs */
                                int32_t* anchors_x, int32_t* anchors_y, int8_t* sid, int32_t *range, 
                                /* Input: Segmentations */
                                size_t *seg_start_arr,
                                /* Output: Score and Previous Anchor */
                                int32_t* f, uint16_t* p, 
                                /* Sizes*/
                                size_t total_n, size_t seg_count,
                                /* Output: Long segs */
                                int32_t* a_x_long, int32_t* a_y_long, int8_t* sid_long, int32_t* range_long, /* aggregated memory space for long seg */
                                size_t* total_n_long, size_t buffer_size_long 
                                , seg_t* long_seg, seg_t* long_seg_og, unsigned int *long_seg_count
                                ,seg_t *mid_seg, unsigned int *mid_seg_count,
                                Misc misc, int long_seg_cutoff,
                                int mid_seg_cutoff,
                                size_t &long_seg_start_idx_shared){
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int tid = item_ct1.get_local_id(2);
    int bid = item_ct1.get_group(2);

    size_t long_seg_start_idx;

    for (int segid = bid; segid < seg_count;
         segid += item_ct1.get_group_range(2)) {
        size_t start_idx = seg_start_arr[segid];
        if (start_idx == SIZE_MAX) continue; // start at a failed cut: continue to next iteration
        size_t end_idx = SIZE_MAX;
        int end_segid = segid + 1;
        while (true) {
            if (end_segid >= seg_count) {
                end_idx = total_n;
                break;
            }
            if (seg_start_arr[end_segid] != SIZE_MAX) {
                end_idx = seg_start_arr[end_segid];
                break;
            }
            ++end_segid;
        }
        if (end_segid > segid + long_seg_cutoff) {
            if (tid == 0) {
                /* Allocate space in long seg buffer */
                long_seg_start_idx = dpct::atomic_fetch_add<
                    sycl::access::address_space::generic_space>(
                    (unsigned long long int *)total_n_long,
                    (unsigned long long int)end_idx - start_idx);
                if (long_seg_start_idx + (end_idx - start_idx) >= buffer_size_long){ // long segement buffer is full
                /* rollback total_n_long */
                    dpct::atomic_fetch_add<
                        sycl::access::address_space::generic_space>(
                        (unsigned long long int *)total_n_long,
                        (unsigned long long int)(start_idx - end_idx));
                    long_seg_start_idx = SIZE_MAX;
                    // fallback to mid kernel
                    int mid_seg_idx = dpct::atomic_fetch_add<
                        sycl::access::address_space::generic_space>(
                        (unsigned long long int *)mid_seg_count, 1);
                    mid_seg[mid_seg_idx].start_idx = start_idx;
                    mid_seg[mid_seg_idx].end_idx = end_idx;
                } else {
                    int long_seg_idx = dpct::atomic_fetch_add<
                        sycl::access::address_space::generic_space>(
                        (unsigned long long int *)long_seg_count, 1);
                    long_seg[long_seg_idx].start_idx = long_seg_start_idx;
                    long_seg[long_seg_idx].end_idx = long_seg_start_idx + (end_idx - start_idx);
                    long_seg_og[long_seg_idx].start_idx = start_idx;
                    long_seg_og[long_seg_idx].end_idx = end_idx;
        //DEBUG: used for debug plchain_cal_long_seg_range_dis LONG_SEG_RANGE_DIS
        #ifdef DEBUG_VERBOSE
                    long_seg_og[long_seg_idx].start_segid = segid;
                    long_seg_og[long_seg_idx].end_segid = end_segid;
        #endif // DEBUG_VERBOSE
                }
            }
            // broadcast long_seg_start_idx to all scalar registers
            if (tid == 0) long_seg_start_idx_shared = long_seg_start_idx;
            sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
            long_seg_start_idx = long_seg_start_idx_shared;
            sycl::group_barrier(sycl::ext::oneapi::this_work_item::get_sub_group());
            if (long_seg_start_idx == SIZE_MAX)
                continue;  // failed to allocate long_seg buffer
            for (uint64_t idx = tid; idx < end_idx - start_idx;
                 idx += item_ct1.get_local_range(2)) {
                a_x_long[long_seg_start_idx + idx] = anchors_x[start_idx + idx];
                a_y_long[long_seg_start_idx + idx] = anchors_y[start_idx + idx];
                sid_long[long_seg_start_idx + idx] = sid[start_idx + idx];
                range_long[long_seg_start_idx + idx] = range[start_idx + idx];
            }
            continue;
        } else if (end_segid > segid + mid_seg_cutoff) {
            if (tid == 0) {
                int mid_seg_idx = dpct::atomic_fetch_add<
                    sycl::access::address_space::generic_space>(mid_seg_count,
                                                                1);
                mid_seg[mid_seg_idx].start_idx = start_idx;
                mid_seg[mid_seg_idx].end_idx = end_idx;
            }
            continue;
        }
        compute_sc_seg_one_wf(anchors_x, anchors_y, sid, range, start_idx,
                              end_idx, f, p, misc);
    }
}


template <size_t mid_block_size>

void score_generation_mid(int32_t* anchors_x, int32_t* anchors_y, int8_t* sid, int32_t *range,
                                seg_t *long_seg, unsigned int* long_seg_count,
                                int32_t* f, uint16_t* p, Misc misc){
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int tid = item_ct1.get_local_id(2);
    int bid = item_ct1.get_group(2);

    for (int segid = bid; segid < *long_seg_count;
         segid += item_ct1.get_group_range(2)) {
        seg_t seg = long_seg[segid];
        /*
        DPCT1118:3: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        compute_sc_seg_multi_wf(anchors_x, anchors_y, sid, range, seg.start_idx,
                                seg.end_idx, f, p, misc);
    }
}

template <size_t long_block_size>

void score_generation_long(int32_t* anchors_x, int32_t* anchors_y, int8_t* sid, int32_t *range,
                                seg_t *long_seg, unsigned int* long_seg_count,
                                int32_t* f, uint16_t* p, Misc misc){
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int tid = item_ct1.get_local_id(2);
    int bid = item_ct1.get_group(2);

    for (int segid = bid; segid < *long_seg_count;
         segid += item_ct1.get_group_range(2)) {
        seg_t seg = long_seg[segid];
        /*
        DPCT1118:4: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        compute_sc_seg_multi_wf(anchors_x, anchors_y, sid, range, seg.start_idx,
                                seg.end_idx, f, p, misc);
    }
}

// FIXME: merge together
template <size_t long_block_size>

void score_generation_long_map(int32_t* anchors_x, int32_t* anchors_y, int8_t* sid, int32_t *range,
                                seg_t *long_seg, unsigned int* long_seg_count,
                                int32_t* f, uint16_t* p, unsigned int* map,
                                Misc misc, unsigned &curr_long_segid,
                                unsigned int &segid){
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int tid = item_ct1.get_local_id(2);
    int bid = item_ct1.get_group(2);
    unsigned int seg_count = 0;

    // #ifdef DEBUG_CHECK
    // auto start = clock64();
    // #endif

    if (tid == 0 && bid == 0) {
        // init the first batch as the size of the grid
        curr_long_segid = item_ct1.get_group_range(2);
    }
    if (tid == 0) {
        segid = bid;
    }

    /*
    DPCT1065:19: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    while (segid < *long_seg_count) {
        seg_t seg = long_seg[map[segid]]; // sorted
        // seg_t seg = long_seg[segid]; // unsorted
        /*
        DPCT1118:5: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        compute_sc_seg_multi_wf(anchors_x, anchors_y, sid, range, seg.start_idx,
                                seg.end_idx, f, p, misc);
        seg_count++;
        if (tid == 0) segid =
            dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
                &curr_long_segid, 1);
        /*
        DPCT1118:6: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:20: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
}

void score_generation_naive(int32_t* anchors_x, int32_t* anchors_y, int8_t* sid, int32_t *range,
                        size_t *seg_start_arr, 
                        int32_t* f, uint16_t* p, size_t total_n, size_t seg_count,
                        Misc misc) {

    // NOTE: each block deal with one batch 
    // the number of threads in a block is fixed, so we need to calculate iter
    // n = end_idx_arr - start_idx_arr
    // iter = (range[i] - 1) / num_threads + 1

    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    int tid = item_ct1.get_local_id(2);
    int bid = item_ct1.get_group(2);
    for (int segid = bid; segid < seg_count;
         segid += item_ct1.get_group_range(2)) {
        /* calculate the segement for current block */
        size_t start_idx = seg_start_arr[segid];
        if (start_idx == SIZE_MAX) continue; // start at a failed cut: continue to next iteration
        size_t end_idx = SIZE_MAX;
        int end_segid = segid + 1;
        while (true) {
            if (end_segid >= seg_count) {
                end_idx = total_n;
                break;
            }
            if (seg_start_arr[end_segid] != SIZE_MAX) {
                end_idx = seg_start_arr[end_segid];
                break;
            }
            ++end_segid;
        }
        // assert(end_idx <= total_n);
        compute_sc_seg_one_wf(anchors_x, anchors_y, sid, range, start_idx,
                              end_idx, f, p, misc);
    }
}

/* kernels end */

/* host functions begin */
score_kernel_config_t score_kernel_config;

void plscore_upload_misc(Misc input_misc) try {
  // dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::queue q(sycl::property::queue::in_order{});
  sycl::queue &q_ct1 = q; 
  q_ct1.memcpy(misc.get_ptr(), &input_misc, sizeof(Misc));
  q_ct1.memcpy(long_seg_cutoff.get_ptr(), &score_kernel_config.long_seg_cutoff, sizeof(int));
  q_ct1.memcpy(mid_seg_cutoff.get_ptr(), &score_kernel_config.mid_seg_cutoff, sizeof(int));
  q_ct1.wait_and_throw();
} catch (const sycl::exception &e) {
  fprintf(stderr, "Error in %s:%i %s(): %s.\n", __FILE__, __LINE__, __func__, e.what());
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void plscore_async_short_mid_forward_dp(deviceMemPtr *dev_mem,
                                        dpct::queue_ptr *stream,
                                        sycl::event *stop_event_short) {
    size_t total_n = dev_mem->total_n;
    size_t cut_num = dev_mem->num_cut;
    size_t buffer_size_long = dev_mem->buffer_size_long;
    dpct::dim3 shortDimGrid(score_kernel_config.short_griddim, 1, 1);
    dpct::dim3 midDimGrid(score_kernel_config.mid_griddim, 1, 1);
    dpct::dim3 shortDimBlock(score_kernel_config.short_blockdim, 1, 1);

    // Run kernel;
    (*stream)->memset(dev_mem->d_mid_seg_count, 0, sizeof(unsigned int));

    if (score_kernel_config.short_blockdim == 32 ){
    misc.init(**stream);
    long_seg_cutoff.init(**stream);
    mid_seg_cutoff.init(**stream);

    *stop_event_short = (*stream)->submit([&](sycl::handler &cgh) {
      auto misc_ptr_ct1 = misc.get_ptr();
      auto long_seg_cutoff_ptr_ct1 = long_seg_cutoff.get_ptr();
      auto mid_seg_cutoff_ptr_ct1 = mid_seg_cutoff.get_ptr();

      sycl::local_accessor<size_t, 0> long_seg_start_idx_shared_acc_ct1(cgh);

      auto dev_mem_d_ax_ct0 = dev_mem->d_ax;
      auto dev_mem_d_ay_ct1 = dev_mem->d_ay;
      auto dev_mem_d_sid_ct2 = dev_mem->d_sid;
      auto dev_mem_d_range_ct3 = dev_mem->d_range;
      auto dev_mem_d_cut_ct4 = dev_mem->d_cut;
      auto dev_mem_d_f_ct5 = dev_mem->d_f;
      auto dev_mem_d_p_ct6 = dev_mem->d_p;
      auto dev_mem_d_ax_long_ct9 = dev_mem->d_ax_long;
      auto dev_mem_d_ay_long_ct10 = dev_mem->d_ay_long;
      auto dev_mem_d_sid_long_ct11 = dev_mem->d_sid_long;
      auto dev_mem_d_range_long_ct12 = dev_mem->d_range_long;
      auto dev_mem_d_total_n_long_ct13 = dev_mem->d_total_n_long;
      auto dev_mem_d_long_seg_ct15 = dev_mem->d_long_seg;
      auto dev_mem_d_long_seg_og_ct16 = dev_mem->d_long_seg_og;
      auto dev_mem_d_long_seg_count_ct17 = dev_mem->d_long_seg_count;
      auto dev_mem_d_mid_seg_ct18 = dev_mem->d_mid_seg;
      auto dev_mem_d_mid_seg_count_ct19 = dev_mem->d_mid_seg_count;

      cgh.parallel_for(
          sycl::nd_range<3>(shortDimGrid * sycl::range<3>(1, 1, 32),
                            sycl::range<3>(1, 1, 32)),
          [=](sycl::nd_item<3> item_ct1) {
            score_generation_short<32>(
                dev_mem_d_ax_ct0, dev_mem_d_ay_ct1, dev_mem_d_sid_ct2,
                dev_mem_d_range_ct3, dev_mem_d_cut_ct4, dev_mem_d_f_ct5,
                dev_mem_d_p_ct6, total_n, cut_num, dev_mem_d_ax_long_ct9,
                dev_mem_d_ay_long_ct10, dev_mem_d_sid_long_ct11,
                dev_mem_d_range_long_ct12, dev_mem_d_total_n_long_ct13,
                buffer_size_long, dev_mem_d_long_seg_ct15,
                dev_mem_d_long_seg_og_ct16, dev_mem_d_long_seg_count_ct17,
                dev_mem_d_mid_seg_ct18, dev_mem_d_mid_seg_count_ct19,
                *misc_ptr_ct1, *long_seg_cutoff_ptr_ct1,
                *mid_seg_cutoff_ptr_ct1, long_seg_start_idx_shared_acc_ct1);
          });
    });
    } else if (score_kernel_config.short_blockdim == 64) {
    misc.init(**stream);
    long_seg_cutoff.init(**stream);
    mid_seg_cutoff.init(**stream);

    *stop_event_short = (*stream)->submit([&](sycl::handler &cgh) {
      auto misc_ptr_ct1 = misc.get_ptr();
      auto long_seg_cutoff_ptr_ct1 = long_seg_cutoff.get_ptr();
      auto mid_seg_cutoff_ptr_ct1 = mid_seg_cutoff.get_ptr();

      sycl::local_accessor<size_t, 0> long_seg_start_idx_shared_acc_ct1(cgh);

      auto dev_mem_d_ax_ct0 = dev_mem->d_ax;
      auto dev_mem_d_ay_ct1 = dev_mem->d_ay;
      auto dev_mem_d_sid_ct2 = dev_mem->d_sid;
      auto dev_mem_d_range_ct3 = dev_mem->d_range;
      auto dev_mem_d_cut_ct4 = dev_mem->d_cut;
      auto dev_mem_d_f_ct5 = dev_mem->d_f;
      auto dev_mem_d_p_ct6 = dev_mem->d_p;
      auto dev_mem_d_ax_long_ct9 = dev_mem->d_ax_long;
      auto dev_mem_d_ay_long_ct10 = dev_mem->d_ay_long;
      auto dev_mem_d_sid_long_ct11 = dev_mem->d_sid_long;
      auto dev_mem_d_range_long_ct12 = dev_mem->d_range_long;
      auto dev_mem_d_total_n_long_ct13 = dev_mem->d_total_n_long;
      auto dev_mem_d_long_seg_ct15 = dev_mem->d_long_seg;
      auto dev_mem_d_long_seg_og_ct16 = dev_mem->d_long_seg_og;
      auto dev_mem_d_long_seg_count_ct17 = dev_mem->d_long_seg_count;
      auto dev_mem_d_mid_seg_ct18 = dev_mem->d_mid_seg;
      auto dev_mem_d_mid_seg_count_ct19 = dev_mem->d_mid_seg_count;

      cgh.parallel_for(
          sycl::nd_range<3>(shortDimGrid * sycl::range<3>(1, 1, 64),
                            sycl::range<3>(1, 1, 64)),
          [=](sycl::nd_item<3> item_ct1) {
            score_generation_short<64>(
                dev_mem_d_ax_ct0, dev_mem_d_ay_ct1, dev_mem_d_sid_ct2,
                dev_mem_d_range_ct3, dev_mem_d_cut_ct4, dev_mem_d_f_ct5,
                dev_mem_d_p_ct6, total_n, cut_num, dev_mem_d_ax_long_ct9,
                dev_mem_d_ay_long_ct10, dev_mem_d_sid_long_ct11,
                dev_mem_d_range_long_ct12, dev_mem_d_total_n_long_ct13,
                buffer_size_long, dev_mem_d_long_seg_ct15,
                dev_mem_d_long_seg_og_ct16, dev_mem_d_long_seg_count_ct17,
                dev_mem_d_mid_seg_ct18, dev_mem_d_mid_seg_count_ct19,
                *misc_ptr_ct1, *long_seg_cutoff_ptr_ct1,
                *mid_seg_cutoff_ptr_ct1, long_seg_start_idx_shared_acc_ct1);
          });
    });
    } else {
        fprintf(stderr,
                "[ERROR] Unsupported warpsize: %d. mm2-gb only supports device "
                "with a warpsize of 32 / 64. ",
                score_kernel_config.short_blockdim);
        exit(1);
    }
    sycl_check(**stream);


    if (score_kernel_config.mid_blockdim == 128){
    misc.init(**stream);

    (*stream)->submit([&](sycl::handler &cgh) {
      auto misc_ptr_ct1 = misc.get_ptr();

      auto dev_mem_d_ax_ct0 = dev_mem->d_ax;
      auto dev_mem_d_ay_ct1 = dev_mem->d_ay;
      auto dev_mem_d_sid_ct2 = dev_mem->d_sid;
      auto dev_mem_d_range_ct3 = dev_mem->d_range;
      auto dev_mem_d_mid_seg_ct4 = dev_mem->d_mid_seg;
      auto dev_mem_d_mid_seg_count_ct5 = dev_mem->d_mid_seg_count;
      auto dev_mem_d_f_ct6 = dev_mem->d_f;
      auto dev_mem_d_p_ct7 = dev_mem->d_p;

      cgh.parallel_for(sycl::nd_range<3>(midDimGrid * sycl::range<3>(1, 1, 128),
                                         sycl::range<3>(1, 1, 128)),
                       [=](sycl::nd_item<3> item_ct1) {
                         score_generation_mid<128>(
                             dev_mem_d_ax_ct0, dev_mem_d_ay_ct1,
                             dev_mem_d_sid_ct2, dev_mem_d_range_ct3,
                             dev_mem_d_mid_seg_ct4, dev_mem_d_mid_seg_count_ct5,
                             dev_mem_d_f_ct6, dev_mem_d_p_ct7, *misc_ptr_ct1);
                       });
    });
    } else if (score_kernel_config.mid_blockdim == 256){
    misc.init(**stream);

    (*stream)->submit([&](sycl::handler &cgh) {
      auto misc_ptr_ct1 = misc.get_ptr();

      auto dev_mem_d_ax_ct0 = dev_mem->d_ax;
      auto dev_mem_d_ay_ct1 = dev_mem->d_ay;
      auto dev_mem_d_sid_ct2 = dev_mem->d_sid;
      auto dev_mem_d_range_ct3 = dev_mem->d_range;
      auto dev_mem_d_mid_seg_ct4 = dev_mem->d_mid_seg;
      auto dev_mem_d_mid_seg_count_ct5 = dev_mem->d_mid_seg_count;
      auto dev_mem_d_f_ct6 = dev_mem->d_f;
      auto dev_mem_d_p_ct7 = dev_mem->d_p;

      cgh.parallel_for(sycl::nd_range<3>(midDimGrid * sycl::range<3>(1, 1, 256),
                                         sycl::range<3>(1, 1, 256)),
                       [=](sycl::nd_item<3> item_ct1) {
                         score_generation_mid<256>(
                             dev_mem_d_ax_ct0, dev_mem_d_ay_ct1,
                             dev_mem_d_sid_ct2, dev_mem_d_range_ct3,
                             dev_mem_d_mid_seg_ct4, dev_mem_d_mid_seg_count_ct5,
                             dev_mem_d_f_ct6, dev_mem_d_p_ct7, *misc_ptr_ct1);
                       });
    });
    } else if (score_kernel_config.mid_blockdim == 512){
        /*
        DPCT1049:7: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
    misc.init(**stream);

    (*stream)->submit([&](sycl::handler &cgh) {
      auto misc_ptr_ct1 = misc.get_ptr();

      auto dev_mem_d_ax_ct0 = dev_mem->d_ax;
      auto dev_mem_d_ay_ct1 = dev_mem->d_ay;
      auto dev_mem_d_sid_ct2 = dev_mem->d_sid;
      auto dev_mem_d_range_ct3 = dev_mem->d_range;
      auto dev_mem_d_mid_seg_ct4 = dev_mem->d_mid_seg;
      auto dev_mem_d_mid_seg_count_ct5 = dev_mem->d_mid_seg_count;
      auto dev_mem_d_f_ct6 = dev_mem->d_f;
      auto dev_mem_d_p_ct7 = dev_mem->d_p;

      cgh.parallel_for(sycl::nd_range<3>(midDimGrid * sycl::range<3>(1, 1, 512),
                                         sycl::range<3>(1, 1, 512)),
                       [=](sycl::nd_item<3> item_ct1) {
                         score_generation_mid<512>(
                             dev_mem_d_ax_ct0, dev_mem_d_ay_ct1,
                             dev_mem_d_sid_ct2, dev_mem_d_range_ct3,
                             dev_mem_d_mid_seg_ct4, dev_mem_d_mid_seg_count_ct5,
                             dev_mem_d_f_ct6, dev_mem_d_p_ct7, *misc_ptr_ct1);
                       });
    });
    } else if (score_kernel_config.mid_blockdim == 1024){
        /*
        DPCT1049:8: The work-group size passed to the SYCL kernel may exceed the
        limit. To get the device limit, query info::device::max_work_group_size.
        Adjust the work-group size if needed.
        */
    misc.init(**stream);

    (*stream)->submit([&](sycl::handler &cgh) {
      auto misc_ptr_ct1 = misc.get_ptr();

      auto dev_mem_d_ax_ct0 = dev_mem->d_ax;
      auto dev_mem_d_ay_ct1 = dev_mem->d_ay;
      auto dev_mem_d_sid_ct2 = dev_mem->d_sid;
      auto dev_mem_d_range_ct3 = dev_mem->d_range;
      auto dev_mem_d_mid_seg_ct4 = dev_mem->d_mid_seg;
      auto dev_mem_d_mid_seg_count_ct5 = dev_mem->d_mid_seg_count;
      auto dev_mem_d_f_ct6 = dev_mem->d_f;
      auto dev_mem_d_p_ct7 = dev_mem->d_p;

      cgh.parallel_for(
          sycl::nd_range<3>(midDimGrid * sycl::range<3>(1, 1, 1024),
                            sycl::range<3>(1, 1, 1024)),
          [=](sycl::nd_item<3> item_ct1) {
            score_generation_mid<1024>(
                dev_mem_d_ax_ct0, dev_mem_d_ay_ct1, dev_mem_d_sid_ct2,
                dev_mem_d_range_ct3, dev_mem_d_mid_seg_ct4,
                dev_mem_d_mid_seg_count_ct5, dev_mem_d_f_ct6, dev_mem_d_p_ct7,
                *misc_ptr_ct1);
          });
    });
    } else {
        fprintf(stderr,
                "[ERROR] Unsupported mid_blockdim: %d. mm2-gb only supports a "
                "blockdim of 128/256/512/1024 for mid kernel \n\n"
                "Please adjust score_kernel:mid_blockdim in gpu config file. ",
                score_kernel_config.mid_blockdim);
        exit(1);
    }
    sycl_check(**stream);

#ifdef DEBUG_PRINT
    // fprintf(stderr, "[Info] %s (%s:%d) short mid score kernel launched\n", __func__, __FILE__, __LINE__);
#endif
}

void plscore_async_long_forward_dp(deviceMemPtr *dev_mem,
                                   dpct::queue_ptr *stream) {
    size_t total_n = dev_mem->total_n;
    size_t cut_num = dev_mem->num_cut;
    size_t buffer_size_long = dev_mem->buffer_size_long;
    dpct::dim3 longDimGrid(score_kernel_config.long_griddim, 1, 1);

#ifdef DEBUG_VERBOSE
    fprintf(stderr, "[Debug] %s (%s:%d) Long Grid Dim = %d\n", __func__, __FILE__, __LINE__, longDimGrid.x);
#endif // DEBUG_VERBOSE


    if (score_kernel_config.long_blockdim == 1024){
    /*
    DPCT1049:9: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    misc.init(**stream);
    curr_long_segid.init(**stream);

    (*stream)->submit([&](sycl::handler &cgh) {
      auto misc_ptr_ct1 = misc.get_ptr();
      auto curr_long_segid_ptr_ct1 = curr_long_segid.get_ptr();

      sycl::local_accessor<unsigned int, 0> segid_acc_ct1(cgh);

      auto dev_mem_d_ax_long_ct0 = dev_mem->d_ax_long;
      auto dev_mem_d_ay_long_ct1 = dev_mem->d_ay_long;
      auto dev_mem_d_sid_long_ct2 = dev_mem->d_sid_long;
      auto dev_mem_d_range_long_ct3 = dev_mem->d_range_long;
      auto dev_mem_d_long_seg_ct4 = dev_mem->d_long_seg;
      auto dev_mem_d_long_seg_count_ct5 = dev_mem->d_long_seg_count;
      auto dev_mem_d_f_long_ct6 = dev_mem->d_f_long;
      auto dev_mem_d_p_long_ct7 = dev_mem->d_p_long;
      auto dev_mem_d_map_ct8 = dev_mem->d_map;

      cgh.parallel_for(
          sycl::nd_range<3>(longDimGrid * sycl::range<3>(1, 1, 1024),
                            sycl::range<3>(1, 1, 1024)),
          [=](sycl::nd_item<3> item_ct1) {
            score_generation_long_map<1024>(
                dev_mem_d_ax_long_ct0, dev_mem_d_ay_long_ct1,
                dev_mem_d_sid_long_ct2, dev_mem_d_range_long_ct3,
                dev_mem_d_long_seg_ct4, dev_mem_d_long_seg_count_ct5,
                dev_mem_d_f_long_ct6, dev_mem_d_p_long_ct7, dev_mem_d_map_ct8,
                *misc_ptr_ct1, *curr_long_segid_ptr_ct1, segid_acc_ct1);
          });
    });
    } else {
        fprintf(stderr,
                "[ERROR] Unsupported MaxThreadsPerBlock: %d. mm2-gb only supports a blockdim of 1024 for long kernel ",
                score_kernel_config.long_blockdim);
        exit(1);
    }

    sycl_check(**stream);

#ifdef DEBUG_PRINT
    // fprintf(stderr, "[Info] %s (%s:%d) long score generation launched\n", __func__, __FILE__, __LINE__);
#endif
}

void plscore_async_naive_forward_dp(deviceMemPtr *dev_mem,
                                    dpct::queue_ptr *stream) {
    size_t total_n = dev_mem->total_n;
    size_t cut_num = dev_mem->num_cut;
    dpct::dim3 DimBlock(score_kernel_config.long_blockdim, 1, 1);
    dpct::dim3 longDimGrid(score_kernel_config.long_griddim, 1, 1);
    dpct::dim3 shortDimGrid(score_kernel_config.short_griddim, 1, 1);

    // Run kernel
    // printf("Grid Dim, %d\n", DimGrid.x);
    /*
    DPCT1049:10: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
  {
    misc.init(**stream);

    (*stream)->submit([&](sycl::handler &cgh) {
      auto misc_ptr_ct1 = misc.get_ptr();

      auto dev_mem_d_ax_ct0 = dev_mem->d_ax;
      auto dev_mem_d_ay_ct1 = dev_mem->d_ay;
      auto dev_mem_d_sid_ct2 = dev_mem->d_sid;
      auto dev_mem_d_range_ct3 = dev_mem->d_range;
      auto dev_mem_d_cut_ct4 = dev_mem->d_cut;
      auto dev_mem_d_f_ct5 = dev_mem->d_f;
      auto dev_mem_d_p_ct6 = dev_mem->d_p;

      cgh.parallel_for(sycl::nd_range<3>(shortDimGrid * DimBlock, DimBlock),
                       [=](sycl::nd_item<3> item_ct1) {
                         score_generation_naive(
                             dev_mem_d_ax_ct0, dev_mem_d_ay_ct1,
                             dev_mem_d_sid_ct2, dev_mem_d_range_ct3,
                             dev_mem_d_cut_ct4, dev_mem_d_f_ct5,
                             dev_mem_d_p_ct6, total_n, cut_num, *misc_ptr_ct1);
                       });
    });
  }
  sycl_check(**stream);
#ifdef DEBUG_VERBOSE
    fprintf(stderr, "[M::%s] score generation kernel launch success\n", __func__);
#endif
}

