/* GPU memory management  */

#ifdef DEBUG_PRINT
#define DPCT_PROFILING_ENABLED
#endif

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "syclcheck.cpp"
#include "plmem.dp.hpp"
#include "plrange.dp.hpp"
#include "plscore.dp.hpp"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
void plmem_malloc_host_mem(hostMemPtr *host_mem, size_t anchor_per_batch,
                           int range_grid_size, size_t buffer_size_long) try {
    sycl::queue q(sycl::property::queue::in_order{});
    sycl::queue &q_ct1 = q; 
#ifdef DEBUG_PRINT
  size_t host_mem_size;
  host_mem_size =
      anchor_per_batch * (sizeof(int32_t) + sizeof(int32_t) + sizeof(int8_t) +
                          sizeof(int32_t) + sizeof(int32_t) + sizeof(uint16_t));
  host_mem_size +=
      range_grid_size * (sizeof(size_t) + sizeof(size_t) + sizeof(size_t));
  fprintf(stderr, "[Info] Host Malloc Pinned Memory Size %.2f GB\n",
          (float)host_mem_size / OneG);
#endif

  // data array
  host_mem->ax = sycl::malloc_host<int32_t>(anchor_per_batch, q_ct1);
  host_mem->ay = sycl::malloc_host<int32_t>(anchor_per_batch, q_ct1);
  host_mem->sid = sycl::malloc_host<int8_t>(anchor_per_batch, q_ct1);
  host_mem->xrev = sycl::malloc_host<int32_t>(anchor_per_batch, q_ct1);
  host_mem->f = sycl::malloc_host<int32_t>(anchor_per_batch, q_ct1);
  host_mem->p = sycl::malloc_host<uint16_t>(anchor_per_batch, q_ct1);

  // index
  host_mem->start_idx = sycl::malloc_host<size_t>(range_grid_size, q_ct1);
  host_mem->read_end_idx = sycl::malloc_host<size_t>(range_grid_size, q_ct1);
  host_mem->cut_start_idx = sycl::malloc_host<size_t>(range_grid_size, q_ct1);

  host_mem->long_segs_num = sycl::malloc_host<unsigned int>(1, q_ct1);

} catch (const sycl::exception &e) {
  fprintf(stderr, "Error in %s:%i %s(): %s.\n", __FILE__, __LINE__, __func__, e.what());
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void plmem_malloc_long_mem(longMemPtr *long_mem, size_t buffer_size_long) try {
  sycl::queue q(sycl::property::queue::in_order{});
  sycl::queue &q_ct1 = q; 
#ifdef DEBUG_PRINT
  size_t host_mem_size;
  host_mem_size =
      buffer_size_long /
      (score_kernel_config.long_seg_cutoff * score_kernel_config.cut_unit) *
      sizeof(seg_t);
  host_mem_size += buffer_size_long * (sizeof(int32_t) + sizeof(uint16_t));
  fprintf(stderr, "[Info] Host Malloc Pinned Memory Size %.2f GB (long seg)\n",
          (float)host_mem_size / OneG);
#endif
  // data array
  long_mem->long_segs_og_idx = sycl::malloc_host<seg_t>(
      buffer_size_long /
          (score_kernel_config.long_seg_cutoff * score_kernel_config.cut_unit),
      q_ct1);
  long_mem->f_long = sycl::malloc_host<int32_t>(buffer_size_long, q_ct1);
  long_mem->p_long = sycl::malloc_host<uint16_t>(buffer_size_long, q_ct1);
  long_mem->total_long_segs_num = sycl::malloc_host<unsigned int>(1, q_ct1);
  long_mem->total_long_segs_n = sycl::malloc_host<size_t>(1, q_ct1);
} catch (const sycl::exception &e) {
  fprintf(stderr, "Error in %s:%i %s(): %s.\n", __FILE__, __LINE__, __func__, e.what());
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void plmem_free_host_mem(hostMemPtr *host_mem) try {
  sycl::queue q(sycl::property::queue::in_order{});
  sycl::queue &q_ct1 = q; 
  sycl::free(host_mem->ax, q_ct1);
  sycl::free(host_mem->ay, q_ct1);
  sycl::free(host_mem->sid, q_ct1);
  sycl::free(host_mem->xrev, q_ct1);
  sycl::free(host_mem->f, q_ct1);
  sycl::free(host_mem->p, q_ct1);
  sycl::free(host_mem->start_idx, q_ct1);
  sycl::free(host_mem->read_end_idx, q_ct1);
  sycl::free(host_mem->cut_start_idx, q_ct1);
  sycl::free(host_mem->long_segs_num, q_ct1);
} catch (const sycl::exception &e) {
  fprintf(stderr, "Error in %s:%i %s(): %s.\n", __FILE__, __LINE__, __func__, e.what());
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void plmem_free_long_mem(longMemPtr *long_mem) try {
  sycl::queue q(sycl::property::queue::in_order{});
  sycl::queue &q_ct1 = q; 
  sycl::free(long_mem->long_segs_og_idx, q_ct1);
  sycl::free(long_mem->f_long, q_ct1);
  sycl::free(long_mem->p_long, q_ct1);
  sycl::free(long_mem->total_long_segs_num, q_ct1);
  sycl::free(long_mem->total_long_segs_n, q_ct1);
} catch (const sycl::exception &e) {
  fprintf(stderr, "Error in %s:%i %s(): %s.\n", __FILE__, __LINE__, __func__, e.what());
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void plmem_malloc_device_mem(deviceMemPtr *dev_mem, size_t anchor_per_batch,
                             int range_grid_size, int num_cut) try {
  sycl::queue q(sycl::property::queue::in_order{});
  sycl::queue &q_ct1 = q; 
  // data array
  dev_mem->d_ax = sycl::malloc_device<int32_t>(anchor_per_batch, q_ct1);
  dev_mem->d_ay = sycl::malloc_device<int32_t>(anchor_per_batch, q_ct1);
  dev_mem->d_sid = sycl::malloc_device<int8_t>(anchor_per_batch, q_ct1);
  dev_mem->d_xrev = sycl::malloc_device<int32_t>(anchor_per_batch, q_ct1);
  dev_mem->d_range = sycl::malloc_device<int32_t>(anchor_per_batch, q_ct1);
  dev_mem->d_f = sycl::malloc_device<int32_t>(anchor_per_batch, q_ct1);
  dev_mem->d_p = sycl::malloc_device<uint16_t>(anchor_per_batch, q_ct1);

  // index
  dev_mem->d_start_idx = sycl::malloc_device<size_t>(range_grid_size, q_ct1);
  dev_mem->d_read_end_idx = sycl::malloc_device<size_t>(range_grid_size, q_ct1);
  dev_mem->d_cut_start_idx = sycl::malloc_device<size_t>(range_grid_size, q_ct1);

  // cut
  dev_mem->d_cut = sycl::malloc_device<size_t>(num_cut, q_ct1);
  dev_mem->d_long_seg_count = sycl::malloc_device<unsigned int>(1, q_ct1);
  dev_mem->d_long_seg = sycl::malloc_device<seg_t>(
      dev_mem->buffer_size_long /
          (score_kernel_config.long_seg_cutoff * score_kernel_config.cut_unit),
      q_ct1);
  dev_mem->d_long_seg_og = sycl::malloc_device<seg_t>(
      dev_mem->buffer_size_long /
          (score_kernel_config.long_seg_cutoff * score_kernel_config.cut_unit),
      q_ct1);
  dev_mem->d_mid_seg_count = sycl::malloc_device<unsigned int>(1, q_ct1);
  dev_mem->d_mid_seg = sycl::malloc_device<seg_t>(
      num_cut / (score_kernel_config.mid_seg_cutoff + 1), q_ct1);

  size_t gpu_free_mem, gpu_total_mem;
  /*
  DPCT1106:21: 'cudaMemGetInfo' was migrated with the Intel extensions for
  device information which may not be supported by all compilers or runtimes.
  You may need to adjust the code.
  */
  // dpct::get_current_device().get_memory_info(gpu_free_mem, gpu_total_mem);
#ifdef DEBUG_PRINT
  // fprintf(stderr,
  //         "[Info] GPU free mem: %f GB, total mem: %f GB (before alloc long seg "
  //         "buffer) \n",
  //         (float)gpu_free_mem / OneG, (float)gpu_total_mem / OneG);
#endif

  // long seg buffer
  dev_mem->d_ax_long =
      sycl::malloc_device<int32_t>(dev_mem->buffer_size_long, q_ct1);
  dev_mem->d_ay_long =
      sycl::malloc_device<int32_t>(dev_mem->buffer_size_long, q_ct1);
  dev_mem->d_sid_long =
      sycl::malloc_device<int8_t>(dev_mem->buffer_size_long, q_ct1);
  dev_mem->d_range_long =
      sycl::malloc_device<int32_t>(dev_mem->buffer_size_long, q_ct1);
  dev_mem->d_total_n_long = sycl::malloc_device<size_t>(1, q_ct1);
  dev_mem->d_f_long =
      sycl::malloc_device<int32_t>(dev_mem->buffer_size_long, q_ct1);
  dev_mem->d_p_long =
      sycl::malloc_device<uint16_t>(dev_mem->buffer_size_long, q_ct1);
} catch (const sycl::exception &e) {
  fprintf(stderr, "Error in %s:%i %s(): %s.\n", __FILE__, __LINE__, __func__, e.what());
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void plmem_free_device_mem(deviceMemPtr *dev_mem) try {
  sycl::queue q(sycl::property::queue::in_order{});
  sycl::queue &q_ct1 = q; 
  sycl::free(dev_mem->d_ax, q_ct1);
  sycl::free(dev_mem->d_ay, q_ct1);
  sycl::free(dev_mem->d_sid, q_ct1);
  sycl::free(dev_mem->d_xrev, q_ct1);
  sycl::free(dev_mem->d_range, q_ct1);
  sycl::free(dev_mem->d_f, q_ct1);
  sycl::free(dev_mem->d_p, q_ct1);

  sycl::free(dev_mem->d_start_idx, q_ct1);
  sycl::free(dev_mem->d_read_end_idx, q_ct1);
  sycl::free(dev_mem->d_cut_start_idx, q_ct1);

  sycl::free(dev_mem->d_cut, q_ct1);
  sycl::free(dev_mem->d_long_seg, q_ct1);
  sycl::free(dev_mem->d_long_seg_count, q_ct1);
  sycl::free(dev_mem->d_mid_seg, q_ct1);
  sycl::free(dev_mem->d_mid_seg_count, q_ct1);

  sycl::free(dev_mem->d_ax_long, q_ct1);
  sycl::free(dev_mem->d_ay_long, q_ct1);
  sycl::free(dev_mem->d_sid_long, q_ct1);
  sycl::free(dev_mem->d_range_long, q_ct1);
  sycl::free(dev_mem->d_total_n_long, q_ct1);
} catch (const sycl::exception &e) {
  fprintf(stderr, "Error in %s:%i %s(): %s.\n", __FILE__, __LINE__, __func__, e.what());
  fflush(stderr);
  exit(EXIT_FAILURE);
}

/**
 * Input
 *  reads[]:    array
 *  n_reads
 *  config:     range kernel configuartions
 * Output
 *  *host_mem   populate host_mem
 */
void plmem_reorg_input_arr(chain_read_t *reads, int n_read,
                           hostMemPtr *host_mem, range_kernel_config_t config) {
  size_t total_n = 0, cut_num = 0;
  size_t griddim = 0;

  host_mem->size = n_read;
  for (int i = 0; i < n_read; i++) {
    total_n += reads[i].n;
  }
  host_mem->total_n = total_n;

  size_t idx = 0;
  for (int i = 0; i < n_read; i++) {
    int n = reads[i].n;
    int block_num = (n - 1) / config.anchor_per_block + 1;

    host_mem->start_idx[griddim] = idx;
    size_t end_idx = idx + config.anchor_per_block;
    host_mem->read_end_idx[griddim] = idx + n;
    host_mem->cut_start_idx[griddim] = cut_num;
    for (int j = 1; j < block_num; j++) {
      cut_num += (config.anchor_per_block / config.blockdim);
      host_mem->start_idx[griddim + j] = end_idx;
      end_idx = host_mem->start_idx[griddim + j] + config.anchor_per_block;
      host_mem->read_end_idx[griddim + j] = idx + n;
      host_mem->cut_start_idx[griddim + j] = cut_num;
    }
    cut_num +=
        (n - (block_num - 1) * config.anchor_per_block - 1) / config.blockdim;
    end_idx = idx + n;

    griddim += block_num;

    for (int j = 0; j < n; j++) {
      host_mem->ax[idx] = (int32_t)reads[i].a[j].x;
      host_mem->ay[idx] = (int32_t)reads[i].a[j].y;
      host_mem->sid[idx] =
          (reads[i].a[j].y & MM_SEED_SEG_MASK) >> MM_SEED_SEG_SHIFT;
      host_mem->xrev[idx] = reads[i].a[j].x >> 32;
      ++idx;
    }
  }
  host_mem->cut_num = cut_num;
  host_mem->griddim = griddim;
}

void plmem_async_h2d_short_memcpy(stream_ptr_t *stream_ptrs, size_t uid) {
  hostMemPtr *host_mem = &stream_ptrs->host_mems[uid];
  deviceMemPtr *dev_mem = &stream_ptrs->dev_mem;
  sycl::queue **stream = &stream_ptrs->cudastream;
  /*
  DPCT1124:22: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(dev_mem->d_ax, host_mem->ax,
                  sizeof(int32_t) * host_mem->total_n);
  /*
  DPCT1124:23: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(dev_mem->d_ay, host_mem->ay,
                  sizeof(int32_t) * host_mem->total_n);
  /*
  DPCT1124:24: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(dev_mem->d_sid, host_mem->sid,
                  sizeof(int8_t) * host_mem->total_n);
  /*
  DPCT1124:25: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(dev_mem->d_xrev, host_mem->xrev,
                  sizeof(int32_t) * host_mem->total_n);
  /*
  DPCT1124:26: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(dev_mem->d_start_idx, host_mem->start_idx,
                  sizeof(size_t) * host_mem->griddim);
  /*
  DPCT1124:27: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(dev_mem->d_read_end_idx, host_mem->read_end_idx,
                  sizeof(size_t) * host_mem->griddim);
  /*
  DPCT1124:28: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(dev_mem->d_cut_start_idx, host_mem->cut_start_idx,
                  sizeof(size_t) * host_mem->griddim);
  (*stream)->memset(dev_mem->d_cut, 0xff, sizeof(size_t) * host_mem->cut_num);
  (*stream)->memset(dev_mem->d_f, 0, sizeof(int32_t) * host_mem->total_n);
  (*stream)->memset(dev_mem->d_p, 0, sizeof(uint16_t) * host_mem->total_n);
  sycl_check(**stream);
  dev_mem->total_n = host_mem->total_n;
  dev_mem->num_cut = host_mem->cut_num;
  dev_mem->size = host_mem->size;
  dev_mem->griddim = host_mem->griddim;
}

void plmem_async_h2d_memcpy(stream_ptr_t *stream_ptrs) {
  size_t uid = 0;
  hostMemPtr *host_mem = &stream_ptrs->host_mems[uid];
  deviceMemPtr *dev_mem = &stream_ptrs->dev_mem;
  sycl::queue **stream = &stream_ptrs->cudastream;
  /*
  DPCT1124:29: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(dev_mem->d_ax, host_mem->ax,
                  sizeof(int32_t) * host_mem->total_n);
  /*
  DPCT1124:30: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(dev_mem->d_ay, host_mem->ay,
                  sizeof(int32_t) * host_mem->total_n);
  /*
  DPCT1124:31: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(dev_mem->d_sid, host_mem->sid,
                  sizeof(int8_t) * host_mem->total_n);
  /*
  DPCT1124:32: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(dev_mem->d_xrev, host_mem->xrev,
                  sizeof(int32_t) * host_mem->total_n);
  /*
  DPCT1124:33: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(dev_mem->d_start_idx, host_mem->start_idx,
                  sizeof(size_t) * host_mem->griddim);
  /*
  DPCT1124:34: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(dev_mem->d_read_end_idx, host_mem->read_end_idx,
                  sizeof(size_t) * host_mem->griddim);
  /*
  DPCT1124:35: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(dev_mem->d_cut_start_idx, host_mem->cut_start_idx,
                  sizeof(size_t) * host_mem->griddim);
  (*stream)->memset(dev_mem->d_cut, 0xff, sizeof(size_t) * host_mem->cut_num);
  (*stream)->memset(dev_mem->d_f, 0, sizeof(int32_t) * host_mem->total_n);
  (*stream)->memset(dev_mem->d_p, 0, sizeof(uint16_t) * host_mem->total_n);
  sycl_check(**stream);
  dev_mem->total_n = host_mem->total_n;
  dev_mem->num_cut = host_mem->cut_num;
  dev_mem->size = host_mem->size;
  dev_mem->griddim = host_mem->griddim;
}

void plmem_sync_h2d_memcpy(hostMemPtr *host_mem, deviceMemPtr *dev_mem) try {
  sycl::queue q(sycl::property::queue::in_order{});
  sycl::queue &q_ct1 = q; 
  q_ct1.memcpy(dev_mem->d_ax, host_mem->ax, sizeof(int32_t) * host_mem->total_n)
      .wait_and_throw();
  q_ct1.memcpy(dev_mem->d_ay, host_mem->ay, sizeof(int32_t) * host_mem->total_n)
      .wait_and_throw();
  q_ct1
      .memcpy(dev_mem->d_sid, host_mem->sid, sizeof(int8_t) * host_mem->total_n)
      .wait_and_throw();
  q_ct1
      .memcpy(dev_mem->d_xrev, host_mem->xrev,
              sizeof(int32_t) * host_mem->total_n)
      .wait_and_throw();
  q_ct1
      .memcpy(dev_mem->d_start_idx, host_mem->start_idx,
              sizeof(size_t) * host_mem->griddim)
      .wait_and_throw();
  q_ct1
      .memcpy(dev_mem->d_read_end_idx, host_mem->read_end_idx,
              sizeof(size_t) * host_mem->griddim)
      .wait_and_throw();
  q_ct1
      .memcpy(dev_mem->d_cut_start_idx, host_mem->cut_start_idx,
              sizeof(size_t) * host_mem->griddim)
      .wait_and_throw();
  q_ct1.memset(dev_mem->d_cut, 0xff, sizeof(size_t) * host_mem->cut_num).wait_and_throw();
  dev_mem->total_n = host_mem->total_n;
  dev_mem->num_cut = host_mem->cut_num;
  dev_mem->size = host_mem->size;
  dev_mem->griddim = host_mem->griddim;
} catch (const sycl::exception &e) {
  fprintf(stderr, "Error in %s:%i %s(): %s.\n", __FILE__, __LINE__, __func__, e.what());
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void plmem_async_d2h_memcpy(stream_ptr_t *stream_ptrs) {
  size_t uid = 0;
  hostMemPtr *host_mem = &stream_ptrs->host_mems[uid];
  longMemPtr *long_mem = &stream_ptrs->long_mem;
  deviceMemPtr *dev_mem = &stream_ptrs->dev_mem;
  sycl::queue **stream = &stream_ptrs->cudastream;
  /*
  DPCT1124:36: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(host_mem->f, dev_mem->d_f,
                  sizeof(int32_t) * host_mem->total_n);
  /*
  DPCT1124:37: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(host_mem->p, dev_mem->d_p,
                  sizeof(uint16_t) * host_mem->total_n);
  /*
  DPCT1124:38: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(
      long_mem->long_segs_og_idx, dev_mem->d_long_seg_og,
      dev_mem->buffer_size_long /
          (score_kernel_config.long_seg_cutoff * score_kernel_config.cut_unit) *
          sizeof(seg_t));
  /*
  DPCT1124:39: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(host_mem->long_segs_num, dev_mem->d_long_seg_count,
                  sizeof(unsigned int));
  /*
  DPCT1124:40: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(long_mem->f_long, dev_mem->d_f_long,
                  sizeof(int32_t) * dev_mem->buffer_size_long);
  /*
  DPCT1124:41: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(long_mem->p_long, dev_mem->d_p_long,
                  sizeof(uint16_t) * dev_mem->buffer_size_long);
  sycl_check(**stream);
}

void plmem_async_d2h_short_memcpy(stream_ptr_t *stream_ptrs, size_t uid) {
  hostMemPtr *host_mem = &stream_ptrs->host_mems[uid];
  deviceMemPtr *dev_mem = &stream_ptrs->dev_mem;
  sycl::queue **stream = &stream_ptrs->cudastream;
  /*
  DPCT1124:42: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(host_mem->f, dev_mem->d_f,
                  sizeof(int32_t) * host_mem->total_n);
  /*
  DPCT1124:43: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(host_mem->p, dev_mem->d_p,
                  sizeof(uint16_t) * host_mem->total_n);
  // copy back d_long_seg_count to long_segs_num, this is an accumulative value
  /*
  DPCT1124:44: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(host_mem->long_segs_num, dev_mem->d_long_seg_count,
                  sizeof(unsigned int));
  sycl_check(**stream);
}

void plmem_async_d2h_long_memcpy(stream_ptr_t *stream_ptrs) {
  size_t uid = 0;
  longMemPtr *long_mem = &stream_ptrs->long_mem;
  deviceMemPtr *dev_mem = &stream_ptrs->dev_mem;
  sycl::queue **stream = &stream_ptrs->cudastream;
  /*
  DPCT1124:45: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(
      long_mem->long_segs_og_idx, dev_mem->d_long_seg_og,
      dev_mem->buffer_size_long /
          (score_kernel_config.long_seg_cutoff * score_kernel_config.cut_unit) *
          sizeof(seg_t));
  // cudaMemcpyAsync(&long_mem->total_long_segs_num, dev_mem->d_long_seg_count,
  //                 sizeof(unsigned int), cudaMemcpyDeviceToHost, (*stream));
  /*
  DPCT1124:46: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(long_mem->f_long, dev_mem->d_f_long,
                  sizeof(int32_t) * dev_mem->buffer_size_long);
  /*
  DPCT1124:47: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(long_mem->p_long, dev_mem->d_p_long,
                  sizeof(uint16_t) * dev_mem->buffer_size_long);
  /*
  DPCT1124:48: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(long_mem->total_long_segs_n, dev_mem->d_total_n_long,
                  sizeof(size_t));
  /*
  DPCT1124:49: cudaMemcpyAsync is migrated to asynchronous memcpy API. While the
  origin API might be synchronous, it depends on the type of operand memory, so
  you may need to call wait() on event return by memcpy API to ensure
  synchronization behavior.
  */
  (*stream)->memcpy(long_mem->total_long_segs_num, dev_mem->d_long_seg_count,
                  sizeof(unsigned int));
  sycl_check(**stream);
}

void plmem_sync_d2h_memcpy(hostMemPtr *host_mem, deviceMemPtr *dev_mem) try {
  sycl::queue q(sycl::property::queue::in_order{});
  sycl::queue &q_ct1 = q; 
  q_ct1.memcpy(host_mem->f, dev_mem->d_f, sizeof(int32_t) * host_mem->total_n)
      .wait_and_throw();
  q_ct1.memcpy(host_mem->p, dev_mem->d_p, sizeof(uint16_t) * host_mem->total_n)
      .wait_and_throw();
} catch (const sycl::exception &e) {
  fprintf(stderr, "Error in %s:%i %s(): %s.\n", __FILE__, __LINE__, __func__, e.what());
  fflush(stderr);
  exit(EXIT_FAILURE);
}

//////////////////// Initialization and Cleanup
///< mmpriv.h>////////////////////////
streamSetup_t stream_setup;

#include "../cJSON/cJSON.h"
#include <cmath>

cJSON *plmem_parse_gpu_config(const char filename[]) {
  // read json file to cstring
  char *buffer = 0;
  long length;
  FILE *f = fopen(filename, "rb");

  if (f) {
    fseek(f, 0, SEEK_END);
    length = ftell(f);
    fseek(f, 0, SEEK_SET);
    buffer = (char *)malloc(length);
    if (buffer) {
      fread(buffer, 1, length, f);
    }
    fclose(f);
  }

  if (!buffer) {
    fprintf(stderr, "[Error] fail to open gpu config file %s\n", filename);
    exit(1);
  }

  cJSON *json = cJSON_Parse(buffer);
  if (!json) {
    const char *error_ptr = cJSON_GetErrorPtr();
    if (error_ptr != NULL) {
      fprintf(stderr, "[Error] cJSON error before %s\n", error_ptr);
    }
    exit(1);
  }

  return json;
}

int get_json_int(cJSON *json, const char name[]) {
  cJSON *elt = cJSON_GetObjectItem(json, name);
  if (!cJSON_IsNumber(elt)) {
    fprintf(stderr, "[Error] cJSON error failed to get field %s\n", name);
    exit(1);
  }
  return elt->valueint;
}

void plmem_config_kernels(cJSON *json) {
  cJSON *range_config_json = cJSON_GetObjectItem(json, "range_kernel");
  range_kernel_config.blockdim = get_json_int(range_config_json, "blockdim");
  range_kernel_config.cut_check_anchors =
      get_json_int(range_config_json, "cut_check_anchors");
  range_kernel_config.anchor_per_block =
      get_json_int(range_config_json, "anchor_per_block");

  cJSON *score_config_json = cJSON_GetObjectItem(json, "score_kernel");
  dpct::device_info device_prop;
  dpct::get_device(0).get_device_info(device_prop);
  score_kernel_config.short_blockdim = device_prop.get_max_sub_group_size();
  // Prevent exceeding max supported work group size of 1024
  if (device_prop.get_max_work_group_size() > 1024) {
      score_kernel_config.long_blockdim = 1024;
  } else {
      score_kernel_config.long_blockdim = device_prop.get_max_work_group_size();
  }
  score_kernel_config.mid_blockdim =
      get_json_int(score_config_json, "mid_blockdim");
  score_kernel_config.short_griddim =
      get_json_int(score_config_json, "short_griddim");
  score_kernel_config.long_griddim =
      get_json_int(score_config_json, "long_griddim");
  score_kernel_config.mid_griddim =
      get_json_int(score_config_json, "mid_griddim");
  score_kernel_config.long_seg_cutoff =
      get_json_int(score_config_json, "long_seg_cutoff");
  score_kernel_config.mid_seg_cutoff =
      get_json_int(score_config_json, "mid_seg_cutoff");
  score_kernel_config.cut_unit = range_kernel_config.blockdim;
  score_kernel_config.micro_batch =
      get_json_int(score_config_json, "micro_batch");
  if (score_kernel_config.micro_batch > MAX_MICRO_BATCH) {
    fprintf(
        stderr,
        "[Error: gpu config] score_kernel:micro_batch should be less than %d\n"
        "\t\t or recompile with MAX_MICRO_BATCH=%d",
        MAX_MICRO_BATCH, score_kernel_config.micro_batch);
    exit(1);
  }
}

void plmem_config_stream(size_t *max_range_grid_, size_t *max_num_cut_,
                         size_t max_total_n, size_t max_read, size_t min_n) {
  size_t max_range_grid, max_num_cut;
  max_range_grid =
      (max_total_n - 1) / range_kernel_config.anchor_per_block + 1 + max_read;
  max_num_cut = (max_total_n - 1) / range_kernel_config.blockdim + 1 + max_read;
  *max_range_grid_ = max_range_grid;
  *max_num_cut_ = max_num_cut;

  dpct::device_info prop;
  dpct::get_device(0).get_device_info(prop);
  // sycl_check();

  /*
  DPCT1022:50: There is no exact match between the maxGridSize and the
  max_nd_range size. Verify the correctness of the code.
  */
  if (*max_range_grid_ > prop.get_max_nd_range_size<int *>()[0]) {
    fprintf(stderr, "Invalid memory config!\n");
    exit(1);
  }
}

template <bool is_blooking = false>
void plmem_config_batch(cJSON *json, int *num_stream_, int *min_n_,
                        size_t *max_total_n_, int *max_read_,
                        size_t *long_seg_buffer_size_) {
  if (is_blooking)
    *num_stream_ = 16;
  else
    *num_stream_ = get_json_int(json, "num_streams");

  size_t min_anchors = get_json_int(json, "min_n");
  *min_n_ = min_anchors;

  /* If Use define max_total_n & max_read */
  // FIXME: this is limited by int32max
  cJSON *max_total_n_json = cJSON_GetObjectItem(json, "max_total_n");
  cJSON *max_read_json = cJSON_GetObjectItem(json, "max_read");
  cJSON *long_seg_buffer_size_json =
      cJSON_GetObjectItem(json, "long_seg_buffer_size");
  if (max_total_n_json && max_read_json) {
    *max_total_n_ = (size_t)max_total_n_json->valuedouble;
    *max_read_ = max_read_json->valueint;
    *long_seg_buffer_size_ = long_seg_buffer_size_json->valueint;
    return;
  }

  /* Determine configuration smartly */
  dpct::device_info prop;
  dpct::get_device(0).get_device_info(prop);

  size_t avail_mem_per_stream =
      (prop.get_global_mem_size() / *num_stream_) * 0.9;

  // memory per anchor = (ax + ay + range + f + p) + (start_idx + read_end_idx
  // + cut_start_idx) + cut + long_seg size: F1 = ax + ay + range + f + p; F2
  // = start_idx + read_end_idx + cut_start_idx; F3 = cut; F4 = long_seg
  int F1 = 8 + 8 + 4 + 4 + 2, F2 = 8 + 8 + 8, F3 = 8, F4 = 16;
  // TODO: define these data types

  // max iteration of each block, must be an integer
  // int max_it =
  //     range_kernel_config.anchor_per_block / range_kernel_config.blockdim;
  // int blockdim = range_kernel_config.blockdim;

  // g = max_grid_size
  // g * F2 + g * blockdim * max_it * F1 + max_cut * F3 + max_cut/2 * F4 <
  // mem_per_stream max_cut = g * max_it
  /*
  size_t cost_per_anchor = F1;
  size_t cost_per_grid = F2;
  size_t cost_per_cut = F3 + F4 / 2;
  */
  size_t avg_read_n = get_json_int(json, "avg_read_n");

  /**
   * Assume max_total_n = max_read * avg_read_n
   * max_grid = max_total_n / range.anchor_per_block + max_read
   *          = max_read *( avg_read_n / anchor_per_block + 1)
   * max_cut  = max_total_n / range.blockdim + max_read
   *          = max_read * ( avg_read_n / blockdim + 1)
   * total_mem = max_grid * cost_per_grid + max_cut * cost_per_cut + max_total_n
   * * cost_per_anchor
   */

  float grid_cost_per_read =
      (avg_read_n / (float)range_kernel_config.anchor_per_block + 1) * F2;
  float cut_cost_per_read =
      (avg_read_n / (float)range_kernel_config.blockdim + 1) * (F3 + F4 / 2);
  *max_read_ =
      floor(avail_mem_per_stream /
            (grid_cost_per_read + cut_cost_per_read + F1 * avg_read_n));
  *max_total_n_ = *max_read_ * avg_read_n;
}

// intialize and config kernels for gpu blocking setup
void plmem_initialize(size_t *max_total_n_, int *max_read_, int *min_anchors_) {
#ifndef GPU_CONFIG
  cJSON *json = plmem_parse_gpu_config("gpu/gpu_config.json");
#else
  cJSON *json = plmem_parse_gpu_config(GPU_CONFIG);
#endif
  plmem_config_kernels(json);
  int num_streams;
  size_t buffer_size_long;
  plmem_config_batch<true>(json, &num_streams, min_anchors_, max_total_n_,
                           max_read_, &buffer_size_long);
}

// initialize global variable stream_setup
void plmem_stream_initialize(size_t *max_total_n_, int *max_read_,
                             int *min_anchors_, char *gpu_config_file) {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  // sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  sycl::queue q(sycl::property::queue::in_order{});
  sycl::queue &q_ct1 = q; 

  int num_stream;
  size_t max_anchors_stream, max_range_grid, max_num_cut, long_seg_buffer_size;

  cJSON *json = plmem_parse_gpu_config(gpu_config_file);

  plmem_config_kernels(json);
  size_t gpu_free_mem, gpu_total_mem;
  /*
  DPCT1106:51: 'cudaMemGetInfo' was migrated with the Intel extensions for
  device information which may not be supported by all compilers or runtimes.
  You may need to adjust the code.
  */
  // dpct::get_current_device().get_memory_info(gpu_free_mem, gpu_total_mem);
  plmem_config_batch<false>(json, &num_stream, min_anchors_,
                            &max_anchors_stream, max_read_,
                            &long_seg_buffer_size);
  plmem_config_stream(&max_range_grid, &max_num_cut, max_anchors_stream,
                      *max_read_, *min_anchors_);

  stream_setup.num_stream = num_stream;
  // assert(num_stream > 1);

  stream_setup.streams = new stream_ptr_t[num_stream];
#ifdef DEBUG_PRINT
  fprintf(stderr,
          "[Info] max anchors per stream: %zu, max range grid %zu "
          "max_num_cut %zu long_seg_buffer_size %zu\n",
          max_anchors_stream, max_range_grid, max_num_cut,
          long_seg_buffer_size);
#endif // DEBUG_PRINT

  for (int i = 0; i < num_stream; i++) {
    stream_setup.streams[i].busy = false;
    stream_setup.streams[i].cudastream = dev_ct1.create_queue();
    stream_setup.streams[i].stopevent = new sycl::event();
    stream_setup.streams[i].startevent = new sycl::event();
    stream_setup.streams[i].long_kernel_event = new sycl::event();
    // sycl_check();
    stream_setup.streams[i].dev_mem.buffer_size_long = long_seg_buffer_size;
    // one stream has multiple host mems
    for (int j = 0; j < score_kernel_config.micro_batch; j++) {
      plmem_malloc_host_mem(&stream_setup.streams[i].host_mems[j],
                            max_anchors_stream, max_range_grid,
                            long_seg_buffer_size);
      stream_setup.streams[i].short_kernel_start_event[j] = new sycl::event();
      stream_setup.streams[i].short_kernel_stop_event[j] = new sycl::event();
    }
    // one stream has one long mem and one device mem
    plmem_malloc_long_mem(&stream_setup.streams[i].long_mem,
                          long_seg_buffer_size);
    plmem_malloc_device_mem(&stream_setup.streams[i].dev_mem,
                            max_anchors_stream, max_range_grid, max_num_cut);
    try {
      q_ct1
          .memset(stream_setup.streams[i].dev_mem.d_long_seg_count, 0,
                  sizeof(unsigned int))
          .wait();
      q_ct1
          .memset(stream_setup.streams[i].dev_mem.d_mid_seg_count, 0,
                  sizeof(unsigned int))
          .wait();
      q_ct1
          .memset(stream_setup.streams[i].dev_mem.d_total_n_long, 0,
                  sizeof(size_t))
          .wait();
    } catch (const sycl::exception &e) {
      fprintf(stderr, "Error in %s:%i %s(): %s.\n", __FILE__, __LINE__, __func__, e.what());
      fflush(stderr);
      exit(EXIT_FAILURE);
    }
  }

  /*
  DPCT1106:52: 'cudaMemGetInfo' was migrated with the Intel extensions for
  device information which may not be supported by all compilers or runtimes.
  You may need to adjust the code.
  */
  // dpct::get_current_device().get_memory_info(gpu_free_mem, gpu_total_mem);
#ifdef DEBUG_PRINT
  // fprintf(stderr, "[Info] GPU free mem: %f GB, total mem: %f GB\n",
  //         (float)gpu_free_mem / OneG, (float)gpu_total_mem / OneG);
#endif

  *max_total_n_ = max_anchors_stream * score_kernel_config.micro_batch;
  *max_read_ = *max_read_ * score_kernel_config.micro_batch;

  stream_setup.max_anchors_stream = max_anchors_stream;
  stream_setup.max_range_grid = max_range_grid;
  stream_setup.max_num_cut = max_num_cut;
  stream_setup.long_seg_buffer_size_stream = long_seg_buffer_size;
  // sycl_check();
}

void plmem_stream_cleanup() {
  for (int i = 0; i < stream_setup.num_stream; i++) {
    try {
      dpct::get_current_device().destroy_queue(stream_setup.streams[i].cudastream);
      delete stream_setup.streams[i].stopevent;
      delete stream_setup.streams[i].startevent;
      delete stream_setup.streams[i].long_kernel_event;
    } catch (const sycl::exception &e) {
      fprintf(stderr, "Error in %s:%i %s(): %s.\n", __FILE__, __LINE__, __func__, e.what());
      fflush(stderr);
      exit(EXIT_FAILURE);
    }
    // free multiple host mems
    for (int j = 0; j < score_kernel_config.micro_batch; j++) {
      plmem_free_host_mem(&stream_setup.streams[i].host_mems[j]);
    }
    plmem_free_long_mem(&stream_setup.streams[i].long_mem);
    plmem_free_device_mem(&stream_setup.streams[i].dev_mem);
  }
  delete[] stream_setup.streams;
}
