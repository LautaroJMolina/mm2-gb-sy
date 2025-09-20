#ifndef __PLANALYZE_H__
#define __PLANALYZE_H__

/* Implement kernel performance analysis that requires extra device
 * synchornization. disabled unless DEBUG_LEVEL is set to analyze.
 * Enable individual verbose prints in planalyze.cu 
 */

#ifdef DEBUG_PRINT
#define DPCT_PROFILING_ENABLED
#endif

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "plchain.h"
#include "plutils.h"
#include "plmem.dp.hpp"
#include "plscore.dp.hpp"

#ifdef DEBUG_CHECK
void planalyze_short_kernel(stream_ptr_t stream, int uid, float throughput[]);
void planalyze_long_kernel(stream_ptr_t stream, float* throughput);
#endif // DEBUG_CHECK

#endif // __PLANALYZE_H__