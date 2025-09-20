#ifndef _PLSCORE_CUH_
#define _PLSCORE_CUH_

#ifdef DEBUG_PRINT
#define DPCT_PROFILING_ENABLED
#endif

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "../mmpriv.h"
#include "plmem.dp.hpp"

#ifdef __cplusplus
extern "C" {
#endif

#define MM_QSPAN 15

void plscore_upload_misc(Misc misc);
void plscore_async_naive_forward_dp(deviceMemPtr *dev_mem,
                                    dpct::queue_ptr *stream);
void plscore_async_short_mid_forward_dp(deviceMemPtr *dev_mem,
                                        dpct::queue_ptr *stream,
                                        sycl::event *stop_event_short);
void plscore_async_long_forward_dp(deviceMemPtr *dev_mem,
                                   dpct::queue_ptr *stream);

extern score_kernel_config_t score_kernel_config;

#ifdef __cplusplus
}
#endif

#endif // _PLSCORE_CUH_
