#ifndef _PLSCORE_CUH_
#define _PLSCORE_CUH_

#include <sycl/sycl.hpp>
#include "../mmpriv.h"
#include "plmem.hpp"

#ifdef __cplusplus
extern "C" {
#endif

#define MM_QSPAN 15

void plscore_upload_misc(Misc misc, sycl::queue &q_ct1);
void plscore_free_misc(sycl::queue &q_ct1);
void plscore_async_naive_forward_dp(deviceMemPtr *dev_mem,
                                    sycl::queue **stream);
void plscore_async_short_mid_forward_dp(deviceMemPtr *dev_mem,
                                        sycl::queue **stream,
                                        sycl::event *stop_event_short);
void plscore_async_long_forward_dp(deviceMemPtr *dev_mem,
                                   sycl::queue **stream,
                                   sycl::event **event);

extern score_kernel_config_t score_kernel_config;

#ifdef __cplusplus
}
#endif

#endif // _PLSCORE_CUH_
