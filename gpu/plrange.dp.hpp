#ifndef _PLRANGE_CUH_
#define _PLRANGE_CUH_

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "plmem.dp.hpp"
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef __int32_t int32_t;

/* functions declaration */
void plrange_upload_misc(Misc misc);
void plrange_free_misc();
void plrange_async_range_selection(deviceMemPtr *device_mem_ptr,
                                   sycl::queue **stream,
                                   sycl::event *start_event_short);
void plrange_sync_range_selection(deviceMemPtr* dev_mem, Misc misc);

extern range_kernel_config_t range_kernel_config;


#ifdef __cplusplus
}
#endif

#endif  // _PLRANGE_CUH_
