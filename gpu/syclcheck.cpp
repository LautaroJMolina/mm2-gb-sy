#include <sycl/sycl.hpp>
#include <sycl/sycl.hpp>

namespace sycl = acpp::sycl;

inline void sycl_check(sycl::queue q) {
  try {
    q.throw_asynchronous();
  } catch (const sycl::exception &e) {
    fprintf(stderr, "Error in %s:%i %s(): %s.\n", __FILE__, __LINE__, __func__, e.what());
    fflush(stderr);
    exit(EXIT_FAILURE);
  }
}