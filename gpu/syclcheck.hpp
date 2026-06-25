#include <sycl/sycl.hpp>
#include <cstdio>
#include <exception>

// The asynchronous error handler matching the sycl::async_handler signature
inline void sycl_async_handler(sycl::exception_list exceptions_list) {
    for (std::exception_ptr const& e : exceptions_list) {
        try {
            std::rethrow_exception(e);
        }
        catch (sycl::exception const& se) {
            std::fprintf(stderr, "\n[SYCL Async Error]: %s\n", se.what());
            std::fprintf(stderr, "Category: %s\n", se.category().name());
            std::fprintf(stderr, "Error Code: %d\n\n", se.code().value());
        }
        catch (std::exception const& re) {
            std::fprintf(stderr, "\n[Standard Async Error]: %s\n\n", re.what());
        }
    }
}