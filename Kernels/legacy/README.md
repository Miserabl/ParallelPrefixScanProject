# Legacy Kernel Helpers

These files belong to the older compile-time-`D` / `Element`-based scan path.

The active benchmark and profiling flow does not use them anymore. The current
runtime benchmark path is built from:

- `common.cuh`
- `warp_shuffle.cu`
- `blelloch.cu`
- `hillis_steele.cu`
- `chunked_hierarchical_recursive.cuh`
- `benchmark.cu`
- `profile_driver.cu`

This folder keeps the older probes, one-off tests, and compile-time headers for
reference only. They are not maintained as part of the main methodology flow.
