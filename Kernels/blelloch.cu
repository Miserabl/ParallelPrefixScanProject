/**
 * blelloch.cu  —  Blelloch up-sweep / down-sweep prefix scan (scalar edition)
 *
 * Each thread handles ONE float pair (a_val, b_val).  No D-loop.
 *
 * Algorithm:
 *  1. Up-sweep (reduce):   build a partial-sum tree in shared memory.
 *  2. Set root to identity, save the total.
 *  3. Down-sweep (distribute): push exclusive prefixes down the tree.
 *  4. Convert exclusive → inclusive by combining each element with its original.
 *
 * Requirement: n must be a power of 2.  For phase1 n=CHUNK_SIZE=1024 ✓.
 *   For phase2 n=num_chunks which is always a power of 2 for the benchmark's
 *   L values (all powers of 2 and CHUNK_SIZE=1024).
 *
 * Shared memory layout (same as common.cuh):
 *   s_a = shmem[0 .. CHUNK_SIZE)
 *   s_b = shmem[CHUNK_SIZE .. 2*CHUNK_SIZE)
 *   (aux_a / aux_b zones not used by Blelloch)
 *
 * NOTE: No conflict-free addressing is applied in this scalar rewrite.
 *   The performance-limiting factor was the serial D-loop in combine(), not
 *   bank conflicts.  Adding it back is a straightforward future optimization.
 */

#include "common.cuh"

struct Blelloch {
    static __device__ __forceinline__ int cf_index(int idx) {
        return idx + (idx >> LOG_NUM_BANKS);
    }

    static __device__ void block_scan(
        float* __restrict__ shmem, int n,
        float* s_tot_a, float* s_tot_b)
    {
        float* compact_a = shmem;
        float* compact_b = shmem + CHUNK_SIZE;
        float* s_a = shmem + 2 * CHUNK_SIZE;
        float* s_b = s_a + BLELLOCH_PADDED_CHUNK;

        const int tid = threadIdx.x;

        // Save originals so we can convert exclusive→inclusive at the end.
        float orig_a = (tid < n) ? compact_a[tid] : 1.0f;
        float orig_b = (tid < n) ? compact_b[tid] : 0.0f;

        // Copy the compact caller-owned layout into a padded workspace so the
        // tree traversal no longer maps power-of-two strides onto the same
        // shared-memory banks.
        if (tid < n) {
            const int p = cf_index(tid);
            s_a[p] = compact_a[tid];
            s_b[p] = compact_b[tid];
        }
        __syncthreads();

        // ----------------------------------------------------------------
        // Up-sweep (reduce phase)
        // After this loop, s_a[n-1] / s_b[n-1] hold the total of all elements.
        // ----------------------------------------------------------------
        int offset = 1;
        for (int d = n >> 1; d > 0; d >>= 1) {
            __syncthreads();
            if (tid < d) {
                int ai = cf_index(offset * (2 * tid + 1) - 1);
                int bi = cf_index(offset * (2 * tid + 2) - 1);
                // s[bi] = combine(left=s[ai], right=s[bi])
                float new_a = s_a[bi] * s_a[ai];
                float new_b = s_a[bi] * s_b[ai] + s_b[bi];
                s_a[bi] = new_a;
                s_b[bi] = new_b;
            }
            offset <<= 1;
        }
        __syncthreads();

        // Save total; set root to identity for exclusive scan.
        if (tid == 0) {
            const int root = cf_index(n - 1);
            *s_tot_a = s_a[root];
            *s_tot_b = s_b[root];
            s_a[root] = 1.0f;
            s_b[root] = 0.0f;
        }
        __syncthreads();

        // ----------------------------------------------------------------
        // Down-sweep (distribute phase)
        // ----------------------------------------------------------------
        for (int d = 1; d < n; d <<= 1) {
            offset >>= 1;
            __syncthreads();
            if (tid < d) {
                int ai = cf_index(offset * (2 * tid + 1) - 1);
                int bi = cf_index(offset * (2 * tid + 2) - 1);

                // Left child gets the parent's exclusive prefix (s[bi] before swap).
                // Right child gets combine(parent_prefix, left_subtree_total).
                float t_a = s_a[ai], t_b = s_b[ai];   // save left-child value
                s_a[ai] = s_a[bi];                     // left child ← parent prefix
                s_b[ai] = s_b[bi];
                // s[bi] = combine(left=parent_prefix, right=left_subtree_total)
                //       = combine(old s[bi], t)
                float new_a = t_a * s_a[ai];           // t_a * old_s_a[bi]
                float new_b = t_a * s_b[ai] + t_b;
                s_a[bi] = new_a;
                s_b[bi] = new_b;
            }
        }
        __syncthreads();

        // ----------------------------------------------------------------
        // Convert exclusive → inclusive:  result[i] = combine(excl[i], orig[i])
        // Read first, barrier, then write to avoid WAR hazard.
        // ----------------------------------------------------------------
        float new_a = 0.0f, new_b = 0.0f;
        if (tid < n) {
            const int p = cf_index(tid);
            new_a = orig_a * s_a[p];
            new_b = orig_a * s_b[p] + orig_b;
            compact_a[tid] = new_a;
            compact_b[tid] = new_b;
        }
        __syncthreads();
    }
};
