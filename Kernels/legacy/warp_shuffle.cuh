#ifndef WARP_SHUFFLE_CUH
#define WARP_SHUFFLE_CUH

// Legacy compile-time-D BLOCK_SIZE tuning for the archived Element path.
// BLOCK_SIZE scales down with D to stay within 164KB shared memory limit
// (A100 maximum with cudaFuncSetAttribute opt-in).
// Shared memory: (BLOCK_SIZE + BLOCK_SIZE/32 + 1) * 2 * D * 4 bytes
//   D=16:  (512 + 17) * 128  = ~66KB  ✓
//   D=64:  (256 + 9)  * 512  = ~133KB ✓
//   D=256: (64  + 3)  * 2048 = ~134KB ✓
//   D=512: (32  + 2)  * 4096 = ~136KB ✓
#ifdef BLOCK_SIZE
#undef BLOCK_SIZE
#endif

#if D <= 16
#define BLOCK_SIZE 512
#elif D <= 64
#define BLOCK_SIZE 256
#elif D <= 256
#define BLOCK_SIZE 64
#else
#define BLOCK_SIZE 32
#endif

#endif // WARP_SHUFFLE_CUH
