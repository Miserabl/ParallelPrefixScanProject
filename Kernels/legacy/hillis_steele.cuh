#ifndef HILLIS_STEELE_CUH
#define HILLIS_STEELE_CUH

// Legacy compile-time-D BLOCK_SIZE tuning for the archived Element path.
// BLOCK_SIZE scales down with D to stay within 164KB shared memory limit
// (A100 maximum with cudaFuncSetAttribute opt-in).
// Hillis-Steele needs two full-size buffers for double-buffering (in + out).
// Total shared memory = 2 * BLOCK_SIZE * 2 * D * 4 bytes
//   D=16:  2 * 512 * 128  = 131KB ✓
//   D=64:  2 * 128 * 512  = 131KB ✓
//   D=256: 2 * 32  * 2048 = 131KB ✓
//   D=512: 2 * 16  * 4096 = 131KB ✓
#ifdef BLOCK_SIZE
#undef BLOCK_SIZE
#endif

#if D <= 16
#define BLOCK_SIZE 512
#elif D <= 64
#define BLOCK_SIZE 128
#elif D <= 256
#define BLOCK_SIZE 32
#else
#define BLOCK_SIZE 16
#endif

#endif // HILLIS_STEELE_CUH
