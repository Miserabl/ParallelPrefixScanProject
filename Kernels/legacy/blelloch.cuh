#ifndef BLELLOCH_CUH
#define BLELLOCH_CUH

// Legacy compile-time-D BLOCK_SIZE tuning for the archived Element path.
// Max L per block for Blelloch, limited by 96KB shared memory.
// Blelloch allocates (L + conflict_free_padding) * sizeof(Element) shared memory.
// sizeof(Element) = 2*D*4 bytes. Must be power-of-2 for up/down sweep.
//   D=16:  128B/elem  → 96KB/128  = 750 → max L=512
//   D=64:  512B/elem  → 96KB/512  = 192 → max L=128
//   D=256: 2048B/elem → 96KB/2048 = 47  → max L=32
//   D=512: 4096B/elem → 96KB/4096 = 23  → max L=16
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

#endif // BLELLOCH_CUH
