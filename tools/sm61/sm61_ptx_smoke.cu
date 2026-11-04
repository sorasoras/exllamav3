// Scratch compile check for the sm_61 gates in ptx.cuh / arch.cuh. Not part of the build.
#include "exllamav3_ext/ptx.cuh"

__global__ void smoke(int* out, int* lock)
{
    FragA fa;
    FragB fb;
    FragC fc;
    FragC_h fch;
    ptx_mma_m8n8k4(*reinterpret_cast<Vec<half2, 2>*>(&fa), *reinterpret_cast<Vec<half2, 2>*>(&fb),
                   *reinterpret_cast<Vec<float, 8>*>(&fc));
    ptx_mma_m16n8k16(fa, fb, fc);
    ptx_mma_m16n8k16(fa, fb, fch);
    ldsm4(fa, lock);
    barrier_acquire(lock, 1);
    barrier_release(lock, 1, false);
    EXL3_NANOSLEEP(32);
    out[0] = (int) ldg_acquire_sys_u32(reinterpret_cast<uint32_t*>(lock));
    stg_release_sys_u32(reinterpret_cast<uint32_t*>(lock), 1);
    out[1] = EXL3_SM61 * 10 + EXL3_SM75;
}
