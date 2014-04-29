#include "slice-kernel.cuh"
#include "window.h"
#include "kernel.h"

__global__
void slice_kernel(void *buf, BufferParameters bp, SliceParameters sp, canonicalOrientation c)
{

    for(int j=0; j<bp.y; j++){
        for(int i=0; i<bp.x; i++){

            // TODO: Incorporate sliceParameters
            int offset = j*bp.y + i;

            // TODO: Bind correct texture (or just leave it in)

            // note: horizontal only

            float3 pos;
            pos.x = (float(i)/((float)bp.x));
            pos.y = (float(j)/((float)bp.y));
            pos.z = 0;

            pos.x += sp.dx;
            pos.y += sp.dy;
            pos.z += sp.dz;

            float sample = tex3(pos.x, pos.y, pos.z);

            buf[offset] = sample;




        }
    }




}
