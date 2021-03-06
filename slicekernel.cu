#include "slicekernel.cuh"
#include "params.h"
#include "kernel.cuh"

#include <iostream>
using std::cout;
using std::endl;

std::ostream& operator<<(std::ostream& os, const SliceParameters p)
{
    os << "dx: " << p.dx << ", dy: " << p.dy << ", dz: " << p.dz
       << "theta: " << p.theta << ", phi: " << p.phi << ", psi: " << p.psi;
    return os;
}

std::ostream& operator<<(std::ostream& os, const BufferParameters p)
{
    os << "height: " << p.height << ", width: " << p.width;
    return os;
}

extern texture<unsigned char, cudaTextureType3D, cudaReadModeNormalizedFloat> texVolume;

void invoke_slice_kernel(float *buffer, BufferParameters bp, SliceParameters sp, canonicalOrientation c)
{

    cout << "pretending to invoke kernel" << endl;

    float* buffer_dev;

    checkCudaErrors( cudaMalloc( &buffer_dev, bp.height*bp.width*sizeof(float)) );

    int dim = 1024;

    dim3 grids(dim/16, dim/16);
    dim3 threads(16,16);

    cout << "launching kernel" << endl;

    cout << "kernel parameters: " << sp << endl << bp << endl;


    slice_kernel<<<grids,threads>>>( buffer_dev, bp, sp, c );
    cout << "kernel launched" << endl;

    checkCudaErrors( cudaMemcpy( buffer, buffer_dev, bp.height*bp.width*sizeof(float), cudaMemcpyDeviceToHost) );

    checkCudaErrors( cudaFree( buffer_dev ) );

}
__global__
void slice_kernel(float *buffer, BufferParameters bp, SliceParameters sp, canonicalOrientation c)
{
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
//    int offset = i + j * blockDim.x * gridDim.x; // ???

    int offset = j*bp.height+i;

    if(j<bp.height && i<bp.width){

            float3 pos;

            pos.z = 0;
            pos.y = ((float)j)/((float)bp.height);
            pos.x = ((float)i)/((float)bp.width);

            pos.x += sp.dx;
            pos.y += sp.dy;
            pos.z += sp.dz;

            float sample = tex3D(texVolume, pos.x, pos.y, pos.z);

            buffer[offset] = sample;

//          buffer[offset] = (pos.x + pos.y)/2;


    }


}
