// CUDA-C includes
#include <cuda.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_math.h>

#include <assert.h>
#include <stdio.h>

#include "kernel.cuh"

static struct cudaGraphicsResource *pixelBuffer, *texture0, *texture1;

typedef texture<uchar4, cudaTextureType2D, cudaReadModeElementType> inTexture2D;
inTexture2D inTexture0, inTexture1;

__device__
float4 sampleVolume(float3 pos)
{
    // TODO: Sample from volume texture

    if ((pos.x - .5f) * (pos.x - .5f) + (pos.z - .5f) * (pos.z - .5f) < .25) {
       return make_float4(1.f, 1.f, 1.f, 0.01f);
    }
    else {
       return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

__global__
void kernel(void *buffer, int width, int height, struct slice_params slice)
{
    const float stepSize = 0.02f;
    const int maxIters = 200;

    uchar4 *pixels = (uchar4*) buffer;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int x = index % width;
    unsigned int y = index / width;

    if (index < width * height) {
        // Roughly adapted from http://graphicsrunner.blogspot.com/2009/01/volume-rendering-101.html

        uchar4 sample0 = tex2D( inTexture0, x, y ),
               sample1 = tex2D( inTexture1, x, y );

        pixels[index] = sample0;

//        if (sample0.w < 0xff || sample1.w < 0xff) {
//            pixels[index] = make_uchar4(1, 0, 0, 1);
//            return;
//        }

//        float3 front = make_float3(sample0.x / 255.f, sample0.y / 255.f, sample0.z / 255.f),
//               back  = make_float3(sample1.x / 255.f, sample1.y / 255.f, sample1.z / 255.f),
//               dist  = back - front;

//        float length = sqrt(dist.x * dist.x + dist.y * dist.y + dist.z * dist.z);
//        if (length < 0.001f) {
//            pixels[index] = make_uchar4(0, 0, 0, 0);
//            return;
//        }

//        float4 accum = make_float4(0.f, 0.f, 0.f, 0.f);

//        float3 ray   = dist / length,
//               step  = ray * stepSize,
//               pos   = front;

//        for (int i = 0; i < maxIters; ++i) {
//            pos += step;
//            if (pos.x > 1.0f || pos.x < 0.0f
//                    || pos.y > 1.0f || pos.y < 0.0f
//                    || pos.z > 1.0f || pos.z < 0.0f) {
//                break;
//            }

//            float4 vox = sampleVolume(pos);

//            accum.x += vox.x * vox.w * (1.f - accum.w);
//            accum.y += vox.y * vox.w * (1.f - accum.w);
//            accum.z += vox.z * vox.w * (1.f - accum.w);
//            accum.w += vox.w * (1.f - accum.w);

//            if (accum.w > .95f) {
//                break;
//            }
//        }
//        // accum = make_float4(fabs(ray.x), fabs(ray.y), fabs(ray.z), 1.0f);

//        accum.x = fminf(accum.x, 1.f);
//        accum.y = fminf(accum.y, 1.f);
//        accum.z = fminf(accum.z, 1.f);
//        accum.w = fminf(accum.w, 1.f);

//        pixels[index] = make_uchar4(accum.x * 0xff, accum.y * 0xff, accum.z * 0xff, accum.w * 0xff);
    }
}

void initCuda() {
    cudaGLSetGLDevice( gpuGetMaxGflopsDeviceId() );

    SDK_CHECK_ERROR_GL();
}

void registerCudaResources(GLuint input0, GLuint input1, GLuint output) {
    assert(input0);
    assert(input1);
    assert(output);

    checkCudaErrors( cudaGraphicsGLRegisterImage(&texture0, input0, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly) );
    checkCudaErrors( cudaGraphicsGLRegisterImage(&texture1, input1, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly) );
    checkCudaErrors( cudaGraphicsGLRegisterBuffer(&pixelBuffer, output, cudaGraphicsRegisterFlagsWriteDiscard) );
}

void runCuda(int width, int height, struct slice_params *slice) {
    cudaGraphicsResource_t resources[3] = { texture0, texture1, pixelBuffer };

    checkCudaErrors( cudaGraphicsMapResources(3, resources) );

    struct cudaArray *array0;
    checkCudaErrors( cudaGraphicsSubResourceGetMappedArray(&array0, texture0, 0, 0) );
    checkCudaErrors( cudaBindTextureToArray(inTexture0, array0) );

    struct cudaArray *array1;
    checkCudaErrors( cudaGraphicsSubResourceGetMappedArray(&array1, texture1, 0, 0) );
    checkCudaErrors( cudaBindTextureToArray(inTexture1, array1) );

    void *devBuffer;
    size_t bufferSize;
    checkCudaErrors( cudaGraphicsResourceGetMappedPointer(&devBuffer, &bufferSize, pixelBuffer) );

    // http://www.drdobbs.com/architecture-and-design/cuda-supercomputing-for-the-masses-part/222600097
    int nThreads = 256, totalThreads = width * height,
        nBlocks = totalThreads / nThreads;
    nBlocks += ((totalThreads % nThreads) > 0 ) ? 1 : 0;

    struct slice_params kernParams;
    if (slice) {
        kernParams = *slice;
    }
    else {
        kernParams.type = -1;
    }
    kernel<<< nBlocks, nThreads >>>(devBuffer, width, height, kernParams);

    checkCudaErrors( cudaUnbindTexture(inTexture0) );

    checkCudaErrors( cudaGraphicsUnmapResources(3, resources) );
}
