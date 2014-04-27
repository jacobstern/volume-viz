// CUDA-C includes
#include <cuda.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_math.h>

#include <assert.h>
#include <stdio.h>

#include "kernel.cuh"
#include "cuPrintf.cu"
#include "assert.h"
#include "params.h"

#include <iostream>

using std::cout;
using std::endl;

static struct cudaGraphicsResource *pixelBuffer, *texture0, *texture1;

typedef texture<uchar4, cudaTextureType2D, cudaReadModeElementType> inTexture2D;
inTexture2D inTexture0, inTexture1;

// volumetric texture
texture<byte,3,cudaReadModeElementType> texVolume;


__device__
float vectorLength(float3 vec)
{
    return sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

__device__
float blockMin(float shared[], int idx, int upper, float poll)
{
    shared[idx] = poll;

    __syncthreads();

    float min = shared[idx];

    for (int i = 0; i < upper; i++) {
        float compare = shared[i];
        if (compare < min) {
            min = compare;
        }
    }

    return min;
}

__device__
float4 sampleVolume(float3 pos)
{
    // TODO: Sample from volume texture
//    if (pos.x > 1.f || pos.y > 1.f || pos.z > 1.f
//            || pos.x < 0.f || pos.y < 0.f || pos.z < 0.f) {
//        return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
//    }

//    if ((pos.x - .5f) * (pos.x - .5f) + (pos.z - .5f) * (pos.z - .5f) < .25) {
//       return make_float4(1.f, 1.f, 1.f, 0.01f);
//    }
//    else {
//       return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
//    }

    if(pos.x < 0.0 || pos.x > 1.0 || pos.y < 0.0 || pos.y > 1.0 || pos.z < 0.0 || pos.z > 1.0){
        return make_float4(1.0, 0, 0, 1.0);
    }

    float cof = STEP_SIZE*STEP_SIZE*STEP_SIZE/23;
    float x = pos.x*cof;
    float y = pos.y*cof;
    float z = pos.z*cof;

    byte sample = tex3D(texVolume, x, y, z);

    if(sample == 1){
        return make_float4(0.5, 0.0, 0, 1);

    }else if(sample == 2){
        return make_float4(0.0, 0.5, 0.0, 1);

    }else if(sample == 3){
        return make_float4(0.0, 0.0, 0.5, 1);

    }else if(sample == 4){
        return make_float4(0.0, 0.5, 0.5, 1);

    }else if(sample == 5){
        return make_float4(0.5, 0.0, 0.5, 1);

    }else{
        return make_float4(sample, sample, sample, 0.05)/8;
    }

//    return make_float4(sample, sample, sample, 0.05)/122;
}

//#define MAX_STEPS 63 // TODO: Turn up to 128 for Sunlab machines
#define MAX_STEPS 64

__global__
void kernel(void *buffer,
            int width,
            int height,
            struct slice_params slice,
            struct camera_params camera )
{
    extern __shared__ unsigned char sharedMemory[];
    uchar4 *pixels = (uchar4*) buffer;

    int x = blockIdx.x * blockDim.x + threadIdx.x,
        y = blockIdx.y * blockDim.y + threadIdx.y;

    int index = y * width + x;

    int slabUpperX = min( (blockIdx.x + 1) * blockDim.x, width ),
        slabUpperY = min( (blockIdx.y + 1) * blockDim.y, height),
        slabLowerX = blockIdx.x * blockDim.x,
        slabLowerY = blockIdx.y * blockDim.y,
        slabWidth  = slabUpperX - slabLowerX,
        slabHeight = slabUpperY - slabLowerY;

    int slabY = y - slabLowerY,
        slabX = x - slabLowerX,
        slabIndex = slabY * slabWidth + slabX,
        slabUpper = slabHeight * slabWidth;

    if (index < width * height) {
        uchar4 sample0 = tex2D( inTexture0, x, y ),
               sample1 = tex2D( inTexture1, x, y );

        float3 front = make_float3(sample0.x / 255.f, sample0.y / 255.f, sample0.z / 255.f),
               back  = make_float3(sample1.x / 255.f, sample1.y / 255.f, sample1.z / 255.f);

        float3  camPos  = make_float3( camera.origin[0], camera.origin[1], camera.origin[2] ),
                camDist = front - camPos;
        float camLength = vectorLength(camDist);

        // Note: all threads in block where index < width * height
        // MUST execute this function. Also NB this includes a thread barrier.
        // float desired = blockMin((float*)sharedMemory, slabIndex, slabUpper, camLength);

        // Update front based on desired distance from camera
        // front = camPos + (camDist / camLength) * desired;

        float3 dist = back - front;
        float length = vectorLength(dist);

        if (length < 0.001f) {
            pixels[index] = make_uchar4(0, 0, 0, 0);
            return;
        }

        float4 accum = make_float4(0.f, 0.f, 0.f, 0.f);

//         pixels[index] = make_uchar4(camLength / 8.f * 0xff, camLength / 8.f * 0xff, camLength / 8.f * 0xff, 0xff );
//         pixels[index] = make_uchar4(desired / 8.f * 0xff, desired / 8.f * 0xff, desired / 8.f * 0xff, 0xff );


        float3 ray   = dist / length,
               step  = ray * sqrtf(3) / MAX_STEPS,
               pos   = front;

        for (int i = 0; i < MAX_STEPS; ++i) {
            pos += step;
            if (pos.x > 1.0f || pos.x < 0.0f
                    || pos.y > 1.0f || pos.y < 0.0f
                    || pos.z > 1.0f || pos.z < 0.0f) {
                break;
            }

            float4 vox = sampleVolume(pos);

            if (vox.w > 1e-6) {
                accum.x += vox.x * vox.w * (1.f - accum.w);
                accum.y += vox.y * vox.w * (1.f - accum.w);
                accum.z += vox.z * vox.w * (1.f - accum.w);
                accum.w += vox.w * (1.f - accum.w);

                if (accum.w > .95f) {
                    break;
                }
            }
        }
        // accum = make_float4(fabs(ray.x), fabs(ray.y), fabs(ray.z), 1.0f);

        accum.x = fminf(accum.x, 1.f);
        accum.y = fminf(accum.y, 1.f);
        accum.z = fminf(accum.z, 1.f);
        accum.w = fminf(accum.w, 1.f);

        pixels[index] = make_uchar4(accum.x * 0xff, accum.y * 0xff, accum.z * 0xff, accum.w * 0xff);
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

void runCuda(int width, int height, struct slice_params slice, struct camera_params camera,
             cudaArray* volumeArray) {
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

    // For convenience, greedily chunk the screen into 256-pixel squares
    dim3 blockSize(16, 16),
         blockDims(width / blockSize.x, height / blockSize.y);
    if (width % blockSize.x)
        ++blockDims.x;
    if (height % blockSize.y)
        ++blockDims.y;

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar1>();
    checkCudaErrors( cudaBindTextureToArray(texVolume, volumeArray, channelDesc));

    size_t sharedMemSize = ( MAX_STEPS + 1 ) * blockSize.x * blockSize.y * sizeof(unsigned char);
    kernel<<< blockDims, blockSize, sharedMemSize >>>(devBuffer, width, height, slice, camera);

    checkCudaErrors( cudaUnbindTexture(inTexture0) );

    checkCudaErrors( cudaGraphicsUnmapResources(3, resources) );
}

// load volumetric texture into the GPU
void cudaLoadVolume(byte* texels, size_t size, Vector3 dims,
                    cudaArray** volumeArray) {

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar1>();

    cout << "mallocing texture array" << endl;
    cudaArray* tex_array;
    cudaExtent extent;
    extent.width = dims.x;
    extent.height = dims.y;
    extent.depth = dims.z;
    checkCudaErrors( cudaMalloc3DArray(&tex_array,&channelDesc,extent,0) );
    cout << "texture array has been malloced" << endl;

    cout << "size: " << size << endl;

    cout << "memcopying texture array" << endl;
    assert(texels);
    assert(tex_array);
    assert(size);

    int width = dims.x;
    int height = dims.y;
    int depth = dims.z;

    cudaMemcpy3DParms params = {0};
    params.srcPtr.pitch = sizeof(byte)*width;
    params.srcPtr.ptr = texels;
    params.srcPtr.xsize = width;
    params.srcPtr.ysize = height;

    params.srcPos.x = 0;
    params.srcPos.y = 0;
    params.srcPos.z = 0;

    params.dstArray = tex_array;

    params.dstPos.x = 0;
    params.dstPos.y = 0;
    params.dstPos.z = 0;

    params.extent.width = width;
    params.extent.depth = depth;
    params.extent.height = height;

    params.kind = cudaMemcpyHostToDevice;

    checkCudaErrors( cudaMemcpy3D(&params) );

    checkCudaErrors( cudaBindTextureToArray(texVolume, tex_array, channelDesc));

    cout << "texture array has been memcopied" << endl;


    cout << "copying data back for testing" << endl;

    // Sanity check: Copy texture back
    byte* back_texels = new byte[size];
    memset(back_texels, '\0', size);


    cudaMemcpy3DParms back_params = {0};

    back_params.dstPtr.pitch = sizeof(byte) * width;
    back_params.dstPtr.ptr = back_texels;
    back_params.dstPtr.xsize = width;
    back_params.dstPtr.ysize = height;

    back_params.srcPos.x = 0;
    back_params.srcPos.y = 0;
    back_params.srcPos.z = 0;

    back_params.srcArray = tex_array;

    back_params.dstPos.x = 0;
    back_params.dstPos.y = 0;
    back_params.dstPos.z = 0;

    back_params.extent.width = width;
    back_params.extent.depth = depth;
    back_params.extent.height = height;

    back_params.kind = cudaMemcpyDeviceToHost;

    cout << "invoking" << endl;
    checkCudaErrors( cudaMemcpy3D(&back_params) );


    // TODO: Copy back for sanity check!
    for(int i=0; i<size; i++){
        if(texels[i] != back_texels[i]){
            printf("i: %d, texels: %u, back_texels: %u\n", i, texels[i], back_texels[i]);
            assert(false);


        }
    }

    *volumeArray = tex_array;

    cout << "data has been copied back for testing" << endl;


}






