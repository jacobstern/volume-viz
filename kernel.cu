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
    if (pos.x > 1.f || pos.y > 1.f || pos.z > 1.f
            || pos.x < 0.f || pos.y < 0.f || pos.z < 0.f) {
        return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    if ((pos.x - .5f) * (pos.x - .5f) + (pos.z - .5f) * (pos.z - .5f) < .25) {
       return make_float4(1.f, 1.f, 1.f, 0.01f);
    }
    else {
       return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
}

#define MAX_STEPS 64 // TODO: Turn up to 128 for Sunlab machines

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

void runCuda(int width, int height, struct slice_params slice, struct camera_params camera) {
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

    kernel<<< blockDims, blockSize, 512 * sizeof(float) >>>(devBuffer, width, height, slice, camera);

    checkCudaErrors( cudaUnbindTexture(inTexture0) );

    checkCudaErrors( cudaGraphicsUnmapResources(3, resources) );
}
