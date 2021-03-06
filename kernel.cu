// CUDA-C includes
#include <cuda.h>
#include <cuda_gl_interop.h>

#include <helper_cuda.h>
#include <helper_cuda_gl.h>
#include <helper_math.h>

#include <assert.h>
#include <stdio.h>

#include "kernel.cuh"
#include "assert.h"
#include "params.h"

#include <iostream>

#include "implicit.cu"

using std::cout;
using std::endl;


#define CACHE_DEPTH             32
#define CACHE_DEPTH_MINUS_TWOF  30.f

#define DIRECT_FACTOR           0.3f
#define ONE_MINUS_DIRECT_FACTOR 0.7f

#define BLOCK_WIDTH             16
#define BLOCK_HEIGHT            16

#define SQRT_3 1.73205081f

static struct cudaGraphicsResource *pixelBuffer, *texture0, *texture1;

typedef texture<uchar4, cudaTextureType2D, cudaReadModeElementType> inTexture2D;
inTexture2D inTexture0, inTexture1;

typedef texture<float4, cudaTextureType1D, cudaReadModeElementType> inTexture1D;
inTexture1D transferTexture;

typedef unsigned char uchar;

// volumetric texture
texture<unsigned char, cudaTextureType3D, cudaReadModeNormalizedFloat> texVolume;

cudaArray *devVolume   = 0;
float     *devTransfer = 0;

dim3      volumeDim;

__device__
float vectorLength(float3 vec)
{
    return sqrtf(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z);
}

__device__
float ucharToFloat(unsigned char c)
{
    return c / 255.f;
}

__device__
bool boundsCheck(float3 pos)
{
    return pos.x < 1.0f && pos.x >= 0.0f
            && pos.y < 1.0f && pos.y >= 0.0f
            && pos.z < 1.0f && pos.z >= 0.0f;
}

__device__ unsigned char getVoxel(unsigned char sharedMemory[], dim3 cacheIdx, dim3 cacheDim, int offset) {
    if (offset < 0 || offset + 1 > CACHE_DEPTH)
        return 0x00;

    return sharedMemory[ offset * cacheDim.x * cacheDim.y + cacheIdx.y * cacheDim.x + cacheIdx.x ];
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
unsigned char sample(float3 pos) {
    if ( boundsCheck(pos) )
        return 0xff * tex3D(texVolume, pos.x, pos.y, pos.z);
    else
        return 0x00;
}

__device__
float4 blend(float4 src, float4 dst) {
    float4 ret;
    float blendFactor = src.w * (1.f - dst.w);

    ret.x = dst.x + src.x * blendFactor;
    ret.y = dst.y + src.y * blendFactor;
    ret.z = dst.z + src.z * blendFactor;
    ret.w = dst.w +         blendFactor;

    return ret;
}

__device__
float4 transferFunction(uchar sampled) {
    return tex1Dfetch( transferTexture, sampled );
}

__device__
void rayMarch(unsigned char cache[],
              dim3   cacheIdx,
              dim3   cacheDim,
              float3 origin,
              float3 direction,
              float3 scale) {
    float3 pos  = origin,
           step = direction;

    for (int i = 0; i < CACHE_DEPTH; ++i) {
        uchar sampled = sample( (pos - .5) / scale + .5 );

        cache[ i * cacheDim.x * cacheDim.y + cacheIdx.y * cacheDim.x + cacheIdx.x ]
                = sampled;

        pos += step;
    }

    __syncthreads();
}

template<int _sliceType>
__device__
float4 shadeVoxel(unsigned char sharedMemory[],
                  dim3 cacheIdx,
                  dim3 cacheDim,
                  int offset,
                  float3 voxelPos,
                  float3 voxelDim,
                  float3 slicePoint,
                  float3 sliceNormal,
                  bool phongShading) {

    uchar sampled
             = getVoxel( sharedMemory, cacheIdx, cacheDim, offset );

    float4 value = transferFunction( sampled );

    if ( phongShading && value.w > 1e-6 ) {
        float l, r, t, b, f, a;

        f = ucharToFloat( getVoxel(sharedMemory, cacheIdx, cacheDim, offset - 1) );
        a = ucharToFloat( getVoxel(sharedMemory, cacheIdx, cacheDim, offset + 1) );

        l = ucharToFloat( getVoxel(sharedMemory, dim3(cacheIdx.x - 1, cacheIdx.y), cacheDim, offset) );
        r = ucharToFloat( getVoxel(sharedMemory, dim3(cacheIdx.x + 1, cacheIdx.y), cacheDim, offset) );
        t = ucharToFloat( getVoxel(sharedMemory, dim3(cacheIdx.x, cacheIdx.y + 1), cacheDim, offset) );
        b = ucharToFloat( getVoxel(sharedMemory, dim3(cacheIdx.x, cacheIdx.y - 1), cacheDim, offset) );

        float3 gradient = make_float3(
                    (r - l) / voxelDim.x,
                    (t - b) / voxelDim.y,
                    (a - f) / voxelDim.z );

        if (gradient.x != 0.f && gradient.y != 0.f&& gradient.z != 0.f)
            gradient = normalize(gradient);

        float  direct = dot( gradient, make_float3( -1.f, -1.f, 1.f ) ) * DIRECT_FACTOR;
        direct        = clamp(direct, 0.f, DIRECT_FACTOR);

        value =  make_float4( value.x * ONE_MINUS_DIRECT_FACTOR,
                              value.y * ONE_MINUS_DIRECT_FACTOR,
                              value.z * ONE_MINUS_DIRECT_FACTOR,
                              value.w );
        value += make_float4( direct, direct, direct, 0.f );
    }

    if (_sliceType == SLICE_PLANE) {
        float dist = distanceToPlane( slicePoint, sliceNormal, voxelPos );
        if (dist < .01f) {
            value.x = clamp( value.x + (.01f - dist ) * 100.f, 0.f, 1.f );
        }
    }

    return value;
}

template<int _sliceType>
__device__
void mainLoop(uchar cache[],
              dim3 cacheIdx,
              dim3 cacheDim,
              dim3 imageDim,
              camera_params camera,
              slice_params slice,
              shading_params shading,
              float3 origin,
              float3 direction,
              float upper,
              float3 step,
              float4 & result)
{
    float  dist = 0.f;
    result = make_float4( 0.f, 0.f, 0.f, 0.f );

    float  tanFovX = tan( camera.fovX * M_PI / (180.f * imageDim.x ) ),
           tanFovY = tan( camera.fovY * M_PI / (180.f * imageDim.y ) );

    float3 slicePoint  = make_float3( slice.params[0], slice.params[1], slice.params[2] ),
           sliceNormal = make_float3( slice.params[3], slice.params[4], slice.params[5] ),
           scale       = make_float3( camera.scale[0], camera.scale[1], camera.scale[2] );

    float3  scaledDirection = direction * step;
    float   scaledStep      = vectorLength( scaledDirection );

    float3  front = origin,
            back = origin + direction * upper;

    if (_sliceType == SLICE_PLANE_CUT) {
        if ( signedDistancePlane(slicePoint, sliceNormal, front) < 1e-6 &&
             signedDistancePlane(slicePoint, sliceNormal, back)  < 1e-6 ) {
            return;
        }

        float t;
        if ( intersectPlaneAndRay(slicePoint, sliceNormal, origin, direction, t) ) {
            dist   = t;
        } else if ( intersectPlaneAndRay(slicePoint, sliceNormal, back, direction * -1.f, t) ) {
            upper -= t;
        }
    }

    while ( dist < upper ) { // No infinite loop plz
        float3 pos = origin + direction * dist;

        rayMarch( cache, cacheIdx, cacheDim, pos, scaledDirection, scale );

        for (int i = 1; i < CACHE_DEPTH - 1; ++i) {
            float voxelDist = i * scaledStep + dist;
            if (voxelDist > upper) {
                break;
            }

            float3 voxelDim = make_float3(
                        tanFovX * ( voxelDist ),
                        tanFovY * ( voxelDist ),
                        scaledStep * 2.f
                        ),
                   voxelPos = origin + direction * voxelDist;
\
            float4 shaded = shadeVoxel<_sliceType>( cache, cacheIdx, cacheDim, i, voxelPos, voxelDim, slicePoint, sliceNormal, shading.phongShading );

            if (shaded.w > 1e-6) {
                result = blend( shaded, result );
            }

            if (result.w > .95f) {
                break;
            }
        }

        dist += scaledStep * CACHE_DEPTH_MINUS_TWOF;
    }
}

template<int _sliceType>
__global__
void kernel(void *buffer,
            int width,
            int height,
            struct camera_params camera,
            struct slice_params   slice,
            struct shading_params shading,
            float3 step)
{
    extern __shared__ unsigned char sharedMemory[];
    uchar4 *pixels = (uchar4*) buffer;

    int x = blockIdx.x * (blockDim.x - 2) + ( ( (int) threadIdx.x ) - 1),
        y = blockIdx.y * (blockDim.y - 2) + ( ( (int) threadIdx.y ) - 1);

    int slabUpperX = min( (blockIdx.x + 1) * (blockDim.x - 2) + 1, width - 1  ),
        slabUpperY = min( (blockIdx.y + 1) * (blockDim.y - 2) + 1, height - 1 ),
        slabLowerX = max( (int) (blockIdx.x  * (blockDim.x - 2)) - 1, 0),
        slabLowerY = max( (int) (blockIdx.y  * (blockDim.y - 2)) - 1, 0),
        slabWidth  = slabUpperX - slabLowerX,
        slabHeight = slabUpperY - slabLowerY;

    bool isBorder = threadIdx.x == 0 || threadIdx.y == 0
            || threadIdx.x + 1 == blockDim.x || threadIdx.y + 1 == blockDim.y;

    x = clamp(x, slabLowerX, slabUpperX - 1);
    y = clamp(y, slabLowerY, slabUpperY - 1);

    int index = y * width + x;

    int slabY = y - slabLowerY,
        slabX = x - slabLowerX,
        slabIndex = slabY * slabWidth + slabX,
        slabUpper = slabHeight * slabWidth;

    uchar4 sample0 = tex2D( inTexture0, (float) x / width, (float) y / height ),
           sample1 = tex2D( inTexture1, (float) x / width, (float) y / height );

    float3 front = make_float3(sample0.x / 255.f, sample0.y / 255.f, sample0.z / 255.f),
           back  = make_float3(sample1.x / 255.f, sample1.y / 255.f, sample1.z / 255.f);

    float3  camPos  = make_float3( camera.origin[0], camera.origin[1], camera.origin[2] ),
            camDist = front - camPos;
    float camLength = vectorLength(camDist);

    // Note: all threads in block where index < width * height
    // MUST execute this function. Also NB this includes a thread barrier.
    float rad = blockMin((float*)sharedMemory, slabIndex, slabUpper, camLength);

    float3 dist = back - front;
    float length = vectorLength(dist);

    if (length < 0.001f && !isBorder) {
        // TODO: set a better min value if this happens
        pixels[index] = make_uchar4(0, 0, 0, 0);
        return;
    }

    float3 ray   = dist / length,
           pos   = front;

    float t;
    bool success = intersectSphereAndRay(camPos, rad, front, -ray, t);
    if (success) {
        // Update front based on desired distance from camera
        pos = pos - ray * t;
    }

    float upper = fminf( SQRT_3, vectorLength( back - pos ) );

    dim3 cacheIdx(slabX, slabY),
         cacheDim(slabWidth, slabHeight),
         imageDim(width, height);

    float4 result;
    mainLoop<_sliceType>(sharedMemory, cacheIdx, cacheDim, imageDim, camera, slice, shading, pos, ray, upper, step, result);

    if (!isBorder) {
        result.x = clamp(result.x, 0.f, 1.f);
        result.y = clamp(result.y, 0.f, 1.f);
        result.z = clamp(result.z, 0.f, 1.f);
        result.w = clamp(result.w, 0.f, 1.f);

        pixels[index] = make_uchar4(result.x * 0xff, result.y * 0xff, result.z * 0xff, result.w * 0xff);
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

    inTexture0.normalized = true;
    inTexture1.normalized = true;
}

void runCuda(int width,
             int height,
             struct slice_params slice,
             struct camera_params camera,
             struct shading_params shading,
             cudaArray* volumeArray) {
    cudaGraphicsResource_t resources[3] = { texture0, texture1, pixelBuffer };

    checkCudaErrors( cudaGraphicsMapResources(3, resources) );

    struct cudaArray *array0;
    checkCudaErrors( cudaGraphicsSubResourceGetMappedArray(&array0, texture0, 0, 0) );
    checkCudaErrors( cudaBindTextureToArray(inTexture0, array0) );

    struct cudaArray *array1;
    checkCudaErrors( cudaGraphicsSubResourceGetMappedArray(&array1, texture1, 0, 0) );
    checkCudaErrors( cudaBindTextureToArray(inTexture1, array1) );

    transferTexture.normalized = false;

    assert(devTransfer);
    checkCudaErrors( cudaBindTexture( NULL, transferTexture, devTransfer, 1024 * sizeof(float) ) );

    void *devBuffer;
    size_t bufferSize;
    checkCudaErrors( cudaGraphicsResourceGetMappedPointer(&devBuffer, &bufferSize, pixelBuffer) );

    float3 step = make_float3( 1.f / volumeDim.x, 1.f / volumeDim.y, 1.f / volumeDim.z );

    // For convenience, greedily chunk the screen into 14x14 squares
    dim3 slabSize(BLOCK_WIDTH - 2, BLOCK_HEIGHT - 2),
         blockDims(width / slabSize.x, height / slabSize.y),
         blockSize(BLOCK_WIDTH, BLOCK_HEIGHT);

    if (width % slabSize.x)
        ++blockDims.x;
    if (height % slabSize.y)
        ++blockDims.y;

    size_t sharedMemSize = ( CACHE_DEPTH ) * ( blockSize.x ) * ( blockSize.y ) * sizeof( uchar );

    switch( slice.type ) {

    case SLICE_NONE:

        kernel< SLICE_NONE ><<< blockDims, blockSize, sharedMemSize >>>( devBuffer, width, height, camera, slice, shading, step );

        break;

    case SLICE_PLANE:

        kernel< SLICE_PLANE ><<< blockDims, blockSize, sharedMemSize >>>( devBuffer, width, height, camera, slice, shading, step );

        break;

    case SLICE_PLANE_CUT:

        kernel< SLICE_PLANE_CUT ><<< blockDims, blockSize, sharedMemSize >>>( devBuffer, width, height, camera, slice, shading, step );

    }


    checkCudaErrors( cudaUnbindTexture(inTexture0) );

    checkCudaErrors( cudaGraphicsUnmapResources(3, resources) );
}

// load volumetric texture into the GPU
void cudaLoadVolume(byte* texels, size_t size, Vector3 dims,
                    float transferFunction[1024], cudaArray** volumeArray) {

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();

    cout << "mallocing texture array" << endl;

    cudaExtent extent = make_cudaExtent( dims.x, dims.y, dims.z );
    checkCudaErrors( cudaMalloc3DArray(&devVolume, &channelDesc, extent) );

    assert(texels);
    assert(devVolume);
    assert(size);

    int width = dims.x;
    int height = dims.y;
    int depth = dims.z;

    volumeDim = dim3( width, height, depth );

    cudaMemcpy3DParms params = {0};
    params.srcPtr = make_cudaPitchedPtr(texels, width * sizeof(unsigned char), width, height);
    params.dstArray = devVolume;
    params.extent = make_cudaExtent(width, height, depth);
    params.kind = cudaMemcpyHostToDevice;

    checkCudaErrors( cudaMemcpy3D(&params) );

    // set addressmode
    texVolume.normalized = true;
    texVolume.filterMode = cudaFilterModeLinear;
    texVolume.addressMode[0] = cudaAddressModeClamp;
    texVolume.addressMode[1] = cudaAddressModeClamp;
    texVolume.addressMode[2] = cudaAddressModeClamp;

    checkCudaErrors( cudaBindTextureToArray(texVolume, devVolume, channelDesc));

    cout << "texture array has been memcopied" << endl;

    checkCudaErrors( cudaMalloc( &devTransfer, 1024 *  sizeof(float) ) );
    checkCudaErrors( cudaMemcpy( devTransfer, transferFunction, 1024 * sizeof(float), cudaMemcpyHostToDevice) );

}



// ################ Robin's extra stuff for slicing
#include "output.h"


void invoke_slice_kernel(float *buffer, BufferParameters bp, SliceParameters sp, canonicalOrientation c, float3 scale)
{    
    float* buffer_dev;
    checkCudaErrors( cudaMalloc( &buffer_dev, bp.height*bp.width*sizeof(float)) );

    int dim = 1024;
    dim3 grids(dim/16, dim/16);
    dim3 threads(16,16);

    slice_kernel<<<grids,threads>>>( buffer_dev, bp, sp, c, scale );

    checkCudaErrors( cudaMemcpy( buffer, buffer_dev, bp.height*bp.width*sizeof(float), cudaMemcpyDeviceToHost) );
    checkCudaErrors( cudaFree( buffer_dev ) );
}


void invoke_advanced_slice_kernel(float *buffer, BufferParameters bp, Matrix4x4 trans, float3 scale)
{
    float* buffer_dev;
    REAL* trans_dev;

    checkCudaErrors( cudaMalloc( &buffer_dev, bp.height*bp.width*sizeof(float)) );
    checkCudaErrors( cudaMalloc( &trans_dev, 16*sizeof(REAL) ) );
    checkCudaErrors( cudaMemcpy( trans_dev, trans.data, 16*sizeof(REAL), cudaMemcpyHostToDevice) );

    int dim = 1024;
    dim3 grids(dim/16, dim/16);
    dim3 threads(16,16);

    advanced_slice_kernel<<<grids,threads>>>( buffer_dev, bp, trans_dev, scale );

    checkCudaErrors( cudaMemcpy( buffer, buffer_dev, bp.height*bp.width*sizeof(float), cudaMemcpyDeviceToHost) );
    checkCudaErrors( cudaFree( buffer_dev ) );


}

__global__
void slice_kernel(float *buffer, BufferParameters bp, SliceParameters sp, canonicalOrientation c, float3 scale)
{
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
//    int offset = i + j * blockDim.x * gridDim.x; // ???

    int offset = j*bp.height+i;

    if(j<bp.height && i<bp.width){

            float3 pos;
            pos.x = 0;
            pos.y = 0;
            pos.z = 0;

            switch(c){

            case SAGITTAL:
                pos.z += 0;
                pos.y += ((float)j)/((float)bp.height);
                pos.x += ((float)i)/((float)bp.width);
                break;

            case HORIZONTAL:
                pos.z += ((float)i)/((float)bp.width);
                pos.y += 0;
                pos.x += ((float)j)/((float)bp.height);
                break;


            case CORONAL:
                pos.z += ((float)i)/((float)bp.width);
                pos.y += ((float)j)/((float)bp.height);
                pos.x += 0;
                break;
            }

            pos.x += sp.dx;
            pos.y += sp.dy;
            pos.z += sp.dz;

            pos = (pos - .5) / scale + .5;

            float sample;
            if (boundsCheck(pos)) {
                sample = tex3D(texVolume, pos.x, pos.y, pos.z);
            }
            else {
                sample = 0.f;
            }

            buffer[offset] = sample;
    }
}

__global__
void advanced_slice_kernel(float *buffer, BufferParameters bp, REAL* trans, float3 scale)
{
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = j*bp.height+i;

    if(j<bp.height && i<bp.width){

        float4 raw;
        raw.x = ((float)i)/((float)bp.width);
        raw.y = ((float)j)/((float)bp.height);
        raw.z = 0.5;
        raw.w = 1;

        float4 pos;

        pos.x = trans[0]*raw.x + trans[1]*raw.y + trans[2]*raw.z + trans[3]*raw.w;
        pos.y = trans[4]*raw.x + trans[5]*raw.y + trans[6]*raw.z + trans[7]*raw.w;
        pos.z = trans[8]*raw.x + trans[9]*raw.y + trans[10]*raw.z + trans[11]*raw.w;

        pos.x /= scale.x;
        pos.y /= scale.y;
        pos.z /= scale.z;

        pos = (pos - .5) / make_float4(scale.x, scale.y, scale.z, 0.f) + .5;

        float sample;
        if (boundsCheck(make_float3(pos.x, pos.y, pos.z))) {
            sample = tex3D(texVolume, pos.x, pos.y, pos.z);
        }
        else {
            sample = 0.f;
        }

//        float sample;
//        if(pos.x <= 0.0 || pos.x >= 1.0 || pos.y <= 0.0 || pos.y >= 1.0 || pos.z <= 0.0 || pos.z >= 1.0){
//            sample = 0.0;
//        }else{

//            sample = tex3D(texVolume, pos.x, pos.y, pos.z);
//        }

        buffer[offset] = sample;
    }
}



