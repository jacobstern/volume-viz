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

using std::cout;
using std::endl;
#include "implicit.cu"

static struct cudaGraphicsResource *pixelBuffer, *texture0, *texture1;

typedef texture<uchar4, cudaTextureType2D, cudaReadModeElementType> inTexture2D;
inTexture2D inTexture0, inTexture1;

// volumetric texture
texture<unsigned char, cudaTextureType3D, cudaReadModeNormalizedFloat> texVolume;

cudaArray *devVolume = 0;

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

//    float cof = STEP_SIZE*STEP_SIZE*STEP_SIZE/23;
    float cof = 1.0;
    float x = pos.x*cof;
    float y = pos.y*cof;
    float z = pos.z*cof;

    byte sample = tex3D(texVolume, x, y, z);
    float s = ((float)sample)/255.0;

    float r = s;
    float g = s;
    float b = 5*s;
    float a = 0.05;

    return make_float4(r, g, b, a);

//    if(sample == 1){
//        return make_float4(0.5, 0.0, 0, 1);

//    }else if(sample == 2){
//        return make_float4(0.0, 0.5, 0.0, 1);

//    }else if(sample == 3){
//        return make_float4(0.0, 0.0, 0.5, 1);

//    }else if(sample == 4){
//        return make_float4(0.0, 0.5, 0.5, 1);

//    }else if(sample == 5){
//        return make_float4(0.5, 0.0, 0.5, 1);

//    }else{
//        return make_float4(sample, sample, sample, 0.05)/8;
//    }

//    return make_float4(sample, sample, sample, 0.05)/122;
}

#define MAX_STEPS 31

__device__
unsigned char sample(float3 pos) {
    if ( !boundsCheck(pos) ) {
        return 0x00;
    }

    return 0xff * tex3D(texVolume, pos.x, pos.y, pos.z);
}

__device__
void rayMarch(unsigned char sharedMemory[], float3 origin, float3 step, dim3 cacheIdx, dim3 cacheDim, int lower=0, int upper=MAX_STEPS) {
    float3 pos = origin + lower * step;
    unsigned char i = 0x00;

    for (; i < lower; ++i)
        sharedMemory[ (i + 1) * cacheDim.x * cacheDim.y + cacheIdx.y * cacheDim.x + cacheIdx.x ]
                = 0x00;

    for (; i < upper; ++i) {
        if (pos.x < 1.0f && pos.x >= 0.0f
                && pos.y < 1.0f && pos.y >= 0.0f
                && pos.z < 1.0f && pos.z >= 0.0f) {
            break;
        }

        sharedMemory[ (i + 1) * cacheDim.x * cacheDim.y + cacheIdx.y * cacheDim.x + cacheIdx.x ]
                = sample(pos);

        pos += step;
    }


    for (; i < upper; ++i) {
        if (pos.x >= 1.0f || pos.x < 0.0f
                || pos.y >= 1.0f || pos.y < 0.0f
                || pos.z >= 1.0f || pos.z < 0.0f) {
            break;
        }

        sharedMemory[ (i + 1) * cacheDim.x * cacheDim.y + cacheIdx.y * cacheDim.x + cacheIdx.x ]
                = sample(pos);

        pos += step;
    }

    sharedMemory[ cacheIdx.y * cacheDim.x + cacheIdx.x ] = i;

    // Fill up the rest of the cache with zeros

    for (; i < MAX_STEPS; ++i)
        sharedMemory[ (i + 1) * cacheDim.x * cacheDim.y + cacheIdx.y * cacheDim.x + cacheIdx.x ]
                = 0x00;

    __syncthreads();
}

__device__
float4 shadePhong(unsigned char sharedMemory[], dim3 cacheIdx, dim3 cacheDim) {
    unsigned char upper = sharedMemory[ cacheIdx.y * cacheDim.x + cacheIdx.x ];

    for (unsigned char i = 0; i < upper; ++i) {
        unsigned char sampled =  sharedMemory[ (i + 1) * cacheDim.x * cacheDim.y + cacheIdx.y * cacheDim.x + cacheIdx.x ];

        if (sampled) {
            return make_float4(1.f, 1.f, 1.f, 1.f);
        }
    }

    return make_float4(0.f, 0.f, 0.f, 0.f);
}

__device__ unsigned char getVoxel(unsigned char sharedMemory[], dim3 cacheIdx, dim3 cacheDim, int offset) {
    if (offset < 0 || offset + 1 > MAX_STEPS)
        return 0x00;

    return sharedMemory[ (offset + 1) * cacheDim.x * cacheDim.y + cacheIdx.y * cacheDim.x + cacheIdx.x ];
}

#define DEBUG_TRANSPARENT

__device__
float4 shadeVoxel(unsigned char sharedMemory[], dim3 cacheIdx, dim3 cacheDim, int offset, float stepSize, float tanFovX, float tanFovY) {
#ifdef DEBUG_TRANSPARENT
    unsigned char sampled =  getVoxel(sharedMemory, cacheIdx, cacheDim, offset);

    return make_float4(sampled / 255.f, sampled / 255.f, sampled / 255.f, sampled * stepSize / 255.f);
#endif

#ifdef DEBUG_PHONG
    unsigned char c = getVoxel(sharedMemory, cacheIdx, cacheDim, offset);

    if ( c ) {
        float l, r, t, b, f, a;

        f = ucharToFloat( getVoxel(sharedMemory, cacheIdx, cacheDim, offset - 1) );
        a = ucharToFloat( getVoxel(sharedMemory, cacheIdx, cacheDim, offset + 1) );

        l = ucharToFloat( getVoxel(sharedMemory, dim3(cacheIdx.x - 1, cacheIdx.y), cacheDim, offset) );
        r = ucharToFloat( getVoxel(sharedMemory, dim3(cacheIdx.x + 1, cacheIdx.y), cacheDim, offset) );
        t = ucharToFloat( getVoxel(sharedMemory, dim3(cacheIdx.x, cacheIdx.y + 1), cacheDim, offset) );
        b = ucharToFloat( getVoxel(sharedMemory, dim3(cacheIdx.x, cacheIdx.y - 1), cacheDim, offset) );

        float3 gradient = make_float3(0.f, 0.f, 0.f);
        gradient.x = (r - l) / (tanFovX * offset * stepSize);
        gradient.y = (t - b) / (tanFovY * offset * stepSize);
        gradient.z = (a - f) / (stepSize * 2.f);

        gradient = normalize(gradient);

        return make_float4(fabs(gradient.x), fabs(gradient.y), fabs(gradient.z), 1.f);
    }

#endif

    return make_float4(0.f, 0.f, 0.f, 0.f);
}

__device__
float4 shade(unsigned char sharedMemory[], dim3 cacheIdx, dim3 cacheDim, float normalize, struct camera_params cameraParams, int width, int height) {
    unsigned char upper = sharedMemory[ cacheIdx.y * cacheDim.x + cacheIdx.x ];
    float4 accum = make_float4(0.f, 0.f, 0.f, 0.f);

    double tanFovX = tan( cameraParams.fovX * M_PI / (180.f * width ) ),
           tanFovY = tan( cameraParams.fovY * M_PI / (180.f * height) );

    for (unsigned char i = 0x00; i < upper; ++i) {
        float4 vox = shadeVoxel(sharedMemory, cacheIdx, cacheDim, i, normalize, tanFovX, tanFovY);

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

    accum.x = fminf(accum.x, 1.f);
    accum.y = fminf(accum.y, 1.f);
    accum.z = fminf(accum.z, 1.f);
    accum.w = fminf(accum.w, 1.f);

    return accum;
}

#define SQRT_3 1.73205081f

__global__
void kernel(void *buffer,
            int width,
            int height,
            struct slice_params slice,
            struct camera_params camera )
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

    uchar4 sample0 = tex2D( inTexture0, x, y ),
           sample1 = tex2D( inTexture1, x, y );

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

    if (slice.type == SLICE_PLANE) {
        float3 point  = make_float3( slice.params[0], slice.params[1], slice.params[2] ),
               normal = make_float3( slice.params[3], slice.params[4], slice.params[5] );

        // TODO: Slicing
    }

    float t;
    bool success = intersectSphereAndRay(camPos, rad, front, -ray, t);
    if (success) {
        // Update front based on desired distance from camera
        pos = pos - ray * t;
    }

    float  distActual = vectorLength(back - pos),
           stepSize = fminf( SQRT_3, distActual ) / MAX_STEPS;

    float3 step = ray * stepSize;

    dim3 cacheIdx(slabX, slabY),
         cacheDim(slabWidth, slabHeight);

    rayMarch(sharedMemory, pos, step, cacheIdx, cacheDim);

    if (!isBorder) {
        float4 shaded = shade(sharedMemory, cacheIdx, cacheDim, stepSize, camera, width, height);

        pixels[index] = make_uchar4(shaded.x * 0xff, shaded.y * 0xff, shaded.z * 0xff, shaded.w * 0xff);
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

    // For convenience, greedily chunk the screen into 14x14 squares
    dim3 blockSize(14, 14),
         blockDims(width / blockSize.x, height / blockSize.y);
    if (width % blockSize.x)
        ++blockDims.x;
    if (height % blockSize.y)
        ++blockDims.y;

//    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar1>();
//    checkCudaErrors( cudaBindTextureToArray(texVolume, volumeArray, channelDesc));

    blockSize.x += 2;
    blockSize.y += 2;

    size_t sharedMemSize = ( MAX_STEPS + 1 ) * ( blockSize.x ) * ( blockSize.y ) * sizeof( unsigned char );
    kernel<<< blockDims, blockSize, sharedMemSize >>>(devBuffer, width, height, slice, camera);

    checkCudaErrors( cudaUnbindTexture(inTexture0) );

    checkCudaErrors( cudaGraphicsUnmapResources(3, resources) );
}

// load volumetric texture into the GPU
void cudaLoadVolume(byte* texels, size_t size, Vector3 dims,
                    cudaArray** volumeArray) {

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();

    cout << "mallocing texture array" << endl;

    cudaExtent extent = make_cudaExtent( dims.x, dims.y, dims.z );
    checkCudaErrors( cudaMalloc3DArray(&devVolume, &channelDesc, extent) );
    cout << "texture array has been malloced" << endl;

    cout << "size: " << size << endl;

    cout << "memcopying texture array" << endl;
    assert(texels);
    assert(devVolume);
    assert(size);

    int width = dims.x;
    int height = dims.y;
    int depth = dims.z;

    cudaMemcpy3DParms params = {0};
    params.srcPtr = make_cudaPitchedPtr(texels, width * sizeof(unsigned char), width, height);
    params.dstArray = devVolume;
    params.extent = make_cudaExtent(width, height, depth);
    params.kind = cudaMemcpyHostToDevice;

    checkCudaErrors( cudaMemcpy3D(&params) );

//    params.srcPtr.pitch = sizeof(byte)*width;
//    params.srcPtr.ptr = texels;
//    params.srcPtr.xsize = width;
//    params.srcPtr.ysize = height;

//    params.srcPos.x = 0;
//    params.srcPos.y = 0;
//    params.srcPos.z = 0;

//    params.dstArray = devVolume;

//    params.dstPos.x = 0;
//    params.dstPos.y = 0;
//    params.dstPos.z = 0;

//    params.extent.width = width;
//    params.extent.depth = depth;
//    params.extent.height = height;

//    params.kind = cudaMemcpyHostToDevice;

    // set addressmode
    texVolume.normalized = true;
    texVolume.filterMode = cudaFilterModeLinear;
    texVolume.addressMode[0] = cudaAddressModeWrap;
    texVolume.addressMode[1] = cudaAddressModeWrap;
    texVolume.addressMode[2] = cudaAddressModeWrap;

    checkCudaErrors( cudaBindTextureToArray(texVolume, devVolume, channelDesc));

    cout << "texture array has been memcopied" << endl;


//    cout << "copying data back for testing" << endl;

//    // Sanity check: Copy texture back
//    byte* back_texels = new byte[size];
//    memset(back_texels, '\0', size);


//    cudaMemcpy3DParms back_params = {0};

//    back_params.dstPtr.pitch = sizeof(byte) * width;
//    back_params.dstPtr.ptr = back_texels;
//    back_params.dstPtr.xsize = width;
//    back_params.dstPtr.ysize = height;

//    back_params.srcPos.x = 0;
//    back_params.srcPos.y = 0;
//    back_params.srcPos.z = 0;

//    back_params.srcArray = tex_array;

//    back_params.dstPos.x = 0;
//    back_params.dstPos.y = 0;
//    back_params.dstPos.z = 0;

//    back_params.extent.width = width;
//    back_params.extent.depth = depth;
//    back_params.extent.height = height;

//    back_params.kind = cudaMemcpyDeviceToHost;

//    cout << "invoking" << endl;
//    checkCudaErrors( cudaMemcpy3D(&back_params) );


//    // TODO: Copy back for sanity check!
//    for(int i=0; i<size; i++){
//        if(texels[i] != back_texels[i]){
//            printf("i: %d, texels: %u, back_texels: %u\n", i, texels[i], back_texels[i]);
//            assert(false);


//        }
//    }



//    cout << "data has been copied back for testing" << endl;


}



// ################ Robin's extra stuff for slicing

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

            switch(c){

            case SAGITTAL:
                pos.z = 0;
                pos.y = ((float)j)/((float)bp.height);
                pos.x = ((float)i)/((float)bp.width);
                break;

            case HORIZONTAL:
                pos.z = ((float)i)/((float)bp.width);
                pos.y = 0;
                pos.x = ((float)j)/((float)bp.height);
                break;


            case CORONAL:
                pos.z = ((float)i)/((float)bp.width);
                pos.y = ((float)j)/((float)bp.height);
                pos.x = 0;
                break;
            }

            pos.x += sp.dx;
            pos.y += sp.dy;
            pos.z += sp.dz;

            float sample = tex3D(texVolume, pos.x, pos.y, pos.z);

            buffer[offset] = sample;

//          buffer[offset] = (pos.x + pos.y)/2;


    }


}



