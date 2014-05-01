#ifndef _KERNEL_H
#define _KERNEL_H

#include <GL/gl.h>


#include <cuda.h>
#include <cuda_gl_interop.h>

#include "volumegenerator.h"

#include "params.h"

#include "cs123math/CS123Algebra.h"

extern "C" {

#define SLICE_NONE  -1
#define SLICE_PLANE  0

#define TRANSFER_PRESET_DEFAULT -1
#define TRANSFER_PRESET_ENGINE   0
#define TRANSFER_PRESET_MRI      1

    struct slice_params {
        int   type;
        float params[6];
    };

    struct camera_params {
        float origin[3];
        float fovX, fovY;
        float scale[3];    // TODO: This probably shouldn't be a member of camera_params
    };

    struct shading_params {
        int transferPreset;
        bool  phongShading;
    };



    void initCuda();
    void registerCudaResources(GLuint input0, GLuint input1, GLuint output);
    void runCuda(int width,
                 int height,
                 struct slice_params slice,
                 struct camera_params camera,
                 struct shading_params shading,
                 cudaArray* volumeArray);

    void cudaLoadVolume(byte* texels, size_t size, Vector3 dims,
                        cudaArray** volumeArray); // load volumetric texture
}


// ############## Robin's extra stuff for slicing
void invoke_slice_kernel(float *buffer, BufferParameters bp, SliceParameters sp, canonicalOrientation c);

void invoke_advanced_slice_kernel(float *buffer, BufferParameters bp, Matrix4x4 trans);

// NOTE: Perhaps pass in matrix format...
__global__
void slice_kernel(float *buffer, BufferParameters bp, SliceParameters sp, canonicalOrientation c);

__global__
void advanced_slice_kernel(float *buffer, BufferParameters bp, REAL* trans);



#endif
