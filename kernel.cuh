#ifndef _KERNEL_H
#define _KERNEL_H

#include <GL/gl.h>

extern "C" {

#define SLICE_NONE  -1
#define SLICE_PLANE  0

    struct slice_params {
        int   type;
        float params[6];
    };

    struct camera_params {
        float origin[3];
        float fovX, fovY;
    };

    void initCuda();
    void registerCudaResources(GLuint input0, GLuint input1, GLuint output);
    void runCuda(int width, int height, struct slice_params slice, struct camera_params camera);
}

#endif
