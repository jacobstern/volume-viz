#ifndef _KERNEL_H
#define _KERNEL_H

#include <OpenGL/gl.h>

extern "C" {
    struct slice_params {
        int   type;
        float params[6];
    };

    void initCuda();
    void registerCudaResources(GLuint input0, GLuint input1, GLuint output);
    void runCuda(int width, int height, struct slice_params *slice);
}

#endif
