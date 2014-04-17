#ifndef _KERNEL_H
#define _KERNEL_H

#include <OpenGL/gl.h>

extern "C" {
    void initCuda();
    void registerCudaResources(GLuint input0, GLuint input1, GLuint output);
    void runCuda(int width, int height);
}

#endif
