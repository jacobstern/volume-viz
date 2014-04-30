#include <cuda.h>
#include <cuda_gl_interop.h>
#include "params.h"


//#include <helper_cuda.h>

// DANGER: HACK!
#define checkCudaErrors(A) A

//#include <helper_cuda_gl.h>
#include <helper_math.h>

#include <assert.h>
#include <stdio.h>

#include "kernel.cuh"
#include "assert.h"
#include "params.h"



#include <iostream>

void invoke_slice_kernel(float *buffer, BufferParameters bp, SliceParameters sp, canonicalOrientation c);

// NOTE: Perhaps pass in matrix format...
__global__
void slice_kernel(float *buffer, BufferParameters bp, SliceParameters sp, canonicalOrientation c);



