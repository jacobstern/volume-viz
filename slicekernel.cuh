#include <cuda.h>
#include <cuda_gl_interop.h>

struct SliceParameters {

    SliceParameters(float x, float y, float z) : dx(x), dy(y), dz(z) {}
    SliceParameters(float x, float y, float z, float t, float ph, float ps) : dx(x), dy(y), dz(z), theta(t), phi(ph), psi(ps) {}

    float dx;
    float dy;
    float dz;
    float theta;
    float phi;
    float psi;
};

struct BufferParameters {
    BufferParameters(size_t x_, size_t y_) : x(x_), y(y_) {}

    size_t x;
    size_t y;
};
