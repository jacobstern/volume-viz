#include <helper_math.h>

__device__
bool ray_inter_plane(float3 p0, float3 n, float3 l0, float3 l, float &t)
{
    // from scratchapixel.com
    float denom = dot(n, l);

    if (denom > 1e-6) {
        float3 p0l0 = p0 - l0;
        t = dot(p0l0, n) / denom;

        return t >= 0;
    }

    return false;
}

__device__
bool ray_inter_sphere(float3 p0, float r, float3 l0, float3 l, float &t)
{
    float3 l0p0 = l0 - p0;

    float b = dot(l, l0p0);
    float c = dot(l0p0, l0p0) - r * r;

    float discrim = b * b - c;
    if (discrim >= 0.f) {
        t = b * -1.f - sqrt(discrim);

        return t > -1e-6;
    }

    return false;
}
