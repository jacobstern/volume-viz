#ifndef PARAMS_H
#define PARAMS_H

#include "output.h"

#include <iostream>

#define VOLUME_RESOLUTION 256

#define DEFAULT_TEXTURE_PATH "/home/rmartens/shared/cs224textures/head.t3d"
#define DEFAULT_SAVE_PATH "/home/rmartens/shared/cs224slices/myfirstslice.jpg"

#define N_SLICE_SLIDERS 6

#define SLICE_EDGELENGTH 256

#define N_DEFAULT_TEXTURES 2

#define SLICE_SIZE 256
#define RENDER_SIZE 512

#define LEFT_COLUMN_WIDTH 450
#define LEFT_COLUMN_HEIGHT 600

#define SLICE_SLIDER_MIN -100
#define SLICE_SLIDER_MAX 100
#define SLICE_SLIDER_INIT 0


static const char *g_texture_names[N_DEFAULT_TEXTURES] = {"MRI head",
                                                  "Engine"};
static const char *g_texture_paths[N_DEFAULT_TEXTURES] = {"/home/rmartens/shared/cs224textures/head.t3d",
                                                 "/home/rmartens/shared/cs224textures/engine.t3d"};

static char *g_savepath_default = "Save slice as:";

static char* g_slice_slider_captions[N_SLICE_SLIDERS] = {"x offset", "y offset", "z offset",
                                                         "x rotation", "y rotation", "z rotation"};

typedef enum{ HORIZONTAL, SAGITTAL, CORONAL, N_CANONICAL_ORIENTATIONS, FREE_FORM} canonicalOrientation;
static char* g_canonical_orientation_captions[N_CANONICAL_ORIENTATIONS] = {"horizontal", "sagittal", "coronal"};

typedef enum{BMP, JPG, PNG, TIFF, N_OUTPUT_FILE_FORMATS} outputFileFormat;
static const char* g_output_file_formats[N_OUTPUT_FILE_FORMATS] = {"BMP", "JPG", "PNG", "TIFF"};

#define N_SLICE_VISUALIZATIONS 3
typedef enum { SLICE_VIS_NONE = 0, SLICE_VIS_LASER, SLICE_VIS_CROSS_SECTION } sliceVisualization;
static const char* g_slice_visualization_captions[N_SLICE_VISUALIZATIONS] = {"None", "Laser", "Cross section"};

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
    BufferParameters(size_t height_, size_t width_) : height(height_), width(width_) {}

    size_t height;
    size_t width;
};

std::ostream& operator<<(std::ostream& os, const SliceParameters p);
std::ostream& operator<<(std::ostream& os, const BufferParameters p);

#endif // PARAMS_H
