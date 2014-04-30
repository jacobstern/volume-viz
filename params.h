#ifndef PARAMS_H
#define PARAMS_H

#define VOLUME_RESOLUTION 256

#define DEFAULT_TEXTURE_PATH "/home/rmartens/shared/cs224textures/head.t3d"

#define N_SLICE_SLIDERS 6

#define N_DEFAULT_TEXTURES 5
static char *g_texture_names[N_DEFAULT_TEXTURES] = {"head",
                                                  "engine",
                                                    "foo",
                                                    "bar",
                                                    "baz"};
static char *g_texture_paths[N_DEFAULT_TEXTURES] = {"/home/rmartens/shared/cs224textures/head.t3d",
                                                 "/home/rmartens/shared/cs224textures/engine.t3d",
                                                    "foo/path",
                                                    "bar/path",
                                                    "baz/path"};

static char* g_slice_slider_captions[N_SLICE_SLIDERS] = {"dx", "dy", "dz", "theta", "phi", "psi"};

typedef enum{ HORIZONTAL, SAGITTAL, CORONAL, N_CANONICAL_ORIENTATIONS} canonicalOrientation;
static char* g_canonical_orientation_captions[N_CANONICAL_ORIENTATIONS] = {"horizontal", "sagittal", "coronal"};


#endif // PARAMS_H
