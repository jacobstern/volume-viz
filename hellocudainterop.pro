HEADERS       = glwidget.h \
                window.h \
                qtlogo.h \
    camera.h \
    volumegenerator.h \
    params.h \
    slicewidget.h \
    cs123math/CS123Algebra.h \
    CS123Common.h \
    transfer_functions.h

SOURCES       = glwidget.cpp \
                main.cpp \
                window.cpp \
                qtlogo.cpp \
    camera.cpp \
    volumegenerator.cpp \
    slicewidget.cpp \
    cs123math/CS123Matrix.cpp \
    cs123math/CS123Matrix.inl \
    cs123math/CS123Vector.inl \
    params.cpp
QT           += opengl widgets

# install
target.path = ~/Dropbox/Developer/hellocudainterop
INSTALLS += target

OTHER_FILES += kernel.cu \
    kernel.cuh \
    color_test_pattern.jpg \
    firstpass.frag \
    firstpass.vert \
    screen.frag \
    screen.vert \
    ui.frag \
    ui.vert \
    implicit.cu


CUDA_SOURCES += kernel.cu


# paths to cuda sdk on filesystem
CUDA_SDK = /Developer/NVIDIA/CUDA-5.5/
CUDA_DIR = /Developer/NVIDIA/CUDA-5.5/
# cuda architecture for Jake's laptop is compute_12, which supports atomic memory operations
CUDA_ARCH = sm_21
# flags for the cuda compiler, in particular, verbosity about what ptx assembly is doing
NVCC_FLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v

#include paths for cuda
INCLUDEPATH += $$CUDA_DIR/include \
                $$CUDA_SDK/common/inc \
                $$CUDA_SDK/../shared/inc

INCLUDEPATH += $$_PRO_FILE_PWD_/include/

#libs
LIBS += -L$$CUDA_DIR/lib \
        -L$$CUDA_SDK/lib \
        -lGLU

LIBS += -lcudart
CUDA_INC = $$join(INCLUDEPATH, ' -I', '-I', ' ')

cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o

# tell it to use cuda compilers
#cuda.commands = $$CUDA_DIR/bin/nvcc -g -G -arch=$$CUDA_ARCH -c $$NVCC_FLAGS $$CUDA_INC $$LIBS ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
cuda.commands = /usr/bin/nvcc -g -G -arch=$$CUDA_ARCH -c $$NVCC_FLAGS $$CUDA_INC $$LIBS ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}


cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -g -G -M $$CUDA_INC $$NVCCFLAGS ${QMAKE_FILE_NAME}

QMAKE_EXTRA_COMPILERS += cuda

contains(QT_CONFIG, opengles.) {
    contains(QT_CONFIG, angle): \
        warning("Qt was built with ANGLE, which provides only OpenGL ES 2.0 on top of DirectX 9.0c")
    error("This example requires Qt to be configured with -opengl desktop")
}

RESOURCES += \
    assets.qrc \
    shaders.qrc
