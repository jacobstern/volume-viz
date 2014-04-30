/****************************************************************************
**
** Copyright (C) 2013 Digia Plc and/or its subsidiary(-ies).
** Contact: http://www.qt-project.org/legal
**
** This file is part of the examples of the Qt Toolkit.
**
** $QT_BEGIN_LICENSE:BSD$
** You may use this file under the terms of the BSD license as follows:
**
** "Redistribution and use in source and binary forms, with or without
** modification, are permitted provided that the following conditions are
** met:
**   * Redistributions of source code must retain the above copyright
**     notice, this list of conditions and the following disclaimer.
**   * Redistributions in binary form must reproduce the above copyright
**     notice, this list of conditions and the following disclaimer in
**     the documentation and/or other materials provided with the
**     distribution.
**   * Neither the name of Digia Plc and its Subsidiary(-ies) nor the names
**     of its contributors may be used to endorse or promote products derived
**     from this software without specific prior written permission.
**
**
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
** OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
** LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
** DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
** THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
** (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
** OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
**
** $QT_END_LICENSE$
**
****************************************************************************/

#include <QApplication>
#include <QtGui>
#include <qgl.h>

#include <ctime>
#include <math.h>
#include <iostream>

#include "slicewidget.h"
#include "qtlogo.h"
#include "kernel.cuh"
#include "slicekernel.cuh"
#include "params.h"

using std::cout;
using std::endl;

#ifndef __APPLE__
extern "C" {
    GLAPI void APIENTRY glBindBuffer (GLenum target, GLuint buffer);
    GLAPI void APIENTRY glGenBuffers (GLsizei n, GLuint *buffers);
    GLAPI void APIENTRY glBufferData (GLenum target, GLsizeiptr size, const GLvoid *data, GLenum usage);

    GLAPI void APIENTRY glUseProgram (GLuint program);

    // extrapolation
    GLAPI void APIENTRY glDeleteBuffers (GLenum target, GLuint *buffers);
}
#endif

#define CHECK_GL_ERROR_DEBUG() \
    do { \
        GLenum __error = glGetError(); \
        if(__error) { \
            printf("OpenGL error 0x%04X in %s %s %d\n", __error, __FILE__, __FUNCTION__, __LINE__); \
        } \
    } while (false)

#define FRONT_FACE_BUFFER   0
#define BACK_FACE_BUFFER    1

static inline float glc( float normalized )
{
    return normalized * 2.f - 1.f;
}

SliceWidget::SliceWidget(QWidget *parent)
//    : QWidget(QGLFormat(QGL::SampleBuffers), parent),
//      font("Deja Vu Sans Mono", 8, 4), fovX(0.f), fovY(0.f)
{
}

SliceWidget::~SliceWidget()
{
}

QSize SliceWidget::minimumSizeHint() const
{
    return QSize(50, 50);
}


QSize SliceWidget::sizeHint() const
{
    return QSize(800, 600);
}


void SliceWidget::initializeGL()
{
    glClearColor(0.0, 0.0, 0.0, 1.0);

//    glEnable(GL_TEXTURE_2D);
//    glEnable(GL_DEPTH_TEST);
//    glEnable(GL_LINE_SMOOTH);

//    glMatrixMode(GL_MODELVIEW);

//    loadShaderProgram( firstPass, tr("firstpass") );
//    loadShaderProgram( screen, tr("screen") );
//    loadShaderProgram( ui, tr("ui") );

//    initCuda();
//    loadVolume();
}


void SliceWidget::paintGL()
{

}

void SliceWidget::resizeGL(int width, int height)
{

}

void SliceWidget::renderSlice(SliceParameters sliceParameters,
                              BufferParameters bufferParameters,
                              canonicalOrientation orientation)
{
    if(bufferParameters.height*bufferParameters.width != m_sizeY*m_sizeX){
        cout << "allocating new slice buffer" << endl;
        delete[] m_sliceBuffer;
        m_sliceBuffer = new float[bufferParameters.height*bufferParameters.width];
        m_sizeY = bufferParameters.height;
        m_sizeX = bufferParameters.width;
        cout << "new slice buffer allocated" << endl;
    }

    invoke_slice_kernel(m_sliceBuffer, bufferParameters, sliceParameters, orientation);

    QImage img(bufferParameters.width, bufferParameters.height, QImage::Format_RGB32);

    // TODO: Convert to rgb32
    BGRA* bits = new BGRA[bufferParameters.width*bufferParameters.height];
    for(int j=0; j<bufferParameters.height; j++){
        for(int i=0; i<bufferParameters.width; i++){
            int offset = j*bufferParameters.height+i;
            unsigned int val =(unsigned int)( m_sliceBuffer[offset]*255 );
            bits[offset] = BGRA(0.5, 0.5, 0.5, 1.0);
        }
    }
    memcpy(img.bits(), bits, bufferParameters.width*bufferParameters.height*sizeof(BGRA));
    delete[] bits;
    QPainter painter(this);
    cout << "drawing image" << endl;
    painter.drawImage(QPoint(0,0), img);
    cout << "image drawn" << endl;

    update();
}

float* SliceWidget::getSlice(size_t& height, size_t& width)
{
    height = m_sizeY;
    width = m_sizeX;
    return m_sliceBuffer;
}







