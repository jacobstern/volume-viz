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

#include <QtWidgets>
#include <QtOpenGL>

#include <math.h>
#include <iostream>

#include "glwidget.h"
#include "qtlogo.h"
#include "kernel.cuh"

#define CHECK_GL_ERROR_DEBUG() \
    do { \
        GLenum __error = glGetError(); \
        if(__error) { \
            printf("OpenGL error 0x%04X in %s %s %d\n", __error, __FILE__, __FUNCTION__, __LINE__); \
        } \
    } while (false)

#define FRONT_FACE_BUFFER   0
#define BACK_FACE_BUFFER    1

static void perspectiveFrustum(GLdouble fovy, GLdouble aspect, GLdouble zNear, GLdouble zFar)
{
    GLdouble xmin, xmax, ymin, ymax;

       ymax = zNear * tan(fovy * M_PI / 360.0);
       ymin = -ymax;
       xmin = ymin * aspect;
       xmax = ymax * aspect;


       glFrustum(xmin, xmax, ymin, ymax, zNear, zFar);
}

//! [0]
GLWidget::GLWidget(QWidget *parent)
    : QGLWidget(QGLFormat(QGL::SampleBuffers), parent)
{
    logo = 0;
    xRot = 0;
    yRot = 0;
    zRot = 0;

    qtGreen = QColor::fromCmykF(0.40, 0.0, 1.0, 0.0);
    qtPurple = QColor::fromCmykF(0.39, 0.39, 0.0, 0.0);

    for (int i = 0; i < GLWidget::N_FRAMEBUFFERS; i++) {
        framebuffers[i] = NULL;
    }

    resultBuffer = 0;
}
//! [0]

//! [1]
GLWidget::~GLWidget()
{
    for (int i = 0; i < GLWidget::N_FRAMEBUFFERS; i++) {
        if (framebuffers[i]) {
            delete framebuffers[i];
        }
    }
}
//! [1]

//! [2]
QSize GLWidget::minimumSizeHint() const
{
    return QSize(50, 50);
}
//! [2]

//! [3]
QSize GLWidget::sizeHint() const
//! [3] //! [4]
{
    return QSize(400, 400);
}
//! [4]

static void qNormalizeAngle(int &angle)
{
    while (angle < 0)
        angle += 360 * 16;
    while (angle > 360 * 16)
        angle -= 360 * 16;
}

//! [5]
void GLWidget::setXRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != xRot) {
        xRot = angle;
        emit xRotationChanged(angle);
        updateGL();
    }
}
//! [5]

void GLWidget::setYRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != yRot) {
        yRot = angle;
        emit yRotationChanged(angle);
        updateGL();
    }
}

void GLWidget::setZRotation(int angle)
{
    qNormalizeAngle(angle);
    if (angle != zRot) {
        zRot = angle;
        emit zRotationChanged(angle);
        updateGL();
    }
}

//! [6]
void GLWidget::initializeGL()
{
    glClearColor(0.0, 0.0, 0.0, 1.0);

    glEnable(GL_TEXTURE_2D);
    glEnable(GL_DEPTH_TEST);

//    QImage image, glImage;

//    QString fileName(":/textures/color_test_pattern.jpg");
//    if (!image.load(fileName)) {
//        std::cerr << "Failed to load image named: " << fileName.toStdString();

//        image = QImage(256, 256, QImage::Format_RGB32);
//        image.fill(0xffff00);
//    }
//    glImage = convertToGLFormat(image);

//    glGenTextures(1, &tex);

//    glActiveTexture(GL_TEXTURE0);
//    glBindTexture(GL_TEXTURE_2D, tex);
//    glTexImage2D(GL_TEXTURE_2D, 0, 3, glImage.width(), glImage.height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, glImage.bits());
//    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
//    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

    glMatrixMode(GL_MODELVIEW);

    if (!firstPass.addShaderFromSourceFile(QGLShader::Vertex, ":/vert/firstpass.vert")) {
        qDebug() << "Error compiling vertex shader." << endl;
        qDebug() << firstPass.log();

        return;
    }

    if (!firstPass.addShaderFromSourceFile(QGLShader::Fragment, ":/frag/firstpass.frag")) {
        qDebug() << "Error compiling fragment shader." << endl;
        qDebug() << firstPass.log();

        return;
    }

    if (!firstPass.link()) {
        qDebug() << "Error linking shader." << endl;
        qDebug() << firstPass.log();

        return;
    }

    if (!screen.addShaderFromSourceFile(QGLShader::Vertex, ":/vert/screen.vert")) {
        qDebug() << "Error compiling vertex shader." << endl;
        qDebug() << screen.log();

        return;
    }

    if (!screen.addShaderFromSourceFile(QGLShader::Fragment, ":/frag/screen.frag")) {
        qDebug() << "Error compiling fragment shader." << endl;
        qDebug() << screen.log();

        return;
    }

    if (!screen.link()) {
        qDebug() << "Error linking shader." << endl;
        qDebug() << screen.log();

        return;
    }
}
//! [6]

//! [7]
void GLWidget::paintGL()
{
    int width = this->width(), height = this->height();

    glEnable(GL_CULL_FACE);
    // reset the view to the identity
    glLoadIdentity();

    glScalef(3.0f, 3.0f, 3.0f);

    // move into the screen
    glTranslatef(0.0f, 0.0f, -3.0f);

    // rotate the cube by the rotation value
    glRotatef(xRot / 20, 1.0f, 0.0f, 0.0f);
    glRotatef(yRot / 20, 0.0f, 1.0f, 0.0f);
    glRotatef(zRot / 20, 0.0f, 0.0f, 1.0f);

    {
        QPainter frontFace(framebuffers[FRONT_FACE_BUFFER]);

        firstPass.bind();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        drawProxyGeometry();

        frontFace.end();
    }

    {
        QPainter backFace(framebuffers[BACK_FACE_BUFFER]);

        firstPass.bind();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glCullFace(GL_FRONT);
        drawProxyGeometry();
        glCullFace(GL_BACK);

        backFace.end();
    }

    runCuda( width, height );

    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, resultBuffer);

    // bind texture from PBO
    glBindTexture(GL_TEXTURE_2D, resultTexture);

    // Note: glTexSubImage2D will perform a format conversion if the
    // buffer is a different format from the texture. We created the
    // texture with format GL_RGBA8. In glTexSubImage2D we specified
    // GL_BGRA and GL_UNSIGNED_INT. This is a fast-path combination

    // Note: NULL indicates the data resides in device memory
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height,
                    GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    screen.bind();
    screen.setAttributeValue("texture", GL_TEXTURE0);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    drawTextureQuad();
}
//! [7]

//! [8]
void GLWidget::resizeGL(int width, int height)
{
    // Set the viewport given the resize event
    glViewport(0, 0, width, height);

    // Reset the Projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    perspectiveFrustum(45.0f, (GLfloat) width / (GLfloat) height, 0.1f, 100.0f);

    // Reset the Model View matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    QGLFramebufferObjectFormat format;
    format.setAttachment(QGLFramebufferObject::Depth);
    format.setInternalTextureFormat(GL_RGBA8);

    for (int i = 0; i < GLWidget::N_FRAMEBUFFERS; i++) {
        if (framebuffers[i]) {
            delete framebuffers[i];
        }

        framebuffers[i] = new QGLFramebufferObject(width, height, format);
    }

    if (resultBuffer) {
        glDeleteBuffers(1, &resultBuffer);
    }

    GLsizeiptr bufferSize = width * height * 4 * sizeof(GLubyte);

    glGenBuffers( 1, &resultBuffer );
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, resultBuffer );
    glBufferData( GL_PIXEL_UNPACK_BUFFER, bufferSize, NULL, GL_STREAM_DRAW );

    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );

    glGenTextures(1, &resultTexture);

    // Make this the current texture (remember that GL is state-based)
    glBindTexture(GL_TEXTURE_2D, resultTexture);

    // Allocate the texture memory. The last parameter is NULL since we only
    // want to allocate memory, not initialize it
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
              GL_BGRA,GL_UNSIGNED_BYTE, NULL);

    // Must set the filter mode, GL_LINEAR enables interpolation when scaling
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glBindTexture(GL_TEXTURE_2D, 0);

    registerCudaResources(framebuffers[0]->texture(), framebuffers[1]->texture(), resultBuffer);

    CHECK_GL_ERROR_DEBUG();
}
//! [8]

//! [9]
void GLWidget::mousePressEvent(QMouseEvent *event)
{
    lastPos = event->pos();
}
//! [9]

//! [10]
void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
    int dx = event->x() - lastPos.x();
    int dy = event->y() - lastPos.y();

    if (event->buttons() & Qt::LeftButton) {
        setXRotation(xRot + 8 * dy);
        setYRotation(yRot + 8 * dx);
    } else if (event->buttons() & Qt::RightButton) {
        setXRotation(xRot + 8 * dy);
        setZRotation(zRot + 8 * dx);
    }
    lastPos = event->pos();
}
//! [10]

//! [11]
void GLWidget::drawProxyGeometry()
{
    glEnable(GL_DEPTH_TEST);

    // construct the cube
    glBegin(GL_QUADS);

    glVertex3f(  0.5, -0.5, 0.5 );
    glVertex3f(  0.5,  0.5, 0.5 );
    glVertex3f( -0.5,  0.5, 0.5 );
    glVertex3f( -0.5, -0.5, 0.5 );

    glVertex3f( 0.5, -0.5, -0.5 );
    glVertex3f( 0.5,  0.5, -0.5 );
    glVertex3f( 0.5,  0.5,  0.5 );
    glVertex3f( 0.5, -0.5,  0.5 );

    glVertex3f( -0.5, -0.5,  0.5 );
    glVertex3f( -0.5,  0.5,  0.5 );
    glVertex3f( -0.5,  0.5, -0.5 );
    glVertex3f( -0.5, -0.5, -0.5 );

    glVertex3f(  0.5,  0.5,  0.5 );
    glVertex3f(  0.5,  0.5, -0.5 );
    glVertex3f( -0.5,  0.5, -0.5 );
    glVertex3f( -0.5,  0.5,  0.5 );

    glVertex3f(  0.5, -0.5, -0.5 );
    glVertex3f(  0.5, -0.5,  0.5 );
    glVertex3f( -0.5, -0.5,  0.5 );
    glVertex3f( -0.5, -0.5, -0.5 );

    glVertex3f( -0.5, -0.5, -0.5 );
    glVertex3f( -0.5,  0.5, -0.5 );
    glVertex3f(  0.5,  0.5, -0.5 );
    glVertex3f(  0.5, -0.5, -0.5 );

    glEnd();

    CHECK_GL_ERROR_DEBUG();
}
//! [11]

//! [12]
void GLWidget::drawTextureQuad()
{
    glDisable(GL_DEPTH_TEST);

    glBegin(GL_QUADS);

    glTexCoord2f( 0.0, 0.0 );
    glVertex3f( -1.0, -1.0, 0.0 );

    glTexCoord2f( 1.0, 0.0 );
    glVertex3f( 1.0, -1.0, 0.0 );

    glTexCoord2f( 1.0, 1.0 );
    glVertex3f( 1.0, 1.0, 0.0 );

    glTexCoord2f( 0.0, 1.0 );
    glVertex3f( -1.0, 1.0, 0.0 );

    glEnd();

    CHECK_GL_ERROR_DEBUG();
}

//! [12]
