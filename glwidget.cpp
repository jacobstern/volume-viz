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

#include "glwidget.h"
#include "qtlogo.h"
#include "kernel.cuh"

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

static inline float nrm( float glCoord )
{
    return (glCoord + 1.f) / 2.f;
}

//! [0]
GLWidget::GLWidget(QWidget *parent)
    : QGLWidget(QGLFormat(QGL::SampleBuffers), parent),
      font("Deja Vu Sans Mono", 8, 4), fovX(0.f), fovY(0.f), resolutionScale(4),
      transferPreset(TRANSFER_PRESET_DEFAULT), phongShading(false), filterOutput(true), scaleObject(1.f, 1.f, 1.f)
{
    logo = 0;

    qtGreen = QColor::fromCmykF(0.40, 0.0, 1.0, 0.0);
    qtPurple = QColor::fromCmykF(0.39, 0.39, 0.0, 0.0);

    for (int i = 0; i < GLWidget::N_FRAMEBUFFERS; i++) {
        framebuffers[i] = NULL;
    }

    resultBuffer = 0;

    camera = new Camera( 0 );
    camera->setPosition( QVector3D(0, 0, -4.f) );
    camera->lookAt( QVector3D(0, 0, 0) );

    didStartDragging = false;
    isDragging = false;
    hasCuttingPlane = false;

    renderingDirty = true;
}
//! [0]

void GLWidget::setPhongShading(bool shading)
{
    if (shading != phongShading) {
        phongShading  = shading;
        renderingDirty = true;

        update();
    }
}

//! [1]
GLWidget::~GLWidget()
{
    for (int i = 0; i < GLWidget::N_FRAMEBUFFERS; i++) {
        if (framebuffers[i]) {
            delete framebuffers[i];
        }
    }

    delete camera;
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
    return QSize(768, 768);
}
//! [4]

//! [6]
void GLWidget::initializeGL()
{
    glClearColor(0.0, 0.0, 0.0, 1.0);

    glEnable(GL_TEXTURE_2D);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LINE_SMOOTH);

    glMatrixMode(GL_MODELVIEW);

    loadShaderProgram( firstPass, tr("firstpass") );
    loadShaderProgram( screen, tr("screen") );
    loadShaderProgram( ui, tr("ui") );

    initCuda();
    loadVolume(DEFAULT_TEXTURE_PATH);
}
//! [6]

//! [7]
void GLWidget::paintGL()
{
    int width = this->width(), height = this->height();

    glEnable(GL_CULL_FACE);

    camera->updateView();

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix(); // Object Transformation
    glScalef( scaleObject.x(), scaleObject.y(), scaleObject.z() );

    {
        QPainter frontFace(framebuffers[FRONT_FACE_BUFFER]);

        firstPass.bind();

        glClearColor( 0.f, 0.f, 0.f, 0.f );
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        drawProxyGeometry();

        frontFace.end();
    }

    {
        QPainter backFace(framebuffers[BACK_FACE_BUFFER]);

        firstPass.bind();

        glClearColor( 0.f, 0.f, 0.f, 0.f );
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glCullFace(GL_FRONT);
        drawProxyGeometry();
        glCullFace(GL_BACK);

        backFace.end();
    }

    glPopMatrix(); // Object transformation

    struct slice_params sliceParams;

    if (hasCuttingPlane) {
        sliceParams.type = SLICE_PLANE;

        sliceParams.params[0] = cutPoint.x();
        sliceParams.params[1] = cutPoint.y();
        sliceParams.params[2] = cutPoint.z();

        sliceParams.params[3] = cutNormal.x();
        sliceParams.params[4] = cutNormal.y();
        sliceParams.params[5] = cutNormal.z();
    }
    else {
        sliceParams.type = SLICE_NONE;
    }

    QVector3D pos = camera->getPosition();

    struct camera_params cameraParams;

    cameraParams.origin[0] = pos.x();
    cameraParams.origin[1] = pos.y();
    cameraParams.origin[2] = pos.z();

    cameraParams.fovX      = fovX;
    cameraParams.fovY      = fovY;

    struct shading_params shadingParams;

    shadingParams.transferPreset = transferPreset;
    shadingParams.phongShading   = phongShading;

    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, resultBuffer);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, resultTexture);

    clock_t t = clock();

    if (renderingDirty) {
        runCuda( width / resolutionScale, height / resolutionScale, sliceParams, cameraParams, shadingParams, m_volumeArray);
    }

    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width / resolutionScale, height / resolutionScale,
                    GL_RGBA, GL_UNSIGNED_BYTE, NULL);

    screen.bind();
    screen.setAttributeValue("texture", GL_TEXTURE0);

    glClearColor( 0.f, 0.f, 0.f, 1.f );
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    drawTextureQuad();

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    if (isDragging)
        showDragUI();


    if (renderingDirty) {
        glFinish();
        t = clock() - t;
        lastRenderTime = (float) t / CLOCKS_PER_SEC;

        renderingDirty = false;
    }


    glUseProgram( 0 );
    glColor4f(1.f, 1.f, 1.f, 1.f);
    renderText(10, 20, "Resolution: "  + QString::number( width / resolutionScale )
                                       + " x " + QString::number( height / resolutionScale ), font);
    renderText(10, 34, "Render time: " + QString::number( (double) lastRenderTime ) + " sec", font);
}
//! [7]

//! [8]
void GLWidget::resizeGL(int width, int height)
{
    // Set the viewport given the resize event
    glViewport(0, 0, width, height);

    float aspect = (float) width / (float) height;

    perspective = QMatrix4x4();
    perspective.perspective( 45.f, aspect, 0.1f, 100.f );

    fovY = 45.f;
    fovX = fovY * aspect;

    // Reset the Projection matrix
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixd( perspective.constData() );

    glMatrixMode(GL_MODELVIEW);

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

    GLsizeiptr bufferSize = width * height * 4 * sizeof(GLubyte) / (resolutionScale * resolutionScale);

    glGenBuffers( 1, &resultBuffer );
    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, resultBuffer );
    glBufferData( GL_PIXEL_UNPACK_BUFFER, bufferSize, NULL, GL_STREAM_DRAW );

    glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0 );

    glGenTextures(1, &resultTexture);

    // Make this the current texture (remember that GL is state-based)
    glBindTexture(GL_TEXTURE_2D, resultTexture);

    // Allocate the texture memory. The last parameter is NULL since we only
    // want to allocate memory, not initialize it
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width / resolutionScale, height / resolutionScale, 0,
              GL_BGRA,GL_UNSIGNED_BYTE, NULL);

    GLenum filterMode = filterOutput ? GL_LINEAR : GL_NEAREST;

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filterMode);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filterMode);

    glBindTexture(GL_TEXTURE_2D, 0);

    registerCudaResources(framebuffers[0]->texture(), framebuffers[1]->texture(), resultBuffer);

    CHECK_GL_ERROR_DEBUG();
}
//! [8]

//! [9]
void GLWidget::mousePressEvent(QMouseEvent *event)
{
    QVector2D window( event->x() / (float) width(), event->y() / (float) height() );

    if (event->buttons() & Qt::LeftButton) {
        dragStart = window;
        didStartDragging = true;

        hasCuttingPlane = false;
        renderingDirty  = true;
    }
    else if (event->buttons() & Qt::RightButton) {
        lastPos = event->pos();
    }
}
//! [9]

//! [10]
void GLWidget::mouseMoveEvent(QMouseEvent *event)
{
    int dx = event->x() - lastPos.x();
    int dy = event->y() - lastPos.y();

    if (event->buttons() & Qt::RightButton) {
        // Stolen from CS224 Chameleon
        QVector3D position = camera->getPosition();

        float r = position.length(),
              theta = acos( double(position.y() / r) ) - dy / 200.f,
              phi = atan2( position.z(), position.x() ) + dx / 200.f;

        if (theta < 0.1f)
            theta = 0.1f;

        if (theta > M_PI - 0.1f)
            theta = M_PI - 0.1f;

        camera->setPosition( QVector3D( r * sin(theta) * cos(phi), r * cos(theta), r * sin(theta) * sin(phi) ) );
        camera->lookAt( QVector3D(0.f, 0.f, 0.f) );

        renderingDirty = true;
    } else if (event->buttons() & Qt::MiddleButton && hasCuttingPlane) {
        cutPoint = cutPoint + cutRight * dx / ( width()  );
        cutPoint = cutPoint + cutUp    * dy / ( height() );

        renderingDirty = true;

        onUpdateSlicePlane();
    }


    lastPos = event->pos();

    if ( (event->buttons() & Qt::LeftButton) && didStartDragging) {
        isDragging = true;
        dragEnd = QVector2D( lastPos.x() / (float) width(), lastPos.y() / (float) height() );
    }

    update();
}
//! [10]

void GLWidget::mouseReleaseEvent(QMouseEvent *event)
{
    if (isDragging) {
        QVector2D window( event->x() / (float) width(), event->y() / (float) height() );

        bool success;

        QMatrix4x4 inverse = ( perspective * camera->transformation() ).inverted( &success );

        QVector4D front = inverse * QVector4D( glc( window.x() ), -glc( window.y() ), -1.f, 1.f ),
                  back  = inverse * QVector4D( glc( window.x() ), -glc( window.y() ),  1.f, 1.f ),
                  side  = inverse * QVector4D( glc( dragStart.x() ), -glc( dragStart.y() ),
                                             -1.f, 1.f ),
                  up    = inverse * QVector4D( 0.f, -1.f, 0.f, 0.f ),
                  right = inverse * QVector4D( 1.f,  0.f, 0.f, 0.f );

        front /= front.w();
        back  /= back.w();
        side  /= side.w();

        Q_ASSERT( success );

        QVector4D a = (back - front).normalized(),
                  b = (side - front).normalized();

        QVector3D p( front.x(), front.y(), front.z() ),
                  n( QVector3D::crossProduct(
                       QVector3D( a.x(), a.y(), a.z() ),
                       QVector3D( b.x(), b.y(), b.z() ) ) );

        p.setX( nrm( p.x() ) );
        p.setY( nrm( p.y() ) );
        p.setZ( nrm( p.z() ) );

        hasCuttingPlane = true;
        renderingDirty  = true;

        cutPoint = p;
        cutNormal = n;

        cutUp    = QVector3D( up.x(), up.y(), up.z() );
        cutRight = QVector3D( right.x(), right.y(), right.z() );

        isDragging = false;
        didStartDragging = false;

        onUpdateSlicePlane();
    }

    update();
}

//! [11]
void GLWidget::drawProxyGeometry()
{
    glEnable(GL_DEPTH_TEST);

    // construct the cube
    glBegin(GL_QUADS);

    glVertex3f(  1.0f, -1.0f, 1.0f );
    glVertex3f(  1.0f,  1.0f, 1.0f );
    glVertex3f( -1.0f,  1.0f, 1.0f );
    glVertex3f( -1.0f, -1.0f, 1.0f );

    glVertex3f( 1.0f, -1.0f, -1.0f );
    glVertex3f( 1.0f,  1.0f, -1.0f );
    glVertex3f( 1.0f,  1.0f,  1.0f );
    glVertex3f( 1.0f, -1.0f,  1.0f );

    glVertex3f( -1.0f, -1.0f,  1.0f );
    glVertex3f( -1.0f,  1.0f,  1.0f );
    glVertex3f( -1.0f,  1.0f, -1.0f );
    glVertex3f( -1.0f, -1.0f, -1.0f );

    glVertex3f(  1.0f,  1.0f,  1.0f );
    glVertex3f(  1.0f,  1.0f, -1.0f );
    glVertex3f( -1.0f,  1.0f, -1.0f );
    glVertex3f( -1.0f,  1.0f,  1.0f );

    glVertex3f(  1.0f, -1.0f, -1.0f );
    glVertex3f(  1.0f, -1.0f,  1.0f );
    glVertex3f( -1.0f, -1.0f,  1.0f );
    glVertex3f( -1.0f, -1.0f, -1.0f );

    glVertex3f( -1.0f, -1.0f, -1.0f );
    glVertex3f( -1.0f,  1.0f, -1.0f );
    glVertex3f(  1.0f,  1.0f, -1.0f );
    glVertex3f(  1.0f, -1.0f, -1.0f );

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

void GLWidget::wheelEvent(QWheelEvent *event)
{
    int delta = event->delta();

    if (delta != 0) {
        QVector3D look = camera->getLook(),
                position = camera->getPosition();
        camera->setPosition( position + look *  (delta / 200.f) );

        update();
    }

    renderingDirty = true;
}

void GLWidget::showDragUI()
{
    glDisable( GL_DEPTH_TEST );

    ui.bind();

    glLineWidth( 2.f );
    glColor4f( 1.f, 0.2f, 0.2f, 1.f );

    glBegin( GL_LINES );
    {
        glVertex2f( dragStart.x() * 2.f - 1.f, -dragStart.y() * 2.f + 1.f );
        glVertex2f( dragEnd.x()   * 2.f - 1.f, -dragEnd.y()   * 2.f + 1.f );
    }
    glEnd();

    glEnable( GL_DEPTH_TEST );

    CHECK_GL_ERROR_DEBUG();
}

void GLWidget::loadShaderProgram(QGLShaderProgram &program, QString name)
{
    if (!program.addShaderFromSourceFile(QGLShader::Vertex, tr(":/vert/") + name + tr(".vert") )) {
        qDebug() << "Error compiling vertex shader." << endl;
        qDebug() << program.log();

        return;
    }

    if (!program.addShaderFromSourceFile(QGLShader::Fragment, tr(":/frag/") + name + tr(".frag") )) {
        qDebug() << "Error compiling fragment shader." << endl;
        qDebug() << program.log();

        return;
    }

    if (!program.link()) {
        qDebug() << "Error linking shader." << endl;
        qDebug() << program.log();

        return;
    }
}


void GLWidget::loadVolume(const char* path)
{
    cout << "Generating mock voltex" << endl;
    m_volgen = new VolumeGenerator(0,0,0);

    cout << "loading brain from file " << path << endl;
    m_volgen->loadfrom_raw(path, true);
    cout << "brain has been loaded from file" << endl;

    if (QString(path).endsWith("engine.t3d")) {
        transferPreset = TRANSFER_PRESET_ENGINE;
        scaleObject = QVector3D(1.f, 1.f, 1.f);
    }
    else if (QString(path).endsWith("head.t3d")) {
        transferPreset = TRANSFER_PRESET_MRI;
        scaleObject = QVector3D(1.f, 1.f, 0.8f);
    }
    else {
        transferPreset = TRANSFER_PRESET_DEFAULT;
        scaleObject = QVector3D(1.f, 1.f, 1.f);
    }

    size_t size;
    byte* texels = m_volgen->getBytes(size);
    cout << "size: " << size << endl;

    cout << "Loading mock voltex into CUDA" << endl;
    cudaLoadVolume(texels, size, m_volgen->getDims(), &m_volumeArray);
    cout << "Mock voltex has been loaded into CUDA" << endl;

    delete m_volgen;

    renderingDirty  = true;
    hasCuttingPlane = false;

    update();
}

void GLWidget::onUpdateSlicePlane()
{
    // Update slice in SliceWidget
    qDebug() << "TODO: update slice widget";
}


