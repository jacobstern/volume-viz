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

#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QGLWidget>
#include <QGLShaderProgram>
#include <QGLFramebufferObject>
#include <QGLPixelBuffer>

#include <QTime>

#include <cuda.h>
#include <cuda_gl_interop.h>

#include "camera.h"
#include "volumegenerator.h"
#include "params.h"

class QtLogo;

//! [0]
class GLWidget : public QGLWidget
{
    Q_OBJECT

    friend class Window;

public:
    GLWidget(QWidget *parent = 0);
    ~GLWidget();

    QSize minimumSizeHint() const;
    QSize sizeHint() const;

    const static int N_FRAMEBUFFERS = 2;


    void loadVolume(const char* path);
//! [0]

public slots:
    void setPhongShading(bool);
    void setSliceVisualization(sliceVisualization);
    void invertCrossSection();
    void setSliceCanonical(canonicalOrientation orientation, float displace);
    void setSlicePro(Vector3 offset, Vector3 normal);

//! [2]
protected:
    void initializeGL();
    void paintGL();
    void resizeGL(int width, int height);
    void mousePressEvent(QMouseEvent *event);
    void mouseReleaseEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void wheelEvent(QWheelEvent *event);

    void onUpdateSlicePlane();
//! [2]

//! [3]
private:
    Camera *camera;


    canonicalOrientation lastCanonicalOrientation;

    bool didStartDragging, isDragging;
    QVector2D dragStart, dragEnd;

    bool hasCuttingPlane;
    bool hasPlaneFromImage;
    QVector3D cutPoint, cutNormal, cutUp, cutRight;

    sliceVisualization currentSliceVisualisation;

    void drawProxyGeometry();
    void drawTextureQuad();
    void showDragUI();

    void loadShaderProgram(QGLShaderProgram &program, QString name);

    QtLogo *logo;
    QPoint lastPos;
    QColor qtGreen;
    QColor qtPurple;
    GLuint resultBuffer, resultTexture;
    QGLShaderProgram firstPass, screen, ui;
    QGLFramebufferObject *framebuffers[GLWidget::N_FRAMEBUFFERS];

    int resolutionScale;

    int transferPreset;
    bool  phongShading;

    bool filterOutput;
    bool flipCrossSection;

    bool renderingDirty;
    float lastRenderTime;

    QFont font; // font for rendering text

    QMatrix4x4 perspective;
    float fovX, fovY;

    QVector3D     scaleObject;

    VolumeGenerator* m_volgen;

    cudaArray* m_volumeArray;

};
//! [3]

#endif
