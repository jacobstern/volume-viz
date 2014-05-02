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

#ifndef SliceWidget_H
#define SliceWidget_H

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

#include "cs123math/CS123Algebra.h"


class SliceWidget : public QWidget
{
    Q_OBJECT

public:
    SliceWidget(QWidget *parent = 0);
    ~SliceWidget();

//    QSize minimumSizeHint() const;
//    QSize sizeHint() const;

    void renderSlice(SliceParameters sliceParameters,
                     BufferParameters bufferParameters,
                     canonicalOrientation orientation,
                     float3 scale);

    void saveSliceAs(QString fileName);

    float* getSlice(size_t& height, size_t& width);

    Matrix4x4 getTransformationMatrix(SliceParameters sliceParameters);

protected:
    virtual void paintEvent(QPaintEvent *);

private:
    float* m_sliceBuffer = NULL;
    size_t m_sizeX = 0;
    size_t m_sizeY = 0;

    QImage* m_sliceImage;

};


#endif
