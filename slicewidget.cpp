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

SliceWidget::SliceWidget(QWidget *parent)
{
    int n = SLICE_EDGELENGTH;
    m_sliceImage = new QImage(n,n,QImage::Format_RGB32);
    BGRA* bits = new BGRA[n*n];
    for(int j=0; j<n; j++){
        for(int i=0; i<n; i++){
            int offset = j*n+i;
            bits[offset] = BGRA(0, 20, 50, 255);
        }
    }
    memcpy(m_sliceImage->bits(), bits, n*n*sizeof(BGRA));
    delete[] bits;
}

SliceWidget::~SliceWidget()
{
}

void SliceWidget::renderSlice(SliceParameters sliceParameters,
                              BufferParameters bufferParameters,
                              canonicalOrientation orientation)
{
    cout << "SliceWidget::renderSlice: " << sliceParameters << endl;


    if(bufferParameters.height*bufferParameters.width != m_sizeY*m_sizeX){
        cout << "allocating new slice buffer" << endl;
        delete[] m_sliceBuffer;
        m_sliceBuffer = new float[bufferParameters.height*bufferParameters.width];
        m_sizeY = bufferParameters.height;
        m_sizeX = bufferParameters.width;
        cout << "new slice buffer allocated" << endl;
    }

    if(orientation == FREE_FORM){
        Matrix4x4 trans = getTransformationMatrix(sliceParameters);
        invoke_advanced_slice_kernel(m_sliceBuffer, bufferParameters, trans);


        // print a few examples
//        origin = Vector4(0, 0, 0, 0);

//        cout




    }else{
       invoke_slice_kernel(m_sliceBuffer, bufferParameters, sliceParameters, orientation);
    }

    cout << "Updating slice image" << endl;
    delete m_sliceImage;
    m_sliceImage = new QImage(bufferParameters.width, bufferParameters.height, QImage::Format_RGB32);
    BGRA* bits = new BGRA[bufferParameters.width*bufferParameters.height];
    int size = bufferParameters.width*bufferParameters.height;
    for(int j=0; j<bufferParameters.height; j++){
        for(int i=0; i<bufferParameters.width; i++){
            int offset = j*bufferParameters.height+i;
            unsigned int val =(unsigned int)( m_sliceBuffer[offset]*255 );
            bits[size-offset] = BGRA(val, val, val, 255);
        }
    }
    memcpy(m_sliceImage->bits(), bits, bufferParameters.width*bufferParameters.height*sizeof(BGRA));
    delete[] bits;
    cout << "Slice image updated" << endl;

}

void SliceWidget::paintEvent(QPaintEvent *)
{
    QPainter painter(this);
    if(m_sliceImage){
        painter.drawImage(QPoint(0,0), *m_sliceImage);
    }
    update();
}


float* SliceWidget::getSlice(size_t& height, size_t& width)
{
    height = m_sizeY;
    width = m_sizeX;
    return m_sliceBuffer;
}

void SliceWidget::saveSliceAs(QString fileName)
{
    if(m_sliceImage){
        m_sliceImage->save(fileName);
    }
}

Matrix4x4 SliceWidget::getTransformationMatrix(SliceParameters sliceParameters)
{
    assert(sliceParameters.theta >= -3.2f);
    assert(sliceParameters.theta < 3.2f); // ballpark estimate of 2 pi; might have rounding error, but this bound is tight enough
    assert(sliceParameters.phi >= -3.2f);
    assert(sliceParameters.phi < 3.2f);
    assert(sliceParameters.psi >= -3.2f);
    assert(sliceParameters.psi < 3.2f);

    Matrix4x4 center2origin = getTransMat( Vector4(-0.5, -0.5, -0.5, 0) );
    Matrix4x4 rotX = getRotXMat(sliceParameters.theta);
    Matrix4x4 rotY = getRotYMat(sliceParameters.phi);
    Matrix4x4 rotZ = getRotZMat(sliceParameters.psi);
    Matrix4x4 trans = getTransMat( Vector4(sliceParameters.dx, sliceParameters.dy, sliceParameters.dz, 1.0) );
    Matrix4x4 origin2center = getTransMat( Vector4(0.5, 0.5, 0.5, 0) );
    Matrix4x4 compound = origin2center * trans * rotX * rotY * rotZ * center2origin;

    return compound;
}












