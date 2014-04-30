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

//#include <QtWidgets>

#include <QApplication>
#include <QtGui>

#include "glwidget.h"
#include "window.h"
#include "slicewidget.h"
#include "params.h"

#include <iostream>

using namespace std;


//! [0]
Window::Window()
{
    glWidget = new GLWidget;


//! [0]

//! [1]
    QHBoxLayout *mainLayout = new QHBoxLayout;{
        QVBoxLayout *leftColumn = new QVBoxLayout();{
            m_sliceWidget = new SliceWidget();{
            }leftColumn->addWidget(m_sliceWidget);
            QVBoxLayout *controlBox = new QVBoxLayout();{
                QHBoxLayout *sliceBox = new QHBoxLayout();{
                    QLabel* sliceLabel = new QLabel("Slicing tools");
                    sliceBox->addWidget(sliceLabel);
                }controlBox->addLayout(sliceBox);
                QHBoxLayout *cullBox = new QHBoxLayout();{
                    QLabel* cullLabel= new QLabel("Culling tools");
                    cullBox->addWidget(cullLabel);
                }controlBox->addLayout(cullBox);
                QHBoxLayout *loadBox = new QHBoxLayout;{
                    m_loadButton = new QPushButton("Load Volume");
                    m_lineEdit = new QLineEdit();
                    m_lineEdit->setFocus();
                    m_lineEdit->clear();
                    m_lineEdit->insert(g_texture_paths[0]);
                    m_examples = new QComboBox();
                    for(int i=0; i<N_DEFAULT_TEXTURES; i++){
                        m_examples->addItem(g_texture_names[i]);
                    }
                    connect(m_examples, SIGNAL(activated(int)), this, SLOT(textureSelectionChanged(int)));
                    loadBox->addWidget(m_lineEdit);
                    loadBox->addWidget(m_examples);
                    loadBox->addWidget(m_loadButton);
                    connect(m_loadButton, SIGNAL(clicked()), this, SLOT(loadButtonClicked()));
                }controlBox->addLayout(loadBox);

                QVBoxLayout* simpleSliderBox = new QVBoxLayout();{
                    QLabel* simpleSliderLabel = new QLabel("Simple Slicer");
                    simpleSliderBox->addWidget(simpleSliderLabel);

                    QHBoxLayout* radioBox = new QHBoxLayout();{
                        m_canonicalOrientationButtons = new QRadioButton*[N_CANONICAL_ORIENTATIONS];
                        for(int i=0; i<N_CANONICAL_ORIENTATIONS; i++){
                            QLabel* radioLabel = new QLabel(g_canonical_orientation_captions[i]);
                            radioBox->addWidget(radioLabel);
                            m_canonicalOrientationButtons[i] = new QRadioButton();
                            radioBox->addWidget(m_canonicalOrientationButtons[i]);
                        }
                    }simpleSliderBox->addLayout(radioBox);
                    m_canonicalSliceSlider = new QSlider(Qt::Horizontal);
                    m_canonicalSliceSlider->setRange(0,SLICE_EDGELENGTH);
                    simpleSliderBox->addWidget(m_canonicalSliceSlider);
                    connect(m_canonicalSliceSlider, SIGNAL(valueChanged(int)), this, SLOT(canonicalSliceSliderChanged(int)));
                    m_sliceRenderButton = new QPushButton("Render Slice!");
                    connect(m_sliceRenderButton, SIGNAL(clicked()), this, SLOT(renderSliceButtonClicked()));
                    simpleSliderBox->addWidget(m_sliceRenderButton);
                    m_slicePrintButton = new QPushButton("Print Slice!");
                    connect(m_slicePrintButton, SIGNAL(clicked()), this, SLOT(printSliceButtonClicked()));
                    simpleSliderBox->addWidget(m_slicePrintButton);
                }controlBox->addLayout(simpleSliderBox);

//                QVBoxLayout* sliceSliderBox = new QVBoxLayout();{
//                    QLabel* sliceSliderLabel = new QLabel("Slicer");
//                    sliceSliderBox->addWidget(sliceSliderLabel);
//                    m_sliceSliders = new QSlider*[N_SLICE_SLIDERS];
//                    for(int i=0; i<N_SLICE_SLIDERS; i++){
//                        QHBoxLayout* sliderBox = new QHBoxLayout();{
//                            QLabel* curLabel = new QLabel(g_slice_slider_captions[i]);
//                            sliderBox->addWidget(curLabel);
//                            m_sliceSliders[i] = new QSlider(Qt::Horizontal);
//                            sliderBox->addWidget(m_sliceSliders[i]);
//                        } sliceSliderBox->addLayout(sliderBox);
//                    }
//                }controlBox->addLayout(sliceSliderBox);

            } leftColumn->addLayout(controlBox);
        }mainLayout->addLayout(leftColumn);
    }mainLayout->addWidget(glWidget);

    setLayout(mainLayout);
    setWindowTitle(tr("VolumeViz"));
}
//! [1]

void Window::keyPressEvent(QKeyEvent *e)
{
    if (e->key() == Qt::Key_Escape)
        close();
    else
        QWidget::keyPressEvent(e);
}

void Window::textureSelectionChanged(int idx)
{
    char* new_path = g_texture_paths[idx];
    m_lineEdit->clear();
    m_lineEdit->insert(QString(new_path));

}

void Window::loadButtonClicked()
{
    const char* path = m_lineEdit->text().toUtf8().constData();
    glWidget->loadVolume(path);
}

void Window::renderSliceButtonClicked()
{
    // TODO: get specifications
    float dx = 0.0;
    float dy = 0.0;
    float dz = 0.5;

    float theta = 0.0;
    float phi = 0.0;
    float psi = 0.0;

    int height = SLICE_EDGELENGTH;
    int width = SLICE_EDGELENGTH;

    SliceParameters sliceParameters(dx, dy, dz, theta, phi, psi);
    BufferParameters bufferParameters(height, width);
    canonicalOrientation orientation = SAGITTAL;

    cout << "Window: Rendering slice" << endl;
    m_sliceWidget->renderSlice(sliceParameters, bufferParameters, orientation);
    cout << "Window: Slice rendered" << endl;
}

void Window::printSliceButtonClicked()
{
    cout << "Ready to print slice (not yet doing anything)" << endl;

    size_t height;
    size_t width;

    float* data = m_sliceWidget->getSlice(height, width);

    for(int j=0; j<height; j++){
        for(int i=0; i<width; i++){
            int offset = j*height+i;
            cout << data[offset] << " ";
        }
        cout << endl;
    }
}

void Window::canonicalSliceSliderChanged(int val)
{
    // TODO: get specifications
    float dx = 0.0;
    float dy = 0.0;
    float dz = 0.0;

    canonicalOrientation orientation = HORIZONTAL;

    switch(orientation) {

    case SAGITTAL:
        dz = ((float)val)/((float)SLICE_EDGELENGTH);
        break;

    case HORIZONTAL:
        dy = ((float)val)/((float)SLICE_EDGELENGTH);
        break;

    case CORONAL:
        dx = ((float)val)/((float)SLICE_EDGELENGTH);
        break;
    }


    float theta = 0.0;
    float phi = 0.0;
    float psi = 0.0;

    int height = SLICE_EDGELENGTH;
    int width = SLICE_EDGELENGTH;

    SliceParameters sliceParameters(dx, dy, dz, theta, phi, psi);
    BufferParameters bufferParameters(height, width);

    cout << "Window: Rendering slice" << endl;
    m_sliceWidget->renderSlice(sliceParameters, bufferParameters, orientation);
    cout << "Window: Slice rendered" << endl;
}










