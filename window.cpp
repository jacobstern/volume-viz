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

Window::Window()
{
    // EVERYTHING GOES IN HERE
    QHBoxLayout *mainLayout = new QHBoxLayout;{


        // LEFT HALF OF MAIN WINDOW
        QVBoxLayout *leftColumn = new QVBoxLayout();{

            // UPPER HALF OF LEFT COLUMN
            m_sliceWidget = new SliceWidget();{
            }
            m_sliceWidget->setFixedSize(SLICE_SIZE, SLICE_SIZE);
            leftColumn->addWidget(m_sliceWidget);

            // LOWER HALF OF LEFT COLUMN
            QVBoxLayout *controlBox = new QVBoxLayout();{

                QVBoxLayout *loadBox = new QVBoxLayout();{
                    loadBox->addWidget(new QLabel("Load Model:"));
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

                    QGridLayout* loadGrid = new QGridLayout();
                    loadGrid->addWidget(m_lineEdit, 0, 0, 1,2);
                    loadGrid->addWidget(m_examples, 1, 0);
                    loadGrid->addWidget(m_loadButton, 1,1);
                    loadBox->addLayout(loadGrid);

                    connect(m_loadButton, SIGNAL(clicked()), this, SLOT(loadButtonClicked()));

                }
                controlBox->addLayout(loadBox);

                // SLICE CONTROLS
                QVBoxLayout* sliceBox = new QVBoxLayout();{
                    m_sliceTab = new QTabWidget();{

                        // just the label
                        QLabel* sliceLabel = new QLabel("Slice Model");
                        sliceBox->addWidget(sliceLabel);

                        // canonical orientations slicer
                        QVBoxLayout* simpleSliderBox = new QVBoxLayout();{
                                m_canonicalOrientationBox = new QGroupBox();{
                                    QVBoxLayout* radioBox = new QVBoxLayout();{
                                        m_canonicalOrientationButtons = new QRadioButton*[N_CANONICAL_ORIENTATIONS];
                                        for(int i=0; i<N_CANONICAL_ORIENTATIONS; i++){
                                            QLabel* radioLabel = new QLabel(g_canonical_orientation_captions[i]);
                                            radioBox->addWidget(radioLabel);
                                            m_canonicalOrientationButtons[i] = new QRadioButton();
                                            radioBox->addWidget(m_canonicalOrientationButtons[i]);
                                            connect(m_canonicalOrientationButtons[i], SIGNAL(clicked(bool)), this, SLOT(renderSlice()));
                                        }
                                    }m_canonicalOrientationBox->setLayout(radioBox);
                                    m_canonicalOrientationButtons[SAGITTAL]->setChecked(true);
                                }simpleSliderBox->addWidget(m_canonicalOrientationBox);
                                m_canonicalSliceSlider = new QSlider(Qt::Horizontal);
                                m_canonicalSliceSlider->setRange(0,SLICE_EDGELENGTH);
                                simpleSliderBox->addWidget(m_canonicalSliceSlider);
                                m_canonicalSliceSlider->setValue(100);
                                connect(m_canonicalSliceSlider, SIGNAL(valueChanged(int)), this, SLOT(renderSlice()));
                        }
                        QWidget* simpleSliderWidget = new QWidget();
                        simpleSliderWidget->setLayout(simpleSliderBox);
                        m_sliceTab->addTab(simpleSliderWidget, "Simple");

                        // arbitrary slicer
                        QVBoxLayout* sliceSliderBox = new QVBoxLayout();{
                            QLabel* sliceSliderLabel = new QLabel("Slicer");
                            sliceSliderBox->addWidget(sliceSliderLabel);
                            m_sliceSliders = new QSlider*[N_SLICE_SLIDERS];
                            for(int i=0; i<N_SLICE_SLIDERS; i++){
                                QHBoxLayout* sliderBox = new QHBoxLayout();{
                                    QLabel* curLabel = new QLabel(g_slice_slider_captions[i]);
                                    sliderBox->addWidget(curLabel);
                                    m_sliceSliders[i] = new QSlider(Qt::Horizontal);
                                    m_sliceSliders[i]->setRange(SLICE_SLIDER_MIN, SLICE_SLIDER_MAX);
                                    sliderBox->addWidget(m_sliceSliders[i]);
                                    connect(m_sliceSliders[i], SIGNAL(valueChanged(int)), this, SLOT(renderSlice()));
                                    m_sliceSliders[i]->setValue(SLICE_SLIDER_INIT);
                                } sliceSliderBox->addLayout(sliderBox);
                            }
                        }
                        QWidget* sliceSliderWidget = new QWidget();
                        sliceSliderWidget->setLayout(sliceSliderBox);
                        m_sliceTab->addTab(sliceSliderWidget, "Pro");
                    }
                    sliceBox->addWidget(m_sliceTab);
                }
                controlBox->addLayout(sliceBox);

                // OPTION TO SAVE SLICE, UNIVERSAL TO ALL TOOLS
                QVBoxLayout *saveBox = new QVBoxLayout();{
                    saveBox->addWidget(new QLabel("Save Slice:"));

                    m_fileFormats = new QComboBox();
                    for(int i=0; i<N_OUTPUT_FILE_FORMATS; i++){
                        m_fileFormats->addItem(g_output_file_formats[i]);
                    }

                    m_sliceSaveButton = new QPushButton("Save slice");
                    connect(m_sliceSaveButton, SIGNAL(clicked()), this, SLOT(saveSliceButtonClicked()));
                    m_sliceSavePath = new QLineEdit();
                    m_sliceSavePath->clear();
                    m_sliceSavePath->insert(DEFAULT_SAVE_PATH);

                    QGridLayout* saveGrid = new QGridLayout();
                    saveGrid->addWidget(m_sliceSavePath, 0, 0, 1, 2);
                    saveGrid->addWidget(m_fileFormats, 1, 0);
                    saveGrid->addWidget(m_sliceSaveButton, 1, 1);

                    saveBox->addLayout(saveGrid);
                }
                controlBox->addLayout(saveBox);

            }
            leftColumn->addLayout(controlBox);

        }
        QWidget* leftColumnWidget = new QWidget();
        leftColumnWidget->setLayout(leftColumn);
        leftColumnWidget->setFixedWidth(LEFT_COLUMN_WIDTH);
        mainLayout->addWidget(leftColumnWidget);

        // RIGHT HALF OF THE MAIN WINDOW
        glWidget = new GLWidget();
        glWidget->setFixedSize(RENDER_SIZE, RENDER_SIZE);
        mainLayout->addWidget(glWidget);

    }

    // SET MAIN LAYOUT OF WINDOW
    setLayout(mainLayout);
    setWindowTitle(tr("VolumeViz"));
}


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

void Window::saveSliceButtonClicked()
{
    cout << "Trying to save image" << endl;
    m_sliceWidget->saveSliceAs(m_sliceSavePath->text());
}

void Window::renderSlice(int value)
{
    // TODO: get specifications
    float dx = 0.0;
    float dy = 0.0;
    float dz = 0.0;
    float theta = 0.0;
    float phi = 0.0;
    float psi = 0.0;

    canonicalOrientation orientation;

    float val = m_canonicalSliceSlider->value();

    for(int i=0; i<N_CANONICAL_ORIENTATIONS; i++){
        if(m_canonicalOrientationButtons[i]->isChecked()){
            orientation = (canonicalOrientation)i;
            break;
        }
    }

    // TODO: Get active tab

    if(m_sliceTab->currentIndex() == 0){

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

        default:
            cout << "ERROR: Invalid default orientation " << endl;
            assert(false);
        }

    }else if(m_sliceTab->currentIndex() == 1){

        dx = ((float)m_sliceSliders[0]->value())/((float)SLICE_SLIDER_MAX - SLICE_SLIDER_MIN) * 2;
        dy = ((float)m_sliceSliders[1]->value())/((float)SLICE_SLIDER_MAX - SLICE_SLIDER_MIN) * 2;
        dz = ((float)m_sliceSliders[2]->value())/((float)SLICE_SLIDER_MAX - SLICE_SLIDER_MIN) * 2;

        theta = ((float)m_sliceSliders[3]->value())/((float)SLICE_SLIDER_MAX - SLICE_SLIDER_MIN) * 2 * M_PI * 2;
        phi = ((float)m_sliceSliders[4]->value())/((float)SLICE_SLIDER_MAX - SLICE_SLIDER_MIN) * 2 * M_PI * 2;
        psi = ((float)m_sliceSliders[5]->value())/((float)SLICE_SLIDER_MAX - SLICE_SLIDER_MIN) * 2 * M_PI * 2;

    }else{
        cerr << "ERROR: Invalid index for slice tab" << endl;
        assert(false);
    }

    int height = SLICE_EDGELENGTH;
    int width = SLICE_EDGELENGTH;

    SliceParameters sliceParameters(dx, dy, dz, theta, phi, psi);
    BufferParameters bufferParameters(height, width);

    m_sliceWidget->renderSlice(sliceParameters, bufferParameters, orientation);
}










