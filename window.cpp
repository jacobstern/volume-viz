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

#include "numberedit.h"

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
                                }
                                simpleSliderBox->addWidget(m_canonicalOrientationBox);

                                QHBoxLayout* canonicalWrapperBox = new QHBoxLayout();{
                                    m_canonicalSliceSlider = new QSlider(Qt::Horizontal);
                                    m_canonicalSliceSlider->setRange(0,SLICE_EDGELENGTH);
                                    m_canonicalSliceSlider->setValue(100);
                                    canonicalWrapperBox->addWidget(m_canonicalSliceSlider);
                                    NumberEdit* canonicalEdit = new NumberEdit();
                                    canonicalEdit->displayInteger(SLICE_SLIDER_INIT);
                                    canonicalWrapperBox->addWidget(canonicalEdit);
                                    canonicalEdit->setFixedWidth(50);
                                    connect(m_canonicalSliceSlider, SIGNAL(valueChanged(int)), canonicalEdit, SLOT(displayInteger(int)));


                                    connect(m_canonicalSliceSlider, SIGNAL(valueChanged(int)), this, SLOT(renderSlice()));
                                }
                                simpleSliderBox->addLayout(canonicalWrapperBox);

                        }
                        QWidget* simpleSliderWidget = new QWidget();
                        simpleSliderWidget->setLayout(simpleSliderBox);
                        m_sliceTab->addTab(simpleSliderWidget, "Simple");

                        // arbitrary slicer
                        QVBoxLayout* sliceSliderBox = new QVBoxLayout();{
                            QLabel* sliceSliderLabel = new QLabel("Slicer");

                            QGridLayout* sliceSliderSubBox = new QGridLayout();{

                                sliceSliderBox->addWidget(sliceSliderLabel);
                                m_sliceSliders = new QSlider*[N_SLICE_SLIDERS];

                                QVBoxLayout* actualSliders = new QVBoxLayout();{
                                    for(int i=0; i<N_SLICE_SLIDERS; i++){
                                        QHBoxLayout* sliderBox = new QHBoxLayout();{
                                            QLabel* curLabel = new QLabel(g_slice_slider_captions[i]);
                                            sliderBox->addWidget(curLabel);
                                            m_sliceSliders[i] = new QSlider(Qt::Horizontal);
                                            m_sliceSliders[i]->setRange(SLICE_SLIDER_MIN, SLICE_SLIDER_MAX);
                                            sliderBox->addWidget(m_sliceSliders[i]);
                                            connect(m_sliceSliders[i], SIGNAL(valueChanged(int)), this, SLOT(renderSlice()));
                                            m_sliceSliders[i]->setValue(SLICE_SLIDER_INIT);
                                        }
                                        actualSliders->addLayout(sliderBox);
                                    }
                                }
                                QWidget* actualSliderWidget = new QWidget();
                                actualSliderWidget->setLayout(actualSliders);
                                sliceSliderSubBox->addWidget(actualSliderWidget, 0, 0, 1, 12);

                                QVBoxLayout* sliderDisplays = new QVBoxLayout();{
                                    for(int i=0; i<N_SLICE_SLIDERS; i++){
                                            NumberEdit* numberEdit = new NumberEdit();
                                            sliderDisplays->addWidget(numberEdit);
                                            numberEdit->displayInteger(SLICE_SLIDER_INIT);
                                            connect(m_sliceSliders[i], SIGNAL(valueChanged(int)), numberEdit, SLOT(displayInteger(int)));
                                    }
                                }
                                QWidget* displayWidget = new QWidget();
                                displayWidget->setLayout(sliderDisplays);
                                displayWidget->setFixedWidth(50);
                                sliceSliderSubBox->addWidget(displayWidget, 0, 12, 1, 2);
                            }

                            sliceSliderBox->addLayout(sliceSliderSubBox);
                        }

                        QWidget* sliceSliderWidget = new QWidget();
                        sliceSliderWidget->setLayout(sliceSliderBox);
                        m_sliceTab->addTab(sliceSliderWidget, "Pro");


                        QVBoxLayout* freeSliceBox = new QVBoxLayout();{
                            freeSliceBox->addWidget(new QLabel("Please click and drag\nwith the left mouse button\n\n \
                                                               Click and drag the mouse wheel to move the plane "));
                        }
                        QWidget* freeSliceWidget = new QWidget();
                        freeSliceWidget->setLayout(freeSliceBox);
                        m_sliceTab->addTab(freeSliceWidget, "Free");

                        m_sliceTab->setCurrentIndex(1);

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
        QVBoxLayout *rightLayout = new QVBoxLayout; {
            glWidget = new GLWidget();
            glWidget->setFixedSize(RENDER_SIZE, RENDER_SIZE);
            rightLayout->addWidget(glWidget);



            QHBoxLayout *controls0 = new QHBoxLayout; {
                m_phongShading = new QCheckBox("Phong shading");
                connect(m_phongShading, SIGNAL(clicked(bool)), glWidget, SLOT(setPhongShading(bool)) );
                controls0->addWidget(m_phongShading);

                m_invertCrossSection = new QPushButton("Invert cross section");
                connect(m_invertCrossSection, SIGNAL(clicked()), glWidget, SLOT(invertCrossSection()));
                controls0->addWidget(m_invertCrossSection);

                m_invertCrossSection->setEnabled(false);
            }
            rightLayout->addLayout(controls0);

            QHBoxLayout *radioLayout = new QHBoxLayout;

            m_slicingBox = new QGroupBox; {
                QLabel *sliceViewLabel = new QLabel("Slice view");
                radioLayout->addWidget(sliceViewLabel);
                m_slicingButtons = new QRadioButton*[N_SLICE_VISUALIZATIONS];
                for(int i=0; i<N_SLICE_VISUALIZATIONS; i++){
                    m_slicingButtons[i] = new QRadioButton();
                    radioLayout->addWidget(m_slicingButtons[i]);
                    QLabel* radioLabel = new QLabel(g_slice_visualization_captions[i]);
                    radioLayout->addWidget(radioLabel);
                    connect(m_slicingButtons[i], SIGNAL(clicked(bool)), this, SLOT(updateSliceVisualization()));
                }
            }
            m_slicingButtons[SLICE_VIS_NONE]->setChecked(true);

            m_slicingBox->setLayout(radioLayout);
            radioLayout->setAlignment(Qt::AlignLeft);
            rightLayout->addWidget(m_slicingBox);
        }
        rightLayout->setAlignment(Qt::AlignTop);
        mainLayout->addLayout(rightLayout);

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
    const char* new_path = g_texture_paths[idx];
    m_lineEdit->clear();
    m_lineEdit->insert(QString(new_path));

}

void Window::loadButtonClicked()
{
    const char* path = m_lineEdit->text().toAscii().constData();
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
            glWidget->setSliceCanonical(SAGITTAL, dz);
            break;

        case HORIZONTAL:
            dy = ((float)val)/((float)SLICE_EDGELENGTH);
            glWidget->setSliceCanonical(HORIZONTAL, dy);
            break;

        case CORONAL:
            dx = ((float)val)/((float)SLICE_EDGELENGTH);
            glWidget->setSliceCanonical(CORONAL, dx);
            break;

        default:
            cout << "ERROR: Invalid default orientation " << endl;
            assert(false);
        }

    }else if(m_sliceTab->currentIndex() == 1){

        dx = ((float)m_sliceSliders[0]->value())/((float)SLICE_SLIDER_MAX - SLICE_SLIDER_MIN) * 2;
        dy = ((float)m_sliceSliders[1]->value())/((float)SLICE_SLIDER_MAX - SLICE_SLIDER_MIN) * 2;
        dz = ((float)m_sliceSliders[2]->value())/((float)SLICE_SLIDER_MAX - SLICE_SLIDER_MIN) * 2;

        theta = ((float)m_sliceSliders[3]->value())/((float)SLICE_SLIDER_MAX - SLICE_SLIDER_MIN) * 2 * M_PI;
        phi = ((float)m_sliceSliders[4]->value())/((float)SLICE_SLIDER_MAX - SLICE_SLIDER_MIN) * 2 * M_PI;
        psi = ((float)m_sliceSliders[5]->value())/((float)SLICE_SLIDER_MAX - SLICE_SLIDER_MIN) * 2 * M_PI;

        cout << "dx: " << dx << ", dy: " << dy << ", dz: " << dz << ", theta: " << theta << ", phi: " << phi << ", psi: " << psi << endl;

        orientation = FREE_FORM;

    }else if(m_sliceTab->currentIndex() == 2){

        dx = m_point.x;
        dy = m_point.y;
        dz = m_point.z;

        // idea: Just take the dot products
//        theta = acos(m_normal.x);
//        phi = acos(m_normal.y);
//        psi = acos(m_normal.z);

        cout << "dx: " << dx << ", dy: " << dy << ", dz: " << dz << ", theta: " << theta << ", phi: " << phi << ", psi: " << psi << endl;

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


void Window::updateSlicePlane(Vector4 cutPoint, Vector4 cutNormal)
{
//    cout << "Window::updateSlicePlane: " << cutPoint << ", " << cutNormal << endl;

//    m_point = cutPoint;
//    m_normal = cutNormal;

//    m_sliceTab->setCurrentIndex(2);
//    renderSlice(0);
}



void Window::updateSliceVisualization()
{
    for(int i=0; i<N_SLICE_VISUALIZATIONS; i++){
        if (m_slicingButtons[i]->isChecked()) {
            if (i == SLICE_VIS_CROSS_SECTION) {
                m_invertCrossSection->setEnabled(true);
            } else {
                m_invertCrossSection->setEnabled(false);
            }

            glWidget->setSliceVisualization((sliceVisualization) i);
        }
    }
}





