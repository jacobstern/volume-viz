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

#include <iostream>

using namespace std;

#define N_DEFAULT_TEXTURES 5
static char *g_texture_names[N_DEFAULT_TEXTURES] = {"head",
                                                  "engine",
                                                    "foo",
                                                    "bar",
                                                    "baz"};
static char *g_texture_paths[N_DEFAULT_TEXTURES] = {"/home/rmartens/shared/cs224textures/head.t3d",
                                                 "/home/rmartens/shared/cs224textures/engine.t3d",
                                                    "foo/path",
                                                    "bar/path",
                                                    "baz/path"};

//! [0]
Window::Window()
{
    glWidget = new GLWidget;

    m_sliceWidget = new SliceWidget();
//! [0]

//! [1]
    QHBoxLayout *mainLayout = new QHBoxLayout;{
        QVBoxLayout *leftColumn = new QVBoxLayout();{
            QVBoxLayout *controlBox = new QVBoxLayout();{
                QHBoxLayout *loadBox = new QHBoxLayout;
                m_loadButton = new QPushButton("Reload Volume");
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
                controlBox->addLayout(loadBox);
            }
            leftColumn->addWidget(m_sliceWidget);
            leftColumn->addLayout(controlBox);
        }
        mainLayout->addLayout(leftColumn);
    }
    mainLayout->addWidget(glWidget);
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

}














