#ifndef CAMERA_H
#define CAMERA_H

#include <QObject>
//#include <QtOpenGL>
#include <qgl.h>
#include <QVector3D>
#include <QMatrix4x4>

class Camera : public QObject
{
    Q_OBJECT
public:
    explicit Camera(QObject *parent = 0);

    QVector3D getPosition();
    QVector3D getLook();
    QVector3D getUp();

    void lookAt(QVector3D target);
    void setFov(float angle);

public slots:
    void setPosition(QVector3D vector);
    void setLook(QVector3D vector);
    void setUp(QVector3D vector);

    void updateView();
    QMatrix4x4 transformation();
    QMatrix4x4 inverseTransformation();

private:
    QVector3D position;
    QVector3D look;
    QVector3D up;

    float fov;

    QMatrix4x4 transform;
    void lazyComputeTransform();
};

#endif // CAMERA_H
