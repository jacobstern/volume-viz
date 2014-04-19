#ifndef CAMERA_H
#define CAMERA_H

#include <QObject>
#include <QtOpenGL>

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

private:
    QVector3D position;
    QVector3D look;
    QVector3D up;

    float fov;

};

#endif // CAMERA_H
