#include "camera.h"

Camera::Camera(QObject *parent) :
    QObject(parent)
{
    position = QVector3D(0.f, 0.f, 0.f);
    look = QVector3D(0.f, 0.f, 1.f);
    up = QVector3D(0.f, 1.f, 0.f);
}

QVector3D Camera::getPosition()
{
    return position;
}

QVector3D Camera::getLook()
{
    return look;
}

QVector3D Camera::getUp()
{
    return up;
}

void Camera::lookAt(QVector3D target)
{
    look = (target - position).normalized();
}

void Camera::setFov(float angle)
{
    fov = angle;
}

void Camera::setPosition(QVector3D vector)
{
    position = vector;
}

void Camera::setLook(QVector3D vector)
{
    look = vector.normalized();
}

void Camera::setUp(QVector3D vector)
{
    up = vector.normalized();
}

void Camera::updateView()
{
    QVector3D side = QVector3D::crossProduct(look, up).normalized(),
              up = QVector3D::crossProduct(side, look).normalized();

    GLfloat m[4][4];

    m[0][0] = side.x();
    m[1][0] = side.y();
    m[2][0] = side.z();

    m[0][1] = up.x();
    m[1][1] = up.y();
    m[2][1] = up.z();

    m[0][2] = -look.x();
    m[1][2] = -look.y();
    m[2][2] = -look.z();

    m[3][3] = 1.0f;

    glMatrixMode(GL_MODELVIEW);

    glLoadMatrixf(&m[0][0]);
    glTranslatef(-position.x(), -position.y(), -position.z());
}
