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
    lazyComputeTransform();

    glMatrixMode(GL_MODELVIEW);

    glLoadMatrixd( transform.constData() );

//    float debug[16];
//    glGetFloatv(GL_MODELVIEW_MATRIX, debug);

//    for (int i = 0; i < 16; i++) {
//        qDebug() << i << " " << debug[i];
//    }
}

QMatrix4x4 Camera::rotation()
{
    QVector3D side = QVector3D::crossProduct(look,   up).normalized(),
              up   = QVector3D::crossProduct(side, look).normalized();

    return QMatrix4x4( side.x(),   side.y(),   side.z(),   0.f,
                       up.x(),     up.y(),     up.z(),     0.f,
                       -look.x(),  -look.y(),  -look.z(),  0.f,
                       0.f,        0.f,        0.f,        1.f );
}

QMatrix4x4 Camera::transformation()
{
    lazyComputeTransform();

    return transform;
}

void Camera::lazyComputeTransform()
{
    QVector3D side = QVector3D::crossProduct(look, up).normalized(),
              up = QVector3D::crossProduct(side, look).normalized();

    transform = QMatrix4x4( side.x(),   side.y(),   side.z(),   0.f,
                            up.x(),     up.y(),     up.z(),     0.f,
                            -look.x(),  -look.y(),  -look.z(),  0.f,
                            0.f,        0.f,        0.f,        1.f ) *
                QMatrix4x4( 1.0f,       0.0f,       0.0f,       -position.x(),
                            0.0f,       1.0f,       0.0f,       -position.y(),
                            0.0f,       0.0f,       1.0f,       -position.z(),
                            0.0f,       0.0f,       0.0f,       1.0f );
}
