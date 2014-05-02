#include "numberedit.h"
#include <math.h>

NumberEdit::NumberEdit(QWidget *parent) :
    QLineEdit(parent)
{
}

void NumberEdit::displayInteger(int num)
{
    QString toShow = QString::number(num);
    setText(toShow);
}

void NumberEdit::displayFloat(float num)
{
    QString toShow = QString::number(num);
    setText(toShow);
}

void NumberEdit::displayRadiansAsDegrees(float radians)
{
    float num = radians * 360 / (2*M_PI);
    QString toShow = QString::number(num);
    setText(toShow);
}
