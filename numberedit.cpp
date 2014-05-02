#include "numberedit.h"

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
