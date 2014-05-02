#ifndef NUMBEREDIT_H
#define NUMBEREDIT_H

#include <QLineEdit>

class NumberEdit : public QLineEdit
{
    Q_OBJECT
public:
    explicit NumberEdit(QWidget *parent = 0);
    
signals:

    
public slots:
    void displayInteger(int num);
    
};

#endif // NUMBEREDIT_H
