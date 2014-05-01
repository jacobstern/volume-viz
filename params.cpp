#include "params.h"

#include <iostream>

std::ostream& operator<<(std::ostream& os, const SliceParameters p)
{
    os << "dx: " << p.dx << ", dy: " << p.dy << ", dz: " << p.dz
       << ", theta: " << p.theta << ", phi: " << p.phi << ", psi: " << p.psi;
    return os;
}

std::ostream& operator<<(std::ostream& os, const BufferParameters p)
{
    os << "height: " << p.height << ", width: " << p.width;
    return os;
}

