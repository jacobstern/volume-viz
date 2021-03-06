/*!
   @file   CS123Matrix.cpp
   @author Travis Fischer (fisch0920@gmail.com)
   @date   Fall 2008
   
   @brief
      Provides basic functionality for a templated, arbitrarily-sized matrix.
      You will need to fill this file in for the Camtrans assignment.

**/

#include "CS123Algebra.h"
#include <iostream>

//@name Routines which generate specific-purpose transformation matrices
//@{---------------------------------------------------------------------
// @returns the scale matrix described by the vector
Matrix4x4 getScaleMat(const Vector4 &v) {
    // [PASS]
    return Matrix4x4(v.x, 0, 0, 0,
                     0, v.y, 0, 0,
                     0, 0, v.z, 0,
                     0, 0, 0, 1);
}

// @returns the translation matrix described by the vector
Matrix4x4 getTransMat(const Vector4 &v) {
    // [PASS]
    return Matrix4x4(1, 0, 0, v.x,
                     0, 1, 0, v.y,
                     0, 0, 1, v.z,
                     0, 0, 0, 1);
}

// @returns the rotation matrix about the x axis by the specified angle
Matrix4x4 getRotXMat (const REAL radians) {
    // [PASS]
    return Matrix4x4(1, 0, 0, 0,
                     0, cos(radians), -sin(radians), 0,
                     0, sin(radians), cos(radians), 0,
                     0, 0, 0, 1);
}

// @returns the rotation matrix about the y axis by the specified angle
Matrix4x4 getRotYMat (const REAL radians) {
    // [PASS]
    return Matrix4x4(cos(radians), 0, sin(radians), 0,
                     0, 1, 0, 0,
                     -sin(radians), 0, cos(radians), 0,
                     0, 0, 0, 1);
}

// @returns the rotation matrix about the z axis by the specified angle
Matrix4x4 getRotZMat (const REAL radians) {
    // [PASS]
    // NOTE: Make sure the negative signs are correct!
    return Matrix4x4(cos(radians), -sin(radians), 0, 0,
                     sin(radians), cos(radians), 0, 0,
                     0, 0, 1, 0,
                     0, 0, 0, 1);

}

// @returns the rotation matrix around the vector and point by the specified angle
Matrix4x4 getRotMat  (const Vector4 &p, const Vector4 &v, const REAL a) {
    // [PASS]
    REAL theta = atan2(v.z, v.x);
    REAL phi = -atan2(v.y, sqrt(v.x*v.x + v.z*v.z));

    // translate to the origin and back
    Matrix4x4 Mt = getTransMat(-p);
    Matrix4x4 Mt_1 = getInvTransMat(-p);

    Matrix4x4 M1 = getRotYMat(theta);
    Matrix4x4 M2 = getRotZMat(phi);
    Matrix4x4 M3 = getRotXMat(a);
    Matrix4x4 M1_1 = getInvRotYMat(theta);
    Matrix4x4 M2_1 = getInvRotZMat(phi);

    return Mt_1*M1_1*M2_1*M3*M2*M1*Mt;
}


// @returns the inverse scale matrix described by the vector
Matrix4x4 getInvScaleMat(const Vector4 &v) {
    // [PASS]
    return Matrix4x4(1/v.x, 0, 0, 0,
                     0, 1/v.y, 0, 0,
                     0, 0, 1/v.z, 0,
                     0, 0, 0, 1);
}

// @returns the inverse translation matrix described by the vector
Matrix4x4 getInvTransMat(const Vector4 &v) {
    // [PASS]
    return Matrix4x4(1, 0, 0, -v.x,
                     0, 1, 0, -v.y,
                     0, 0, 1, -v.z,
                     0, 0, 0, 1);

}

// @returns the inverse rotation matrix about the x axis by the specified angle
Matrix4x4 getInvRotXMat (const REAL radians) {
    // [PASS]
    return Matrix4x4(1, 0, 0, 0,
                     0, cos(radians), sin(radians), 0,
                     0, -sin(radians), cos(radians), 0,
                     0, 0, 0, 1);
}

// @returns the inverse rotation matrix about the y axis by the specified angle
Matrix4x4 getInvRotYMat (const REAL radians) {
    // [PASS]
    return Matrix4x4(cos(radians), 0, -sin(radians), 0,
                     0, 1, 0, 0,
                     sin(radians), 0, cos(radians), 0,
                     0, 0, 0, 1);

}

// @returns the inverse rotation matrix about the z axis by the specified angle
Matrix4x4 getInvRotZMat (const REAL radians) {
    // [PASS]
    return Matrix4x4(cos(radians), sin(radians), 0, 0,
                     -sin(radians), cos(radians), 0, 0,
                     0, 0, 1, 0,
                     0, 0, 0, 1);

}

// @returns the inverse rotation matrix around the vector and point by the specified angle
Matrix4x4 getInvRotMat  (const Vector4 &p, const Vector4 &v, const REAL a) {
    // [PASS]
    return getRotMat(p, v, -a);
}


//@}---------------------------------------------------------------------

