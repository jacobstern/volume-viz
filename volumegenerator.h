#ifndef VOLUMEGENERATOR_H
#define VOLUMEGENERATOR_H

#include <string>

#include "CS123Common.h"

using std::string;

struct Vector3
{
    Vector3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    float x;
    float y;
    float z;
};

typedef Vector3 Point3;

typedef unsigned char byte;

class VolumeGenerator
{
public:
    VolumeGenerator(int x, int y, int z);
    ~VolumeGenerator();

    // idea: draw ellipsoids at multiple locations
    void drawEllipsoid(const Point3& center, const Vector3& axes, const byte& color); // TODO: Implement full color later, grayscale for now

    // shortcut to draw simple brain-like structure
    void drawDefaultBrain();

    // compile to csv
    string volume2csv();

    // shortcut to save it straight to file
    void saveas_csv(char *path);

    // save raw bytes with 3 uint64 in the header
    void saveas_raw(char *dest, bool header=false);
    void loadfrom_raw(char *source, bool header=false);

    // pointer to raw data, refarg indicates size
    byte* getBytes(size_t& size);

    size_t getVolSize();

    Vector3 getDims();

private:
    byte*  m_volume; // consider color too!
    int     m_x;
    int     m_y;
    int     m_z;
};

#endif // VOLUMEGENERATOR_H
