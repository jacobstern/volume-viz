#include "volumegenerator.h"

#include<sstream>
#include<string>
#include<iostream>
#include<fstream>

using namespace std;

VolumeGenerator::VolumeGenerator(int x, int y, int z)
{
    m_volume = new byte[x*y*z];
    m_x = x;
    m_y = y;
    m_z = z;

    int lim = x*y*z;
    for(int i=0; i<lim; i++){
        m_volume[i] = '\0';
    }
}

VolumeGenerator::~VolumeGenerator()
{
    delete[] m_volume;
    m_volume = 0;
}

void VolumeGenerator::drawEllipsoid(const Point3& center, const Vector3& axes, const byte& color) // TODO: Implement full color later, grayscale for now
{
    // encoded in z-y-x-order
    for(int k=0; k<m_z; k++){
        for(int j=0; j<m_y; j++){
            for(int i=0; i<m_x; i++){
                int offset = k*m_z*m_y + j*m_y + i;
                const Point3& c = center;
                const Vector3& a = axes;
                m_volume[offset] = (((((c.x-i)/a.x) * ((c.x-i)/a.x))
                                     + (((c.y-j)/a.y) * ((c.y-j)/a.y))
                                    + (((c.z-k)/a.z) * ((c.z-k)/a.z))) < 1.0f) ?
                            color :
                            m_volume[offset];
            }
        }
    }
}

// shortcut to draw simple brain-like structure
void VolumeGenerator::drawDefaultBrain()
{
    Point3 centers[2] = {Point3(50.0f, 28.0f, 50.0f), Point3(50.0f, 72.0f, 50.0f)};

    Vector3 layers[4] = {Vector3(45.0f, 25.0f, 30.0f),
                        Vector3(40.0f, 20.0f, 50.0f),
                        Vector3(30.0f, 10.0f, 13.0f),
                        Vector3(20.0f, 3.0f, 8.0f)};

    char shades[4] = {(byte)0.5, (byte)0.6, (byte)0.8, (byte)1.0};

    for(int center_idx=0; center_idx<2; center_idx++){
        for(int layer_idx=0; layer_idx<4; layer_idx++){
            this->drawEllipsoid(centers[center_idx], layers[layer_idx], shades[layer_idx]);
        }
    }
}

// compile to csv
string VolumeGenerator::volume2csv()
{
    ostringstream os;
    for(int k=0; k<m_z; k++){
        for(int j=0; j<m_y; j++){
            for(int i=0; i<m_x; i++){
                int offset = k*m_z*m_y + j*m_y + i;
                os << m_volume[offset] << ",";
            }
            os << "\t";
        }
        os << "\n";
    }
    return os.str();
}

// shortcut to save it straight to file
void VolumeGenerator::saveas_csv(char *path)
{
    ofstream dest;
    dest.open(path, ios::out | ios::trunc);
    dest << this->volume2csv();
    dest.close();
}

byte* VolumeGenerator::getBytes(size_t& size)
{
    size = getVolSize();
    return (byte*)m_volume;
}

size_t VolumeGenerator::getVolSize()
{
    return m_x*m_y*m_z*sizeof(byte);
}















