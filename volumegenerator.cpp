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
        m_volume[i] = (byte)0;
    }
}

VolumeGenerator::~VolumeGenerator()
{
    delete[] m_volume;
    m_volume = 0;
}

void VolumeGenerator::drawEllipsoid(const Point3& center, const Vector3& axes, const byte& color) // TODO: Implement full color later, grayscale for now
{
    float fi;
    float fj;
    float fk;

    // encoded in z-y-x-order
    for(int k=0; k<m_z; k++){
        for(int j=0; j<m_y; j++){
            for(int i=0; i<m_x; i++){
                int offset = k*m_y*m_x + j*m_x + i;
                const Point3& c = center;
                const Vector3& a = axes;
                fi = ((float)i)/((float)m_x);
                fj = ((float)j)/((float)m_y);
                fk = ((float)k)/((float)m_z);

//                cout << "fi: " << fi << ", fj: " << fj << ", fk: " << fk << endl;

                assert(fi >= 0.0);
                assert(fi <= 1.0);
                assert(fj >= 0.0);
                assert(fj <= 1.0);
                assert(fk >= 0.0);
                assert(fk <= 1.0);

                m_volume[offset] = (((((c.x-fi)/a.x) * ((c.x-fi)/a.x))
                                     + (((c.y-fj)/a.y) * ((c.y-fj)/a.y))
                                    + (((c.z-fk)/a.z) * ((c.z-fk)/a.z))) < 1.0) ?
                            color :
//                            (m_volume[offset] == (byte)3 ? (byte)0 : m_volume[offset]);
                            m_volume[offset];




//                            (byte)2;

//                m_volume[offset] = (c.x-fi)*(c.x-fi) + (c.y-fj)*(c.y-fj) + (c.z-fk)*(c.z-fk) <= 0.25 ? byte(2) : byte(3);



//                if(fi >= 0.00 && fi <= 0.35){
//                    m_volume[offset] = (byte)1;
//                }

//                if(fi >= 0.35 && fi <= 0.65){
//                    m_volume[offset] = (byte)2;
//                }

//                if(fi >= 0.65 && fi <= 1.0){
//                    m_volume[offset] = (byte)3;
//                }

                if(fi >= 0.99){
                    m_volume[offset] = (byte)4;
                }

//                m_volume[offset] = (byte)100;
            }
        }
    }



    cout << "max fi,fj,fk: " << fi << ", " << fj << ", " << fk << endl;
}

// shortcut to draw simple brain-like structure
void VolumeGenerator::drawDefaultBrain()
{
    Point3 centers[2] = {Point3(0.25f, 0.50f, 0.50f), Point3(0.75f, 0.50f, 0.50f)};

    Vector3 layers[4] = {Vector3(0.23f, 0.30f, 0.45f),
                        Vector3(0.18f, 0.27f, 0.40f),
                        Vector3(0.10f, 0.23f, 0.30f),
                        Vector3(0.03f, 0.20f, 0.20f)};

//    Vector3 layers[1] = {Vector3(0.15f, 0.25f, 0.47f)};

    byte shades[4] = {(byte)(60), (byte)(80), (byte)(100), (byte)(120)};

    for(int center_idx=0; center_idx<2; center_idx++){
        for(int layer_idx=0; layer_idx<4; layer_idx++){
            printf("%u\n", shades[layer_idx]);
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
                os << (int)m_volume[offset] << ",";
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

void VolumeGenerator::saveas_raw(char *path)
{
    cout << "writing data to output file" << endl;
    ofstream dest;
    dest.open(path, ios::out | ios::trunc | ios::binary);
    size_t size = m_x * m_y * m_z * sizeof(byte);
    const char *ptr = (char*) m_volume;

    cout << "writing sizes" << endl;
    size_t size_x = (size_t)m_x;
    size_t size_y = (size_t)m_y;
    size_t size_z = (size_t)m_z;

//    dest.write(&size_x, sizeof(size_t));




//    dest.write(&size_y, sizeof(size_t));
//    dest.write(&size_z, size)

    cout << "writing binary data" << endl;
    dest.write(ptr, size);
    dest.close();
    cout << "data has been written to output file" << endl;
}

void VolumeGenerator::loadfrom_raw(char *source)
{
//    cout << "reading meta data file" << endl;
    cout << "skipping meta file; hardcode it for now" << endl;

    cout << "reading raw file" << endl;
    ifstream src;
    src.open(source, ios::in | ios::binary);
    size_t size = 512*512*512*sizeof(byte);
    m_x = 512;
    m_y = 512;
    m_z = 512;

    char *ptr = (char*) m_volume;
    src.read(ptr,size);
    src.close();
    cout << "raw file has been read" << endl;


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















