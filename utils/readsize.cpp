// writes size to binary; used to generate t3d files
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
using namespace std;

int main(int argc, char **argv)
{
	if(argc != 2){
		cout << "Error! Need exact 1 argument: Filepath" << endl;
	}

	size_t size_x;
	size_t size_y;
	size_t size_z;

	ifstream dest;
	dest.open(argv[1], ios::in | ios::binary);
	
	dest.read((char*)&size_x, sizeof(size_t));
	dest.read((char*)&size_y, sizeof(size_t));
	dest.read((char*)&size_z, sizeof(size_t));

	dest.close();

	cout << "size_x: " << size_x << endl;
	cout << "size_y: " << size_y << endl;
	cout << "size_z: " << size_z << endl;

	return 0;
}
