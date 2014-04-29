// writes size to binary; used to generate t3d files
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
using namespace std;

int main(int argc, char **argv)
{
	if(argc != 5){
		cout << "Error! Need exact 4 arguments: Filepath and 3 sizes" << endl;
	}

	size_t size_x;
	size_t size_y;
	size_t size_z;

	size_x = (size_t)atoi(argv[2]);
	size_y = (size_t)atoi(argv[3]);
	size_z = (size_t)atoi(argv[4]);

	ofstream dest;
	dest.open(argv[1], ios::out | ios::trunc | ios::binary);
	
	dest.write((char*)&size_x, sizeof(size_t));
	dest.write((char*)&size_y, sizeof(size_t));
	dest.write((char*)&size_z, sizeof(size_t));

	dest.close();

	return 0;
}
