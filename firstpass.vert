varying vec3 cubeSpace;

void main(void)
{
    cubeSpace = gl_Vertex.xyz + 0.5;
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
