varying vec3 cubeSpace;
uniform vec3 scale;

void main(void)
{
    cubeSpace = gl_Vertex.xyz * scale / 2.0 + 0.5;
    gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
}
