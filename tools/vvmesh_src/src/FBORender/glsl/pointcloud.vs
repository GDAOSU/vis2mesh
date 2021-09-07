#version 330
in vec3 vertexMC;
in vec3 vertexColorVSInput;
in float radius;
in uint primitive_id;

uniform mat4 MCVCMatrix;
uniform mat4 VCDCMatrix;
uniform mat4 MCDCMatrix;

out vec3 vertexColorVSOutput;
out uint primitive_id_VS;
void main ()
{
    gl_Position = MCVCMatrix * vec4(vertexMC, 1.);
    gl_PointSize = radius;
    vertexColorVSOutput = vertexColorVSInput;
    primitive_id_VS = primitive_id;
}