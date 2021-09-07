#version 330
in vec3 vertexMC;
in vec2 vertexTC;
in uint primitive_id;

uniform mat4 MCVCMatrix;
uniform mat4 VCDCMatrix;
uniform mat4 MCDCMatrix;

out vec2 fragTC;
out float depth;
flat out uint primitive_id_VS;

void main ()
{
    vec4 vertexVC = MCVCMatrix * vec4(vertexMC, 1.);
    depth = vertexVC.z;

    gl_Position = MCDCMatrix * vec4(vertexMC, 1.);

    fragTC = vertexTC;
    primitive_id_VS = primitive_id;
}