#version 330 core
layout(points) in ;
layout(points, max_vertices = 1) out ;

in vec3 vertexColorVSOutput[];
in uint primitive_id_VS[];
uniform mat4 VCDCMatrix;

out vec3 vertexColorGSOutput;
out float depth;
flat out uint primitive_id_GS;

void main() {
    gl_Position = VCDCMatrix * gl_in[0].gl_Position;
    vertexColorGSOutput = vertexColorVSOutput[0];
    depth = gl_in[0].gl_Position.z;
    primitive_id_GS = primitive_id_VS[0];
    EmitVertex();
    EndPrimitive();
}