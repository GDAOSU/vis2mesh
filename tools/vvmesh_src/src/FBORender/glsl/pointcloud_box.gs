#version 330 core
layout(points) in ;
layout(triangle_strip, max_vertices = 4) out ;

in vec3 vertexColorVSOutput[];
in uint primitive_id_VS[];
uniform mat4 VCDCMatrix;

out vec3 vertexColorGSOutput;
out float depth;
flat out uint primitive_id_GS;

void main() {
    gl_Position = VCDCMatrix * (gl_in[0].gl_Position + vec4(-gl_in[0].gl_PointSize,-gl_in[0].gl_PointSize,0,0));
    vertexColorGSOutput = vertexColorVSOutput[0];
    depth = gl_in[0].gl_Position.z;
    primitive_id_GS = primitive_id_VS[0];
    EmitVertex();

    gl_Position = VCDCMatrix * (gl_in[0].gl_Position + vec4(gl_in[0].gl_PointSize,-gl_in[0].gl_PointSize,0,0));
    vertexColorGSOutput = vertexColorVSOutput[0];
    depth = gl_in[0].gl_Position.z;
    primitive_id_GS = primitive_id_VS[0];
    EmitVertex();

    gl_Position = VCDCMatrix * (gl_in[0].gl_Position + vec4(-gl_in[0].gl_PointSize,gl_in[0].gl_PointSize,0,0));
    vertexColorGSOutput = vertexColorVSOutput[0];
    depth = gl_in[0].gl_Position.z;
    primitive_id_GS = primitive_id_VS[0];
    EmitVertex();

    gl_Position = VCDCMatrix * (gl_in[0].gl_Position + vec4(gl_in[0].gl_PointSize,gl_in[0].gl_PointSize,0,0));
    vertexColorGSOutput = vertexColorVSOutput[0];
    depth = gl_in[0].gl_Position.z;
    primitive_id_GS = primitive_id_VS[0];
    EmitVertex();
    EndPrimitive();
}