#version 330 core
#define MAX_VERTEX 12
layout(points) in ;
layout(triangle_strip, max_vertices = MAX_VERTEX) out ;

const float PI = 3.1415926535897932384626433832795;
const float PI_2 = 1.57079632679489661923;
const float PI_4 = 0.785398163397448309616;

in vec3 vertexColorVSOutput[];
in uint primitive_id_VS[];
uniform mat4 VCDCMatrix;

out vec3 vertexColorGSOutput;
out float depth;
flat out uint primitive_id_GS;

void main() {
    gl_Position = VCDCMatrix * (gl_in[0].gl_Position + vec4(-gl_in[0].gl_PointSize,0,0,0));
    vertexColorGSOutput = vertexColorVSOutput[0];
    depth = gl_in[0].gl_Position.z;
    primitive_id_GS = primitive_id_VS[0];
    EmitVertex();

    float nseg = float(MAX_VERTEX - 2) / 2.f+1;
    float step = PI/float(nseg);
    float phi = -PI_2 + step;
    while(phi<= (PI_2 - step))
    {
        float x = gl_in[0].gl_PointSize * sin(phi);
        float y = gl_in[0].gl_PointSize * cos(phi);
        gl_Position = VCDCMatrix * (gl_in[0].gl_Position + vec4(x,y,0,0));
        vertexColorGSOutput = vertexColorVSOutput[0];
        depth = gl_in[0].gl_Position.z;
        primitive_id_GS = primitive_id_VS[0];
        EmitVertex();

        gl_Position = VCDCMatrix * (gl_in[0].gl_Position + vec4(x,-y,0,0));
        vertexColorGSOutput = vertexColorVSOutput[0];
        depth = gl_in[0].gl_Position.z;
        primitive_id_GS = primitive_id_VS[0];
        EmitVertex();

        phi = phi + step;
    }

    gl_Position = VCDCMatrix * (gl_in[0].gl_Position + vec4(gl_in[0].gl_PointSize,0,0,0));
    vertexColorGSOutput = vertexColorVSOutput[0];
    depth = gl_in[0].gl_Position.z;
    primitive_id_GS = primitive_id_VS[0];
    EmitVertex();
    EndPrimitive();
}