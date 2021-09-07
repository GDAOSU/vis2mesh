#version 330
in vec3 vertexColorGSOutput;
in float depth;
flat in uint primitive_id_GS;

layout(location = 0) out vec4 FragColor0;
layout(location = 1) out float FragColor1;
layout(location = 2) out uint FragColor2;
void main()
{
    FragColor0 = vec4(vertexColorGSOutput, 1);
    FragColor1 = depth;
    FragColor2 = primitive_id_GS;
}