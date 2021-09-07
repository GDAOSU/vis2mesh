#version 330
in float depth;
in vec2 fragTC;
flat in uint primitive_id_VS;

uniform sampler2D texImage;

layout(location = 0) out vec4 FragColor0;
layout(location = 1) out float FragColor1;
layout(location = 2) out uint FragColor2;

void main(){
    FragColor0 = texture(texImage, fragTC);
    FragColor1 = depth;
    FragColor2 = primitive_id_VS;
}