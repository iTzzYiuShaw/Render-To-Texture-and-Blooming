#version 450

layout(location = 0) out vec4 outColor;

layout(location = 0) in vec2 fragTexCoord;

layout(set = 0 ,binding = 0) uniform sampler2D filterTexture;
layout(set = 1 ,binding = 0) uniform sampler2D PBRTexture;


void main() {


    vec3 gloom = texture(filterTexture,fragTexCoord).rgb * 5;
    vec3 PBR = texture(PBRTexture,fragTexCoord).rgb;

    outColor = vec4((PBR+gloom).rgb,1.0);
    //outColor = cin;
}