#version 450

layout(location = 0) out vec4 outColor;

layout(location = 0) in vec2 fragTexCoord;

layout(set = 0 ,binding = 0) uniform sampler2D inputTexture;

void main() {

    vec4 cin = texture(inputTexture, fragTexCoord);
    vec3 crgb = cin.rgb;

    //if(crgb.r >= 1.0)        
        crgb.r = crgb.r / (crgb.r + 1);

    //if(crgb.g >= 1.0) 
        crgb.g = crgb.g / (crgb.g + 1);

    //if(crgb.b >= 1.0) 
        crgb.b = crgb.b / (crgb.b + 1);

    outColor = vec4(crgb,1.0);
    //outColor = cin;
}