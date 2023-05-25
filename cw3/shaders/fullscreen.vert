#version 460

layout(location = 0) in vec2 inPosition;

layout(location = 1) in vec2 inTexCoord;

layout( location = 0 ) out vec2 fragTexCoord;

void main() {

    fragTexCoord = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);

    vec2 vertexPos = fragTexCoord * 2.0 - 1.0;

    gl_Position = vec4(vertexPos, 0.0, 1.0);
}