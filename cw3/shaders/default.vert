#version 450

layout( location = 0 ) in vec3 iPosition;
layout( location = 1 ) in vec2 iTexCoord;
layout( location = 2 ) in vec3 iNormal;


layout( set = 0, binding = 0 ) uniform UScene
{
	mat4 camera;
	mat4 projection;
	mat4 projCam;
	vec3 cameraPos;
} uScene;


layout( location = 0 ) out vec2 v2fTexCoord;
layout( location = 1) out vec3 v2fNormal;
layout( location = 2) out vec3 v2fFragCoord;	
layout( location = 3) out vec3 v2fCameraPos;


void main()
{

	v2fTexCoord = iTexCoord;
	v2fNormal = iNormal;
	v2fFragCoord = iPosition;
	v2fCameraPos = uScene.cameraPos;
	gl_Position = uScene.projCam * vec4( iPosition, 1.f ); 
}
