#version 450

struct LightSource
{
	vec4 position;
	vec4 color;
	float intensity;
};

layout(push_constant) uniform VertexPushConstants {
    int isAlpha;
	int isNormalMap;
} vertexPushConst;

layout( location = 0 ) in vec2 v2fTexCoord;
layout( location = 1) in vec3 v2fNormal;
layout( location = 2) in vec3 v2fFragCoord;
layout( location = 3) in vec3 v2fCameraPos;
layout( location = 0 ) out vec4 oColor; 


layout( set = 2, binding = 0 ) uniform sampler2D uTexColor;
layout( set = 2, binding = 1 ) uniform sampler2D uRoughtness;
layout( set = 2, binding = 2 ) uniform sampler2D uMetalness;



layout(set = 1, binding = 0) uniform Material {
    vec4 baseColor;
    vec4 emissiveColor;
    vec2 roughAndMentalness;
} material;

layout(set = 3, binding = 0) uniform LightData {
    LightSource light;
} lightData;




void main()
{
	
vec3 lightPos = vec3((lightData.light.position).xyz);
	vec3 cameraPos = v2fCameraPos;
	vec3 fragPos = v2fFragCoord;
	vec3 lightColor = lightData.light.color.rgb * 1.0;

	vec3 baseColor = texture(uTexColor,v2fTexCoord).rgb * material.baseColor.rgb;
	vec3 emissiveColor = material.emissiveColor.rgb;

	float alpha = 1.0;
	float roughness = texture(uRoughtness,v2fTexCoord).r *material.roughAndMentalness.x; //Shininess
	float shininess = 2.0 / (pow(roughness,4) + 0.001) - 2;
	float metalness = texture(uMetalness,v2fTexCoord).r *material.roughAndMentalness.y;
	

	//Direction settings
	vec3 N = normalize(v2fNormal);
	vec3 V = normalize(cameraPos - fragPos);
    vec3 L = normalize(lightPos - fragPos);
	vec3 H = normalize(L + V);


	float pi = 3.1415926;
	float NdotL = max(dot(N,L), 0.0);
	float NdotH = max(dot(N,H),0.0);
	float NdotV = max(dot(N,V),0.0);
	float VdotH = dot(V,H);


	//Specular
	vec3 F0 = (1.0 - metalness) * vec3(0.04,0.04,0.04) + metalness*baseColor;
	vec3 Fv = F0 + (1.0 - F0) * pow( (1.0 - dot(H,V)) ,5);

	//Diffuse
	vec3 pDiffuse = baseColor/pi * (vec3(1.0) - Fv) * (1.0 - metalness);

	//Distribution function D
	float Dh = ((shininess + 2.0) / (2.0 * pi)) * pow(NdotH,shininess);

	//Cook-Torrance model
	float G1 = 2.0 * ( (NdotH * NdotV) / VdotH);
	float G2 = 2.0 * ( (NdotH * NdotL) / VdotH);
	float G = min(1.0, min(G1,G2));

	//Ambient
	vec3 pAmbient = (lightColor).rgb * baseColor * 0.02;

	//Specular
	vec3 specular = ( (Dh * Fv * G) / (4.0 * NdotV * NdotL) );
	
	//BRDF
	vec3 BRDF = (pDiffuse + specular);
	BRDF = max(BRDF * lightColor.rgb * NdotL,0) * 1.2;

	vec3 L0 = material.emissiveColor.rgb+ pAmbient + BRDF * lightColor * NdotL;

	//vec3 pColor = (pAmbient + BRDF ) * alpha;
	oColor = vec4(L0, alpha);
}
