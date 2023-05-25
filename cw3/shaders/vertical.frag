#version 450

layout(location = 0) out vec4 outColor;

layout(location = 0) in vec2 fragTexCoord;

layout(set = 0 ,binding = 0) uniform sampler2D inputTexture;

layout(set = 1, binding = 0) uniform vGaussan {
	float gaussianWeights[22];
}Gaussan;

void main()
{
	vec3 pixelColor = texture(inputTexture, fragTexCoord).rgb;

	//oColor = vec4(pixelColor, 1.0f);

	// ==============================================
	//float weight[5] = float[] (0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);
	float weight[22] = Gaussan.gaussianWeights;

	vec2 offset = 1.0 / textureSize(inputTexture, 0);

	vec3 result = texture(inputTexture, fragTexCoord).rgb * weight[0]; // Current fragment

	for(int i = 1; i < 22; i++)
	{
		// Vertical
		result += texture(inputTexture, fragTexCoord + vec2(0.0, offset.y * i)).rgb * weight[i];
		result += texture(inputTexture, fragTexCoord - vec2(0.0, offset.y * i)).rgb * weight[i];
	}

	outColor = vec4(result, 1.0f);
}