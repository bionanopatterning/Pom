#vertex
#version 430 core

layout(location = 0) in vec2 pos;

out vec2 fuv;

void main()
{
    gl_Position = vec4(pos, 0.0, 1.0);
    fuv = 0.5f + pos / 2.0f;
}

#fragment
#version 430 core

in vec2 fuv;
out vec4 FragColor;

layout(binding = 0) uniform sampler2D imageTexture;

void main()
{
    FragColor = texture(imageTexture, fuv);
}