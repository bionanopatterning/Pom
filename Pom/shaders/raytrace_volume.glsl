#compute
#version 430

layout(binding = 0) uniform sampler3D volume;
layout(binding = 1) uniform sampler2D depth_start;
layout(binding = 2) uniform sampler2D depth_stop;
layout(binding = 3, rgba32f) uniform image2D target;
layout(local_size_x = 16, local_size_y=16) in;

uniform mat4 ipMat;
uniform mat4 ivMat;
uniform mat4 pMat;
uniform vec2 viewportSize;
uniform float pixelSize;
uniform int Z;
uniform int Y;
uniform int X;
uniform vec3 C;

uniform float near;
uniform float far;

float linearizeDepth(float depth)
{
    float z = depth; // Back to NDC
    return (2.0 * near * far) / (far + near - z * (far - near));
}

void main()
{
    // Get Ray start position.
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = vec2(px) / viewportSize;

    // Normalized device coordinates to world coordinates. First, find ndc's:
    float z_start = 2.0f * texture(depth_start, uv).r - 1.0f;
    float z_stop = 2.0f * texture(depth_stop, uv).r - 1.0f;
    float x = (2.0f * float(px.x)) / viewportSize.x - 1.0f;
    float y = (2.0f * float(px.y)) / viewportSize.y - 1.0f;

    vec4 cs_start = ipMat * vec4(x, y, z_start, 1.0);  // clip space (cs) start vec
    cs_start /= cs_start.w;
    vec4 cs_stop = ipMat * vec4(x, y, z_stop, 1.0);
    cs_stop /= cs_stop.w;

    vec3 start_pos = (ivMat * cs_start).xyz;
    vec3 stop_pos = (ivMat * cs_stop).xyz;
    vec3 dir = normalize(stop_pos - start_pos);
    vec3 volume_size = vec3(X, Y, Z);

    if (z_start == 1.0f)
    {
        imageStore(target, px, vec4(0.0f, 0.0f, 0.0f, 1.0f));
    }
    else
    {
        vec3 pos = start_pos;
        vec3 uvw;
        vec3 imgSize = vec3(X, Y, Z);
        float rayValue = 0.0f;
        float pathLength = 0.0;
        int MAX_ITER = 1000;
        float i = 0.0f;
        while (i < MAX_ITER)
        {
            i += 1.0f;
            if (length(pos - start_pos) > length(stop_pos - start_pos))
            {
                break;
            }
            pos += dir;
            uvw = pos.xyz / imgSize / pixelSize + 0.5f;
            rayValue += texture(volume, uvw).r;
        }
        // Write to texture.
        vec3 rayColour = C * rayValue / 500.0;
        vec3 pixelColour = imageLoad(target, px).rgb;
        imageStore(target, px, vec4(pixelColour + rayColour, 1.0));

    }
}
