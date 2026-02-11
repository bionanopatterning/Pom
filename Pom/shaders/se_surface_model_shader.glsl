#vertex
#version 420

layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;

uniform mat4 vpMat;

out vec3 fnormal;

void main()
{
    gl_Position = vpMat * vec4(pos, 1.0);
    fnormal = normal;
}

#fragment
#version 420

in vec3 fnormal;
out vec4 fragColour;

uniform vec4 color;
uniform vec3 lightDir;
uniform vec3 viewDir;
uniform int style;
uniform float ambientStrength;
uniform float lightStrength;
uniform vec3 lightColour;

void main()
{
    if (style == 0) // Old Phong
    {
        float F_AMBIENT = ambientStrength;
        float F_DIFFUSE = lightStrength;
        float F_SPECULAR = 0.2f * lightStrength;
        float F_EMISSIVE = ambientStrength * 0.3f;

        float SPEC_POWER = 0.0f;

        vec3 ambient = F_AMBIENT * color.rgb;

        vec3 diffuse = max(0.0, dot(normalize(fnormal), lightDir)) * F_DIFFUSE * lightColour * color.rgb;

        vec3 localViewDir = normalize(gl_FragCoord.xyz);
        vec3 reflDir = reflect(-lightDir, fnormal);
        float specIntensity = pow(min(1.0, max(dot(localViewDir, reflDir), 0.0)), SPEC_POWER);
        vec3 specular = F_SPECULAR * specIntensity * vec3(1.0, 1.0, 1.0);

        vec3 emissive = dot(normalize(fnormal), localViewDir) * F_EMISSIVE * color.rgb;
        fragColour = vec4(ambient + diffuse + specular + emissive, color.a);
    }
    else if (style == 1) // Phong - Enhanced
    {
        // Lighting intensities
        float F_AMBIENT = ambientStrength * 0.7;
        float F_DIFFUSE = lightStrength;
        float F_SPECULAR = 0.35 * lightStrength;
        float F_RIM = 0.4;

        // Specular sharpness (higher = tighter highlights)
        float SPEC_POWER = 48.0;

        vec3 N = normalize(fnormal);
        vec3 L = normalize(lightDir);
        vec3 V = normalize(viewDir);
        vec3 R = reflect(-L, N);

        // === AMBIENT with fake AO ===
        // Darken downward-facing surfaces slightly for depth
        float ao = mix(0.6, 1.0, N.y * 0.5 + 0.5);
        vec3 ambient = F_AMBIENT * color.rgb * ao;

        // === DIFFUSE with wrap-around ===
        // Wrap lighting prevents harsh black shadows
        float NdotL = dot(N, L);
        float diffuseWrap = max(0.0, (NdotL + 0.3) / 1.3);  // Wrap around
        vec3 diffuse = diffuseWrap * F_DIFFUSE * lightColour * color.rgb;

        // === SPECULAR ===
        // Crisp highlights with subtle warm tint
        float specIntensity = pow(max(dot(V, R), 0.0), SPEC_POWER);
        vec3 specColor = mix(vec3(1.0), lightColour * 1.2, 0.3);  // Subtle light color tint
        vec3 specular = F_SPECULAR * specIntensity * specColor;

        // === RIM LIGHT (Fresnel) ===
        // Strong edge highlighting for clarity and style
        float rimDot = 1.0 - max(dot(N, V), 0.0);
        float rimIntensity = pow(rimDot, 3.0);  // Cubic falloff for sharper rims
        vec3 rimColor = mix(color.rgb, vec3(1.0), 0.4);  // Brighten rim slightly
        vec3 rim = F_RIM * rimIntensity * rimColor;

        // === COLOR BOOST ===
        // Slightly increase saturation in lit areas for more pop
        vec3 finalColor = ambient + diffuse + specular + rim;
        float luminance = dot(finalColor, vec3(0.299, 0.587, 0.114));
        vec3 boostedColor = mix(finalColor, color.rgb, -0.1);  // Subtle color boost

        fragColour = vec4(boostedColor, color.a);
    }
    else if (style == 2) // Flat
    {
        fragColour = vec4(color.rgb, color.a);
    }
    else if (style == 3)
    {
        fragColour = vec4(fnormal * 0.5f + 0.5f, 1.0f);
    }
}