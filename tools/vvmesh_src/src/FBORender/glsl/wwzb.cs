#version 450

layout (local_size_x = 1, local_size_y = 1) in;

/// Must use 'binding', otherwise, not works
layout (rgba8, binding = 0) readonly uniform image2D inputColor;
layout (r32f, binding = 1) readonly uniform image2D inputDepth;
layout (r32ui, binding = 2) readonly uniform uimage2D inputID;

layout (rgba8, binding = 3) writeonly uniform image2D outputColor;
layout (r32f, binding = 4) writeonly uniform image2D outputDepth;
layout (r32ui, binding = 5) writeonly uniform uimage2D outputID;

layout(location=6) uniform int kernel_size;
layout(location=7) uniform float focal;
layout(location=8) uniform float angle_thres;

layout(location=9) uniform uint invalid_index;
layout(location=10) uniform float invalid_depth;

void main(void)
{
    // Image Dimension
    int width = int(gl_NumWorkGroups.x);
    int height = int(gl_NumWorkGroups.y);
    // Coordinate from batch
    ivec2 pixel_coords = ivec2(gl_GlobalInvocationID.xy);
    // Input and output value of center pixel
    vec4 rgb = imageLoad(inputColor, pixel_coords);
    vec4 depth = imageLoad(inputDepth, pixel_coords);
    uvec4 id = imageLoad(inputID, pixel_coords);


    // Angle based filtering
    int halfkernel = int(kernel_size - 1) / 2;
    if (halfkernel>0 && id[0] != invalid_index)
    {
        // windows
        int sx = clamp(int(pixel_coords.x) - halfkernel, 0, width - 1);
        int ex = clamp(int(pixel_coords.x) + halfkernel, 0, width - 1);
        int sy = clamp(int(pixel_coords.y) - halfkernel, 0, height - 1);
        int ey = clamp(int(pixel_coords.y) + halfkernel, 0, height - 1);

        bool done = false;
        for (int rii = sy; rii <= ey; ++rii) {
            for (int cii = sx; cii <= ex; ++cii) {
                if (cii == pixel_coords.x && rii == pixel_coords.y) continue;// skip central pixel
                if (imageLoad(inputID, ivec2(cii, rii))[0] == invalid_index) continue;// skip invalid pixel

                vec4 depthP = imageLoad(inputDepth, ivec2(cii, rii));
                float len = sqrt(pow(cii - pixel_coords.x, 2) + pow(rii - pixel_coords.y, 2)) * depth[0] / focal;
                float deltaD = (depth[0] - depthP[0]);
                double theta = atan(deltaD, len);

                if (theta > angle_thres) {
                    depth[0] = invalid_depth;
                    id[0] = invalid_index;
                    rgb.xyz = vec3(0, 0, 0);
                    done = true;
                    break;
                }
            }
            if (done) break;
        }
    }

    // Write out
    imageStore(outputColor, pixel_coords, rgb);
    imageStore(outputDepth, pixel_coords, depth);
    imageStore(outputID, pixel_coords, id);
}