#extension GL_EXT_shader_16bit_storage : require


layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
layout (binding = 1) readonly buffer B {A_TYPE data_b[];};
layout (binding = 2) writeonly buffer D {D_TYPE data_d[];};

layout (push_constant) uniform parameter
{
    uint N;
    uint width;
    uint rows;
    uint planes;
    uint mode;
    float alpha;
    float limit;
    uint a_stride[4];
    uint b_stride[4];
    uint d_stride[4];
    uint a_offset;
    uint b_offset;
    uint d_offset;
} p;
