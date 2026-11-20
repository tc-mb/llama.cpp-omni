void main() {
    const uint i = gl_GlobalInvocationID.z * 262144 + gl_GlobalInvocationID.y * 512 + gl_GlobalInvocationID.x;

    if (i >= p.N) {
        return;
    }

    const uint row = i / p.width;
    const uint col = i - row * p.width;

    const uint plane = row / p.rows;
    const uint i3 = plane / p.planes;
    const uint i2 = plane % p.planes;
    const uint i1 = row % p.rows;

    const uint ai = p.a_offset + col*p.a_stride[0] + i1*p.a_stride[1] + i2*p.a_stride[2] + i3*p.a_stride[3];
    const uint di = p.d_offset + col*p.d_stride[0] + i1*p.d_stride[1] + i2*p.d_stride[2] + i3*p.d_stride[3];

    if (p.mode == 2) {
        const uint bi = p.b_offset + col*p.b_stride[0] + i1*p.b_stride[1] + i2*p.b_stride[2] + i3*p.b_stride[3];
        data_d[di] = D_TYPE(op(float(data_a[ai]), float(data_b[bi])));
    } else {
        const float a = float(data_a[ai]);
        const float b = float(data_a[ai + p.width*p.a_stride[0]]);
        data_d[di] = D_TYPE(p.mode == 0 ? op(a, b) : op(b, a));
    }
}
