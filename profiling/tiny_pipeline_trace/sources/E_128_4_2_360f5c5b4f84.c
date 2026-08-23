typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
void E_128_4_2(float* restrict data0_1024, float* restrict data1_1024) {
  for (int Lidx0 = 0; Lidx0 < 128; Lidx0++) {
    int alu0 = (Lidx0<<3);
    float4 val0 = (*((float4*)((data0_1024+alu0))));
    float4 val1 = (*((float4*)((data1_1024+alu0))));
    *((float4*)((data0_1024+alu0))) = (float4){((0.8999999761581421f*val0[0])+(0.10000000149011612f*val1[0])),((0.8999999761581421f*val0[1])+(0.10000000149011612f*val1[1])),((0.8999999761581421f*val0[2])+(0.10000000149011612f*val1[2])),((0.8999999761581421f*val0[3])+(0.10000000149011612f*val1[3]))};
    int alu2 = (alu0+4);
    float4 val2 = (*((float4*)((data0_1024+alu2))));
    float4 val3 = (*((float4*)((data1_1024+alu2))));
    *((float4*)((data0_1024+alu2))) = (float4){((0.8999999761581421f*val2[0])+(0.10000000149011612f*val3[0])),((0.8999999761581421f*val2[1])+(0.10000000149011612f*val3[1])),((0.8999999761581421f*val2[2])+(0.10000000149011612f*val3[2])),((0.8999999761581421f*val2[3])+(0.10000000149011612f*val3[3]))};
  }
}
