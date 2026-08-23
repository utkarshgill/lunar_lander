typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
void E_2048_4_2(float* restrict data0_16384, float* restrict data1_16384) {
  for (int Lidx0 = 0; Lidx0 < 2048; Lidx0++) {
    int alu0 = (Lidx0<<3);
    float4 val0 = (*((float4*)((data0_16384+alu0))));
    float4 val1 = (*((float4*)((data1_16384+alu0))));
    *((float4*)((data0_16384+alu0))) = (float4){((0.9990000128746033f*val0[0])+(0.0010000000474974513f*val1[0]*val1[0])),((0.9990000128746033f*val0[1])+(0.0010000000474974513f*val1[1]*val1[1])),((0.9990000128746033f*val0[2])+(0.0010000000474974513f*val1[2]*val1[2])),((0.9990000128746033f*val0[3])+(0.0010000000474974513f*val1[3]*val1[3]))};
    int alu2 = (alu0+4);
    float4 val2 = (*((float4*)((data0_16384+alu2))));
    float4 val3 = (*((float4*)((data1_16384+alu2))));
    *((float4*)((data0_16384+alu2))) = (float4){((0.9990000128746033f*val2[0])+(0.0010000000474974513f*val3[0]*val3[0])),((0.9990000128746033f*val2[1])+(0.0010000000474974513f*val3[1]*val3[1])),((0.9990000128746033f*val2[2])+(0.0010000000474974513f*val3[2]*val3[2])),((0.9990000128746033f*val2[3])+(0.0010000000474974513f*val3[3]*val3[3]))};
  }
}
