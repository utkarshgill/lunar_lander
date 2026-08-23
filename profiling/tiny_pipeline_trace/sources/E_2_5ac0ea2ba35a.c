typedef float float2 __attribute__((aligned(8),ext_vector_type(2)));
void E_2(float* restrict data0_2, float* restrict data1_2) {
  float2 val0 = (*((float2*)((data0_2+0))));
  float2 val1 = (*((float2*)((data1_2+0))));
  *((float2*)((data0_2+0))) = (float2){((0.9990000128746033f*val0[0])+(0.0010000000474974513f*val1[0]*val1[0])),((0.9990000128746033f*val0[1])+(0.0010000000474974513f*val1[1]*val1[1]))};
}
