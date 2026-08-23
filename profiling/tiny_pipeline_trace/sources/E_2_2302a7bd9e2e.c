typedef float float2 __attribute__((aligned(8),ext_vector_type(2)));
void E_2(float* restrict data0_2, float* restrict data1_2) {
  float2 val0 = (*((float2*)((data1_2+0))));
  *((float2*)((data0_2+0))) = val0;
}
