typedef float float2 __attribute__((aligned(8),ext_vector_type(2)));
void E_64_2(float* restrict data0_128, float* restrict data1_128) {
  for (int Lidx0 = 0; Lidx0 < 64; Lidx0++) {
    int alu0 = (Lidx0<<1);
    float2 val0 = (*((float2*)((data1_128+alu0))));
    *((float2*)((data0_128+alu0))) = val0;
  }
}
