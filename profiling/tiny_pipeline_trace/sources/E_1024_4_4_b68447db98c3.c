typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
void E_1024_4_4(float* restrict data0_16384, float* restrict data1_16384) {
  for (int Lidx0 = 0; Lidx0 < 1024; Lidx0++) {
    int alu0 = (Lidx0<<4);
    float4 val0 = (*((float4*)((data1_16384+alu0))));
    *((float4*)((data0_16384+alu0))) = val0;
    int alu2 = (alu0+4);
    float4 val1 = (*((float4*)((data1_16384+alu2))));
    *((float4*)((data0_16384+alu2))) = val1;
    int alu4 = (alu0+8);
    float4 val2 = (*((float4*)((data1_16384+alu4))));
    *((float4*)((data0_16384+alu4))) = val2;
    int alu6 = (alu0+12);
    float4 val3 = (*((float4*)((data1_16384+alu6))));
    *((float4*)((data0_16384+alu6))) = val3;
  }
}
