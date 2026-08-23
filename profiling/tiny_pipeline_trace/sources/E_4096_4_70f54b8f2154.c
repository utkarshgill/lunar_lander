typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
void E_4096_4(float* restrict data0_16384, float* restrict data1_16384, float* restrict data2_1, float* restrict data3_16384, float* restrict data4_1, float* restrict data5_16384, float* restrict data6_1) {
  for (int Lidx0 = 0; Lidx0 < 4096; Lidx0++) {
    int alu0 = (Lidx0<<2);
    float4 val0 = (*((float4*)((data1_16384+alu0))));
    float val1 = (*(data2_1+0));
    float4 val2 = (*((float4*)((data3_16384+alu0))));
    float val3 = (*(data4_1+0));
    float alu1 = (1.0f-val3);
    float4 val4 = (*((float4*)((data5_16384+alu0))));
    float val5 = (*(data6_1+0));
    float alu2 = (1.0f-val5);
    *((float4*)((data0_16384+alu0))) = (float4){(val0[0]-(val1*(val2[0]/(alu1*(__builtin_sqrtf((val4[0]/alu2))+9.99999993922529e-09f))))),(val0[1]-(val1*(val2[1]/(alu1*(__builtin_sqrtf((val4[1]/alu2))+9.99999993922529e-09f))))),(val0[2]-(val1*(val2[2]/(alu1*(__builtin_sqrtf((val4[2]/alu2))+9.99999993922529e-09f))))),(val0[3]-(val1*(val2[3]/(alu1*(__builtin_sqrtf((val4[3]/alu2))+9.99999993922529e-09f)))))};
  }
}
