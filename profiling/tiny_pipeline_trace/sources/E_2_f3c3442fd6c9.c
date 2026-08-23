typedef float float2 __attribute__((aligned(8),ext_vector_type(2)));
void E_2(float* restrict data0_2, float* restrict data1_2, float* restrict data2_1, float* restrict data3_2, float* restrict data4_1, float* restrict data5_2, float* restrict data6_1) {
  float2 val0 = (*((float2*)((data1_2+0))));
  float val1 = (*(data2_1+0));
  float2 val2 = (*((float2*)((data3_2+0))));
  float val3 = (*(data4_1+0));
  float alu0 = (1.0f-val3);
  float2 val4 = (*((float2*)((data5_2+0))));
  float val5 = (*(data6_1+0));
  float alu1 = (1.0f-val5);
  *((float2*)((data0_2+0))) = (float2){(val0[0]-(val1*(val2[0]/(alu0*(__builtin_sqrtf((val4[0]/alu1))+9.99999993922529e-09f))))),(val0[1]-(val1*(val2[1]/(alu0*(__builtin_sqrtf((val4[1]/alu1))+9.99999993922529e-09f)))))};
}
