typedef float float2 __attribute__((aligned(8),ext_vector_type(2)));
typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
void r_2_1024_32_16384_32_16384_128_16384_128_256_2_2_4_4(float* restrict data0_2, float* restrict data1_2, float* restrict data2_1024, float* restrict data3_128, float* restrict data4_16384, float* restrict data5_128, float* restrict data6_16384, float* restrict data7_128, float* restrict data8_16384, float* restrict data9_128, float* restrict data10_256, float* restrict data11_2) {
  float2 val0 = (*((float2*)((data1_2+0))));
  float buf0[1];
  *(buf0+0) = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 1024; Ridx0++) {
    float val1 = (*(data2_1024+Ridx0));
    *(buf0+0) = ((*(buf0+0))+(val1*val1));
  }
  float buf1[1];
  *(buf1+0) = 0.0f;
  for (int Ridx2 = 0; Ridx2 < 32; Ridx2++) {
    float4 val2 = (*((float4*)((data3_128+(Ridx2<<2)))));
    *(buf1+0) = ((*(buf1+0))+(val2[0]*val2[0])+(val2[1]*val2[1])+(val2[2]*val2[2])+(val2[3]*val2[3]));
  }
  float buf2[1];
  *(buf2+0) = 0.0f;
  for (int Ridx3 = 0; Ridx3 < 16384; Ridx3++) {
    float val3 = (*(data4_16384+Ridx3));
    *(buf2+0) = ((*(buf2+0))+(val3*val3));
  }
  float buf3[1];
  *(buf3+0) = 0.0f;
  for (int Ridx5 = 0; Ridx5 < 32; Ridx5++) {
    float4 val4 = (*((float4*)((data5_128+(Ridx5<<2)))));
    *(buf3+0) = ((*(buf3+0))+(val4[0]*val4[0])+(val4[1]*val4[1])+(val4[2]*val4[2])+(val4[3]*val4[3]));
  }
  float buf4[1];
  *(buf4+0) = 0.0f;
  for (int Ridx6 = 0; Ridx6 < 16384; Ridx6++) {
    float val5 = (*(data6_16384+Ridx6));
    *(buf4+0) = ((*(buf4+0))+(val5*val5));
  }
  float buf5[1];
  *(buf5+0) = 0.0f;
  for (int Ridx8 = 0; Ridx8 < 128; Ridx8++) {
    float val6 = (*(data7_128+Ridx8));
    *(buf5+0) = ((*(buf5+0))+(val6*val6));
  }
  float buf6[1];
  *(buf6+0) = 0.0f;
  for (int Ridx9 = 0; Ridx9 < 16384; Ridx9++) {
    float val7 = (*(data8_16384+Ridx9));
    *(buf6+0) = ((*(buf6+0))+(val7*val7));
  }
  float buf7[1];
  *(buf7+0) = 0.0f;
  for (int Ridx11 = 0; Ridx11 < 128; Ridx11++) {
    float val8 = (*(data9_128+Ridx11));
    *(buf7+0) = ((*(buf7+0))+(val8*val8));
  }
  float buf8[1];
  *(buf8+0) = 0.0f;
  for (int Ridx12 = 0; Ridx12 < 256; Ridx12++) {
    float val9 = (*(data10_256+Ridx12));
    *(buf8+0) = ((*(buf8+0))+(val9*val9));
  }
  float buf9[1];
  *(buf9+0) = 0.0f;
  for (int Ridx14 = 0; Ridx14 < 2; Ridx14++) {
    float val10 = (*(data11_2+Ridx14));
    *(buf9+0) = ((*(buf9+0))+(val10*val10));
  }
  float buf10[1];
  *(buf10+0) = 0.0f;
  for (int Ridx15 = 0; Ridx15 < 2; Ridx15++) {
    float val11 = (*(data1_2+Ridx15));
    *(buf10+0) = ((*(buf10+0))+(val11*val11));
  }
  float alu33 = (0.5f/(__builtin_sqrtf(((*(buf0+0))+(*(buf1+0))+(*(buf2+0))+(*(buf3+0))+(*(buf4+0))+(*(buf5+0))+(*(buf6+0))+(*(buf7+0))+(*(buf8+0))+(*(buf9+0))+(*(buf10+0))))+9.999999974752427e-07f));
  float alu34 = ((1.0f<alu33)?1.0f:alu33);
  *((float2*)((data0_2+0))) = (float2){(val0[0]*alu34),(val0[1]*alu34)};
}
