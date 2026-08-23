typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
void r_10000_1024_128_16384_32_16384_128_16384_128_128_4(float* restrict data0_1, float* restrict data1_1, float* restrict data2_10000, float* restrict data3_1024, float* restrict data4_128, float* restrict data5_16384, float* restrict data6_128, float* restrict data7_16384, float* restrict data8_128, float* restrict data9_16384, float* restrict data10_128, float* restrict data11_128) {
  float val0 = (*(data1_1+0));
  float buf0[1];
  *(buf0+0) = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 10000; Ridx0++) {
    float val1 = (*(data2_10000+Ridx0));
    *(buf0+0) = ((*(buf0+0))+val1);
  }
  float val2 = (*(buf0+0));
  float buf1[1];
  *(buf1+0) = 0.0f;
  for (int Ridx1 = 0; Ridx1 < 1024; Ridx1++) {
    float val3 = (*(data3_1024+Ridx1));
    *(buf1+0) = ((*(buf1+0))+(val3*val3));
  }
  float buf2[1];
  *(buf2+0) = 0.0f;
  for (int Ridx3 = 0; Ridx3 < 128; Ridx3++) {
    float val4 = (*(data4_128+Ridx3));
    *(buf2+0) = ((*(buf2+0))+(val4*val4));
  }
  float buf3[1];
  *(buf3+0) = 0.0f;
  for (int Ridx4 = 0; Ridx4 < 16384; Ridx4++) {
    float val5 = (*(data5_16384+Ridx4));
    *(buf3+0) = ((*(buf3+0))+(val5*val5));
  }
  float buf4[1];
  *(buf4+0) = 0.0f;
  for (int Ridx6 = 0; Ridx6 < 32; Ridx6++) {
    float4 val6 = (*((float4*)((data6_128+(Ridx6<<2)))));
    *(buf4+0) = ((*(buf4+0))+(val6[0]*val6[0])+(val6[1]*val6[1])+(val6[2]*val6[2])+(val6[3]*val6[3]));
  }
  float buf5[1];
  *(buf5+0) = 0.0f;
  for (int Ridx7 = 0; Ridx7 < 16384; Ridx7++) {
    float val7 = (*(data7_16384+Ridx7));
    *(buf5+0) = ((*(buf5+0))+(val7*val7));
  }
  float buf6[1];
  *(buf6+0) = 0.0f;
  for (int Ridx9 = 0; Ridx9 < 128; Ridx9++) {
    float val8 = (*(data8_128+Ridx9));
    *(buf6+0) = ((*(buf6+0))+(val8*val8));
  }
  float buf7[1];
  *(buf7+0) = 0.0f;
  for (int Ridx10 = 0; Ridx10 < 16384; Ridx10++) {
    float val9 = (*(data9_16384+Ridx10));
    *(buf7+0) = ((*(buf7+0))+(val9*val9));
  }
  float buf8[1];
  *(buf8+0) = 0.0f;
  for (int Ridx12 = 0; Ridx12 < 128; Ridx12++) {
    float val10 = (*(data10_128+Ridx12));
    *(buf8+0) = ((*(buf8+0))+(val10*val10));
  }
  float buf9[1];
  *(buf9+0) = 0.0f;
  for (int Ridx13 = 0; Ridx13 < 128; Ridx13++) {
    float val11 = (*(data11_128+Ridx13));
    *(buf9+0) = ((*(buf9+0))+(val11*val11));
  }
  float alu30 = (0.5f/(__builtin_sqrtf(((*(buf1+0))+(*(buf2+0))+(*(buf3+0))+(*(buf4+0))+(*(buf5+0))+(*(buf6+0))+(*(buf7+0))+(*(buf8+0))+(*(buf9+0))+(val2*val2*3.999999620418748e-08f)))+9.999999974752427e-07f));
  float alu31 = ((1.0f<alu30)?1.0f:alu30);
  float alu32 = (val2*alu31);
  *(data0_1+0) = ((0.9990000128746033f*val0)+(alu32*alu32*3.9999996370720936e-11f));
}
