typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
void r_16384_256_128_16384_128_16384_128_16384_128_256_2_2_4(float* restrict data0_16384, float* restrict data2_1024, float* restrict data3_128, float* restrict data4_16384, float* restrict data5_128, float* restrict data6_16384, float* restrict data7_128, float* restrict data1_16384, float* restrict data8_128, float* restrict data9_256, float* restrict data10_2, float* restrict data11_2) {
  float buf0[1];
  *(buf0+0) = 0.0f;
  float buf1[1];
  *(buf1+0) = 0.0f;
  float buf2[1];
  *(buf2+0) = 0.0f;
  float buf3[1];
  *(buf3+0) = 0.0f;
  float buf4[1];
  *(buf4+0) = 0.0f;
  float buf5[1];
  *(buf5+0) = 0.0f;
  float buf6[1];
  *(buf6+0) = 0.0f;
  float buf7[1];
  *(buf7+0) = 0.0f;
  float buf8[1];
  *(buf8+0) = 0.0f;
  float buf9[1];
  *(buf9+0) = 0.0f;
  float buf10[1];
  *(buf10+0) = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 256; Ridx0++) {
    float4 val0 = (*((float4*)((data2_1024+(Ridx0<<2)))));
    *(buf10+0) = ((*(buf10+0))+(val0[0]*val0[0])+(val0[1]*val0[1])+(val0[2]*val0[2])+(val0[3]*val0[3]));
  }
  for (int Ridx2 = 0; Ridx2 < 128; Ridx2++) {
    float val1 = (*(data3_128+Ridx2));
    *(buf9+0) = ((*(buf9+0))+(val1*val1));
  }
  for (int Ridx3 = 0; Ridx3 < 16384; Ridx3++) {
    float val2 = (*(data4_16384+Ridx3));
    *(buf8+0) = ((*(buf8+0))+(val2*val2));
  }
  for (int Ridx5 = 0; Ridx5 < 128; Ridx5++) {
    float val3 = (*(data5_128+Ridx5));
    *(buf7+0) = ((*(buf7+0))+(val3*val3));
  }
  for (int Ridx6 = 0; Ridx6 < 16384; Ridx6++) {
    float val4 = (*(data6_16384+Ridx6));
    *(buf6+0) = ((*(buf6+0))+(val4*val4));
  }
  for (int Ridx8 = 0; Ridx8 < 128; Ridx8++) {
    float val5 = (*(data7_128+Ridx8));
    *(buf5+0) = ((*(buf5+0))+(val5*val5));
  }
  for (int Ridx9 = 0; Ridx9 < 16384; Ridx9++) {
    float val6 = (*(data1_16384+Ridx9));
    *(buf4+0) = ((*(buf4+0))+(val6*val6));
  }
  for (int Ridx11 = 0; Ridx11 < 128; Ridx11++) {
    float val7 = (*(data8_128+Ridx11));
    *(buf3+0) = ((*(buf3+0))+(val7*val7));
  }
  for (int Ridx12 = 0; Ridx12 < 256; Ridx12++) {
    float val8 = (*(data9_256+Ridx12));
    *(buf2+0) = ((*(buf2+0))+(val8*val8));
  }
  for (int Ridx14 = 0; Ridx14 < 2; Ridx14++) {
    float val9 = (*(data10_2+Ridx14));
    *(buf1+0) = ((*(buf1+0))+(val9*val9));
  }
  for (int Ridx15 = 0; Ridx15 < 2; Ridx15++) {
    float val10 = (*(data11_2+Ridx15));
    *(buf0+0) = ((*(buf0+0))+(val10*val10));
  }
  for (int Lidx16 = 0; Lidx16 < 16384; Lidx16++) {
    float val11 = (*(data1_16384+Lidx16));
    float alu33 = (0.5f/(__builtin_sqrtf(((*(buf10+0))+(*(buf9+0))+(*(buf8+0))+(*(buf7+0))+(*(buf6+0))+(*(buf5+0))+(*(buf4+0))+(*(buf3+0))+(*(buf2+0))+(*(buf1+0))+(*(buf0+0))))+9.999999974752427e-07f));
    float alu34 = ((1.0f<alu33)?1.0f:alu33);
    *(data0_16384+Lidx16) = (val11*alu34);
  }
}
