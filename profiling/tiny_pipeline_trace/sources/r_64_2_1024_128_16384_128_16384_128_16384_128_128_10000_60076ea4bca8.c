typedef float float2 __attribute__((aligned(8),ext_vector_type(2)));
void r_64_2_1024_128_16384_128_16384_128_16384_128_128_10000(float* restrict data0_128, float* restrict data2_1024, float* restrict data3_128, float* restrict data4_16384, float* restrict data5_128, float* restrict data6_16384, float* restrict data7_128, float* restrict data8_16384, float* restrict data1_128, float* restrict data9_128, float* restrict data10_10000) {
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
  for (int Ridx0 = 0; Ridx0 < 1024; Ridx0++) {
    float val0 = (*(data2_1024+Ridx0));
    *(buf9+0) = ((*(buf9+0))+(val0*val0));
  }
  for (int Ridx2 = 0; Ridx2 < 128; Ridx2++) {
    float val1 = (*(data3_128+Ridx2));
    *(buf8+0) = ((*(buf8+0))+(val1*val1));
  }
  for (int Ridx3 = 0; Ridx3 < 16384; Ridx3++) {
    float val2 = (*(data4_16384+Ridx3));
    *(buf7+0) = ((*(buf7+0))+(val2*val2));
  }
  for (int Ridx5 = 0; Ridx5 < 128; Ridx5++) {
    float val3 = (*(data5_128+Ridx5));
    *(buf6+0) = ((*(buf6+0))+(val3*val3));
  }
  for (int Ridx6 = 0; Ridx6 < 16384; Ridx6++) {
    float val4 = (*(data6_16384+Ridx6));
    *(buf5+0) = ((*(buf5+0))+(val4*val4));
  }
  for (int Ridx8 = 0; Ridx8 < 128; Ridx8++) {
    float val5 = (*(data7_128+Ridx8));
    *(buf4+0) = ((*(buf4+0))+(val5*val5));
  }
  for (int Ridx9 = 0; Ridx9 < 16384; Ridx9++) {
    float val6 = (*(data8_16384+Ridx9));
    *(buf3+0) = ((*(buf3+0))+(val6*val6));
  }
  for (int Ridx11 = 0; Ridx11 < 128; Ridx11++) {
    float val7 = (*(data1_128+Ridx11));
    *(buf2+0) = ((*(buf2+0))+(val7*val7));
  }
  for (int Ridx12 = 0; Ridx12 < 128; Ridx12++) {
    float val8 = (*(data9_128+Ridx12));
    *(buf1+0) = ((*(buf1+0))+(val8*val8));
  }
  for (int Ridx13 = 0; Ridx13 < 10000; Ridx13++) {
    float val9 = (*(data10_10000+Ridx13));
    *(buf0+0) = ((*(buf0+0))+val9);
  }
  for (int Lidx14 = 0; Lidx14 < 64; Lidx14++) {
    int alu30 = (Lidx14<<1);
    float2 val10 = (*((float2*)((data1_128+alu30))));
    float val11 = (*(buf0+0));
    float alu31 = (0.5f/(__builtin_sqrtf(((*(buf9+0))+(*(buf8+0))+(*(buf7+0))+(*(buf6+0))+(*(buf5+0))+(*(buf4+0))+(*(buf3+0))+(*(buf2+0))+(*(buf1+0))+(val11*val11*3.999999620418748e-08f)))+9.999999974752427e-07f));
    float alu32 = ((1.0f<alu31)?1.0f:alu31);
    *((float2*)((data0_128+alu30))) = (float2){(val10[0]*alu32),(val10[1]*alu32)};
  }
}
