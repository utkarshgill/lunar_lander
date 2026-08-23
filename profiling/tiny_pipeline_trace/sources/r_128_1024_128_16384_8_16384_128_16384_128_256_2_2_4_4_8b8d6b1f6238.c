typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
void r_128_1024_128_16384_8_16384_128_16384_128_256_2_2_4_4(float* restrict data0_128, float* restrict data2_1024, float* restrict data1_128, float* restrict data3_16384, float* restrict data4_128, float* restrict data5_16384, float* restrict data6_128, float* restrict data7_16384, float* restrict data8_128, float* restrict data9_256, float* restrict data10_2, float* restrict data11_2) {
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
  for (int Ridx0 = 0; Ridx0 < 1024; Ridx0++) {
    float val0 = (*(data2_1024+Ridx0));
    *(buf10+0) = ((*(buf10+0))+(val0*val0));
  }
  for (int Ridx2 = 0; Ridx2 < 128; Ridx2++) {
    float val1 = (*(data1_128+Ridx2));
    *(buf9+0) = ((*(buf9+0))+(val1*val1));
  }
  for (int Ridx3 = 0; Ridx3 < 16384; Ridx3++) {
    float val2 = (*(data3_16384+Ridx3));
    *(buf8+0) = ((*(buf8+0))+(val2*val2));
  }
  for (int Ridx5 = 0; Ridx5 < 8; Ridx5++) {
    int alu17 = (Ridx5<<4);
    float4 val3 = (*((float4*)((data4_128+alu17))));
    float4 val4 = (*((float4*)((data4_128+(alu17+4)))));
    float4 val5 = (*((float4*)((data4_128+(alu17+8)))));
    float4 val6 = (*((float4*)((data4_128+(alu17+12)))));
    *(buf7+0) = ((*(buf7+0))+(val3[0]*val3[0])+(val3[1]*val3[1])+(val3[2]*val3[2])+(val3[3]*val3[3])+(val4[0]*val4[0])+(val4[1]*val4[1])+(val4[2]*val4[2])+(val4[3]*val4[3])+(val5[0]*val5[0])+(val5[1]*val5[1])+(val5[2]*val5[2])+(val5[3]*val5[3])+(val6[0]*val6[0])+(val6[1]*val6[1])+(val6[2]*val6[2])+(val6[3]*val6[3]));
  }
  for (int Ridx6 = 0; Ridx6 < 16384; Ridx6++) {
    float val7 = (*(data5_16384+Ridx6));
    *(buf6+0) = ((*(buf6+0))+(val7*val7));
  }
  for (int Ridx8 = 0; Ridx8 < 128; Ridx8++) {
    float val8 = (*(data6_128+Ridx8));
    *(buf5+0) = ((*(buf5+0))+(val8*val8));
  }
  for (int Ridx9 = 0; Ridx9 < 16384; Ridx9++) {
    float val9 = (*(data7_16384+Ridx9));
    *(buf4+0) = ((*(buf4+0))+(val9*val9));
  }
  for (int Ridx11 = 0; Ridx11 < 128; Ridx11++) {
    float val10 = (*(data8_128+Ridx11));
    *(buf3+0) = ((*(buf3+0))+(val10*val10));
  }
  for (int Ridx12 = 0; Ridx12 < 256; Ridx12++) {
    float val11 = (*(data9_256+Ridx12));
    *(buf2+0) = ((*(buf2+0))+(val11*val11));
  }
  for (int Ridx14 = 0; Ridx14 < 2; Ridx14++) {
    float val12 = (*(data10_2+Ridx14));
    *(buf1+0) = ((*(buf1+0))+(val12*val12));
  }
  for (int Ridx15 = 0; Ridx15 < 2; Ridx15++) {
    float val13 = (*(data11_2+Ridx15));
    *(buf0+0) = ((*(buf0+0))+(val13*val13));
  }
  for (int Lidx16 = 0; Lidx16 < 128; Lidx16++) {
    float val14 = (*(data1_128+Lidx16));
    float alu34 = (0.5f/(__builtin_sqrtf(((*(buf10+0))+(*(buf9+0))+(*(buf8+0))+(*(buf7+0))+(*(buf6+0))+(*(buf5+0))+(*(buf4+0))+(*(buf3+0))+(*(buf2+0))+(*(buf1+0))+(*(buf0+0))))+9.999999974752427e-07f));
    float alu35 = ((1.0f<alu34)?1.0f:alu34);
    *(data0_128+Lidx16) = (val14*alu35);
  }
}
