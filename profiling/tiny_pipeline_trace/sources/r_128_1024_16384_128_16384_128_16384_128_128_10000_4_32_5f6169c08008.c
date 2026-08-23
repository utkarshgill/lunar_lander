typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
void r_128_1024_16384_128_16384_128_16384_128_128_10000_4_32(float* restrict data0_128, float* restrict data2_1024, float* restrict data4_16384, float* restrict data5_128, float* restrict data6_16384, float* restrict data7_128, float* restrict data8_16384, float* restrict data9_128, float* restrict data1_128, float* restrict data10_10000, float* restrict data3_128) {
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
  for (int Ridx0 = 0; Ridx0 < 1024; Ridx0++) {
    float val0 = (*(data2_1024+Ridx0));
    *(buf8+0) = ((*(buf8+0))+(val0*val0));
  }
  for (int Ridx3 = 0; Ridx3 < 16384; Ridx3++) {
    float val1 = (*(data4_16384+Ridx3));
    *(buf7+0) = ((*(buf7+0))+(val1*val1));
  }
  for (int Ridx5 = 0; Ridx5 < 128; Ridx5++) {
    float val2 = (*(data5_128+Ridx5));
    *(buf6+0) = ((*(buf6+0))+(val2*val2));
  }
  for (int Ridx6 = 0; Ridx6 < 16384; Ridx6++) {
    float val3 = (*(data6_16384+Ridx6));
    *(buf5+0) = ((*(buf5+0))+(val3*val3));
  }
  for (int Ridx8 = 0; Ridx8 < 128; Ridx8++) {
    float val4 = (*(data7_128+Ridx8));
    *(buf4+0) = ((*(buf4+0))+(val4*val4));
  }
  for (int Ridx9 = 0; Ridx9 < 16384; Ridx9++) {
    float val5 = (*(data8_16384+Ridx9));
    *(buf3+0) = ((*(buf3+0))+(val5*val5));
  }
  for (int Ridx11 = 0; Ridx11 < 128; Ridx11++) {
    float val6 = (*(data9_128+Ridx11));
    *(buf2+0) = ((*(buf2+0))+(val6*val6));
  }
  for (int Ridx12 = 0; Ridx12 < 128; Ridx12++) {
    float val7 = (*(data1_128+Ridx12));
    *(buf1+0) = ((*(buf1+0))+(val7*val7));
  }
  for (int Ridx13 = 0; Ridx13 < 10000; Ridx13++) {
    float val8 = (*(data10_10000+Ridx13));
    *(buf0+0) = ((*(buf0+0))+val8);
  }
  for (int Lidx14 = 0; Lidx14 < 128; Lidx14++) {
    float val9 = (*(data1_128+Lidx14));
    float4 val10 = (*((float4*)((data3_128+0))));
    float4 val11 = (*((float4*)((data3_128+4))));
    float4 val12 = (*((float4*)((data3_128+8))));
    float4 val13 = (*((float4*)((data3_128+12))));
    float4 val14 = (*((float4*)((data3_128+16))));
    float4 val15 = (*((float4*)((data3_128+20))));
    float4 val16 = (*((float4*)((data3_128+24))));
    float4 val17 = (*((float4*)((data3_128+28))));
    float4 val18 = (*((float4*)((data3_128+32))));
    float4 val19 = (*((float4*)((data3_128+36))));
    float4 val20 = (*((float4*)((data3_128+40))));
    float4 val21 = (*((float4*)((data3_128+44))));
    float4 val22 = (*((float4*)((data3_128+48))));
    float4 val23 = (*((float4*)((data3_128+52))));
    float4 val24 = (*((float4*)((data3_128+56))));
    float4 val25 = (*((float4*)((data3_128+60))));
    float4 val26 = (*((float4*)((data3_128+64))));
    float4 val27 = (*((float4*)((data3_128+68))));
    float4 val28 = (*((float4*)((data3_128+72))));
    float4 val29 = (*((float4*)((data3_128+76))));
    float4 val30 = (*((float4*)((data3_128+80))));
    float4 val31 = (*((float4*)((data3_128+84))));
    float4 val32 = (*((float4*)((data3_128+88))));
    float4 val33 = (*((float4*)((data3_128+92))));
    float4 val34 = (*((float4*)((data3_128+96))));
    float4 val35 = (*((float4*)((data3_128+100))));
    float4 val36 = (*((float4*)((data3_128+104))));
    float4 val37 = (*((float4*)((data3_128+108))));
    float4 val38 = (*((float4*)((data3_128+112))));
    float4 val39 = (*((float4*)((data3_128+116))));
    float4 val40 = (*((float4*)((data3_128+120))));
    float4 val41 = (*((float4*)((data3_128+124))));
    float val42 = (*(buf0+0));
    float alu27 = (0.5f/(__builtin_sqrtf(((*(buf8+0))+(val10[0]*val10[0])+(val10[1]*val10[1])+(val10[2]*val10[2])+(val10[3]*val10[3])+(val11[0]*val11[0])+(val11[1]*val11[1])+(val11[2]*val11[2])+(val11[3]*val11[3])+(val12[0]*val12[0])+(val12[1]*val12[1])+(val12[2]*val12[2])+(val12[3]*val12[3])+(val13[0]*val13[0])+(val13[1]*val13[1])+(val13[2]*val13[2])+(val13[3]*val13[3])+(val14[0]*val14[0])+(val14[1]*val14[1])+(val14[2]*val14[2])+(val14[3]*val14[3])+(val15[0]*val15[0])+(val15[1]*val15[1])+(val15[2]*val15[2])+(val15[3]*val15[3])+(val16[0]*val16[0])+(val16[1]*val16[1])+(val16[2]*val16[2])+(val16[3]*val16[3])+(val17[0]*val17[0])+(val17[1]*val17[1])+(val17[2]*val17[2])+(val17[3]*val17[3])+(val18[0]*val18[0])+(val18[1]*val18[1])+(val18[2]*val18[2])+(val18[3]*val18[3])+(val19[0]*val19[0])+(val19[1]*val19[1])+(val19[2]*val19[2])+(val19[3]*val19[3])+(val20[0]*val20[0])+(val20[1]*val20[1])+(val20[2]*val20[2])+(val20[3]*val20[3])+(val21[0]*val21[0])+(val21[1]*val21[1])+(val21[2]*val21[2])+(val21[3]*val21[3])+(val22[0]*val22[0])+(val22[1]*val22[1])+(val22[2]*val22[2])+(val22[3]*val22[3])+(val23[0]*val23[0])+(val23[1]*val23[1])+(val23[2]*val23[2])+(val23[3]*val23[3])+(val24[0]*val24[0])+(val24[1]*val24[1])+(val24[2]*val24[2])+(val24[3]*val24[3])+(val25[0]*val25[0])+(val25[1]*val25[1])+(val25[2]*val25[2])+(val25[3]*val25[3])+(val26[0]*val26[0])+(val26[1]*val26[1])+(val26[2]*val26[2])+(val26[3]*val26[3])+(val27[0]*val27[0])+(val27[1]*val27[1])+(val27[2]*val27[2])+(val27[3]*val27[3])+(val28[0]*val28[0])+(val28[1]*val28[1])+(val28[2]*val28[2])+(val28[3]*val28[3])+(val29[0]*val29[0])+(val29[1]*val29[1])+(val29[2]*val29[2])+(val29[3]*val29[3])+(val30[0]*val30[0])+(val30[1]*val30[1])+(val30[2]*val30[2])+(val30[3]*val30[3])+(val31[0]*val31[0])+(val31[1]*val31[1])+(val31[2]*val31[2])+(val31[3]*val31[3])+(val32[0]*val32[0])+(val32[1]*val32[1])+(val32[2]*val32[2])+(val32[3]*val32[3])+(val33[0]*val33[0])+(val33[1]*val33[1])+(val33[2]*val33[2])+(val33[3]*val33[3])+(val34[0]*val34[0])+(val34[1]*val34[1])+(val34[2]*val34[2])+(val34[3]*val34[3])+(val35[0]*val35[0])+(val35[1]*val35[1])+(val35[2]*val35[2])+(val35[3]*val35[3])+(val36[0]*val36[0])+(val36[1]*val36[1])+(val36[2]*val36[2])+(val36[3]*val36[3])+(val37[0]*val37[0])+(val37[1]*val37[1])+(val37[2]*val37[2])+(val37[3]*val37[3])+(val38[0]*val38[0])+(val38[1]*val38[1])+(val38[2]*val38[2])+(val38[3]*val38[3])+(val39[0]*val39[0])+(val39[1]*val39[1])+(val39[2]*val39[2])+(val39[3]*val39[3])+(val40[0]*val40[0])+(val40[1]*val40[1])+(val40[2]*val40[2])+(val40[3]*val40[3])+(val41[0]*val41[0])+(val41[1]*val41[1])+(val41[2]*val41[2])+(val41[3]*val41[3])+(*(buf7+0))+(*(buf6+0))+(*(buf5+0))+(*(buf4+0))+(*(buf3+0))+(*(buf2+0))+(*(buf1+0))+(val42*val42*3.999999620418748e-08f)))+9.999999974752427e-07f));
    float alu28 = ((1.0f<alu27)?1.0f:alu27);
    *(data0_128+Lidx14) = (val9*alu28);
  }
}
