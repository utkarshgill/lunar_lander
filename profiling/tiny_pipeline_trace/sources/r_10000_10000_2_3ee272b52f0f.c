typedef float float2 __attribute__((aligned(8),ext_vector_type(2)));
void r_10000_10000_2(float* restrict data0_1, float* restrict data1_10000, float* restrict data2_10000, float* restrict data3_10000, float* restrict data4_2) {
  float buf0[1];
  *(buf0+0) = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 10000; Ridx0++) {
    float val0 = (*(data1_10000+Ridx0));
    float val1 = (*(data2_10000+Ridx0));
    float alu1 = -(val0*val1);
    float alu2 = ((val0<0.800000011920929f)?0.800000011920929f:val0);
    float alu3 = ((1.2000000476837158f<alu2)?1.2000000476837158f:alu2);
    float alu4 = -(alu3*val1);
    float alu5 = ((alu1<alu4)?alu4:alu1);
    *(buf0+0) = ((*(buf0+0))+alu5);
  }
  float buf1[1];
  *(buf1+0) = 0.0f;
  for (int Ridx1 = 0; Ridx1 < 10000; Ridx1++) {
    float val2 = (*(data3_10000+Ridx1));
    *(buf1+0) = ((*(buf1+0))+(val2*val2));
  }
  float2 val3 = (*((float2*)((data4_2+0))));
  float alu11 = ((val3[1]<-5.0f)?-5.0f:val3[1]);
  float alu12 = ((2.0f<alu11)?2.0f:alu11);
  float alu13 = ((val3[0]<-5.0f)?-5.0f:val3[0]);
  float alu14 = ((2.0f<alu13)?2.0f:alu13);
  *(data0_1+0) = (((*(buf0+0))*9.999999747378752e-05f)+((*(buf1+0))*9.999999747378752e-05f)+((alu12+alu14+2.837877035140991f)*-0.0010000000474974513f));
}
