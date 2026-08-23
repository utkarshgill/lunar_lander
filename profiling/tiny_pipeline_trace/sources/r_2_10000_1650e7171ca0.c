typedef float float2 __attribute__((aligned(8),ext_vector_type(2)));
void r_2_10000(float* restrict data0_2, float* restrict data2_20000, float* restrict data1_2) {
  float buf0[2];
  *(buf0+0) = 0.0f;
  *(buf0+1) = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 10000; Ridx0++) {
    float2 val0 = (*((float2*)((data2_20000+(Ridx0<<1)))));
    *(buf0+0) = ((*(buf0+0))+val0[0]);
    *(buf0+1) = ((*(buf0+1))+val0[1]);
  }
  float2 val1 = (*((float2*)((data1_2+0))));
  float alu5 = ((val1[0]<-5.0f)?-5.0f:val1[0]);
  float alu6 = ((2.0f<alu5)?2.0f:alu5);
  float alu7 = (alu6*1.4426950216293335f);
  _Bool alu8 = (alu7!=alu7);
  float alu9 = ((alu7!=((float)(-__builtin_inff())))?alu7:0.0f);
  float alu10 = (alu8?0.0f:alu9);
  float alu11 = ((alu7!=((float)(__builtin_inff())))?alu10:0.0f);
  float alu12 = ((alu11<0.0f)?-0.5f:0.5f);
  int cast0 = ((int)((alu11+alu12)));
  float alu13 = (alu11-((float)(cast0)));
  int alu14 = (cast0>>1);
  float alu15 = (((alu7<128.0f)!=1)?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu13)+0.0013392627006396651f)*alu13)+0.009618384763598442f)*alu13)+0.055503472685813904f)*alu13)+0.24022644758224487f)*alu13)+0.6931471824645996f)*alu13)+1.0f)*__builtin_bit_cast(float, (int)(((alu14+127)<<23)))*__builtin_bit_cast(float, (int)((((cast0-alu14)+127)<<23)))));
  float alu16 = ((alu7<-150.0f)?0.0f:alu15);
  float alu17 = (alu8?((float)(__builtin_nanf(""))):alu16);
  float alu18 = ((val1[1]<-5.0f)?-5.0f:val1[1]);
  float alu19 = ((2.0f<alu18)?2.0f:alu18);
  float alu20 = (alu19*1.4426950216293335f);
  _Bool alu21 = (alu20!=alu20);
  float alu22 = ((alu20!=((float)(-__builtin_inff())))?alu20:0.0f);
  float alu23 = (alu21?0.0f:alu22);
  float alu24 = ((alu20!=((float)(__builtin_inff())))?alu23:0.0f);
  float alu25 = ((alu24<0.0f)?-0.5f:0.5f);
  int cast1 = ((int)((alu24+alu25)));
  float alu26 = (alu24-((float)(cast1)));
  int alu27 = (cast1>>1);
  float alu28 = (((alu20<128.0f)!=1)?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu26)+0.0013392627006396651f)*alu26)+0.009618384763598442f)*alu26)+0.055503472685813904f)*alu26)+0.24022644758224487f)*alu26)+0.6931471824645996f)*alu26)+1.0f)*__builtin_bit_cast(float, (int)(((alu27+127)<<23)))*__builtin_bit_cast(float, (int)((((cast1-alu27)+127)<<23)))));
  float alu29 = ((alu20<-150.0f)?0.0f:alu28);
  float alu30 = (alu21?((float)(__builtin_nanf(""))):alu29);
  *((float2*)((data0_2+0))) = (float2){-((*(buf0+0))/alu17),-((*(buf0+1))/alu30)};
}
