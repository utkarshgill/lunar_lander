typedef float float2 __attribute__((aligned(8),ext_vector_type(2)));
void r_2_10000_10000(float* restrict data0_2, float* restrict data1_2, float* restrict data2_10000, float* restrict data3_10000, float* restrict data4_20000, float* restrict data5_20000) {
  float2 val0 = (*((float2*)((data1_2+0))));
  _Bool alu0 = (val0[0]<-5.0f);
  float alu1 = (alu0?-5.0f:val0[0]);
  _Bool alu2 = (2.0f<alu1);
  float buf0[1];
  *(buf0+0) = 0.0f;
  for (int Ridx0 = 0; Ridx0 < 10000; Ridx0++) {
    float val1 = (*(data2_10000+Ridx0));
    _Bool alu4 = (val1<0.800000011920929f);
    float alu5 = (alu4?0.800000011920929f:val1);
    _Bool alu6 = (1.2000000476837158f<alu5);
    float val2 = (*(data3_10000+Ridx0));
    float alu7 = -(val1*val2);
    float alu8 = (alu6?1.2000000476837158f:alu5);
    float alu9 = -(alu8*val2);
    float alu10 = ((alu7!=alu9)?0.0f:4.999999873689376e-05f);
    float alu11 = ((alu7<alu9)?9.999999747378752e-05f:alu10);
    float alu12 = ((alu4|alu6)?0.0f:-(alu11*val2));
    float alu13 = ((alu9<alu7)?9.999999747378752e-05f:alu10);
    *(buf0+0) = ((*(buf0+0))+(val1*(alu12-(alu13*val2))));
  }
  float alu16 = -(*(buf0+0));
  float buf1[2];
  *(buf1+0) = 0.0f;
  *(buf1+1) = 0.0f;
  for (int Ridx1 = 0; Ridx1 < 10000; Ridx1++) {
    int alu19 = (Ridx1<<1);
    float2 val3 = (*((float2*)((data4_20000+alu19))));
    float2 val4 = (*((float2*)((data5_20000+alu19))));
    *(buf1+0) = ((*(buf1+0))+(val3[0]*val4[0]));
    *(buf1+1) = ((*(buf1+1))+(val3[1]*val4[1]));
  }
  float alu23 = (alu2?2.0f:alu1);
  float alu24 = (alu23*1.4426950216293335f);
  _Bool alu25 = (alu24!=alu24);
  float alu26 = ((alu24!=((float)(-__builtin_inff())))?alu24:0.0f);
  float alu27 = (alu25?0.0f:alu26);
  float alu28 = ((alu24!=((float)(__builtin_inff())))?alu27:0.0f);
  float alu29 = ((alu28<0.0f)?-0.5f:0.5f);
  int cast0 = ((int)((alu28+alu29)));
  float alu30 = (alu28-((float)(cast0)));
  int alu31 = (cast0>>1);
  float alu32 = (((alu24<128.0f)!=1)?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu30)+0.0013392627006396651f)*alu30)+0.009618384763598442f)*alu30)+0.055503472685813904f)*alu30)+0.24022644758224487f)*alu30)+0.6931471824645996f)*alu30)+1.0f)*__builtin_bit_cast(float, (int)(((alu31+127)<<23)))*__builtin_bit_cast(float, (int)((((cast0-alu31)+127)<<23)))));
  float alu33 = ((alu24<-150.0f)?0.0f:alu32);
  float alu34 = (alu25?((float)(__builtin_nanf(""))):alu33);
  float alu35 = ((alu0|alu2)?0.0f:((alu16-((((*(buf1+0))/alu34)/alu34)*alu34))+-0.0010000000474974513f));
  _Bool alu36 = (val0[1]<-5.0f);
  float alu37 = (alu36?-5.0f:val0[1]);
  _Bool alu38 = (2.0f<alu37);
  float alu39 = (alu38?2.0f:alu37);
  float alu40 = (alu39*1.4426950216293335f);
  _Bool alu41 = (alu40!=alu40);
  float alu42 = ((alu40!=((float)(-__builtin_inff())))?alu40:0.0f);
  float alu43 = (alu41?0.0f:alu42);
  float alu44 = ((alu40!=((float)(__builtin_inff())))?alu43:0.0f);
  float alu45 = ((alu44<0.0f)?-0.5f:0.5f);
  int cast1 = ((int)((alu44+alu45)));
  float alu46 = (alu44-((float)(cast1)));
  int alu47 = (cast1>>1);
  float alu48 = (((alu40<128.0f)!=1)?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu46)+0.0013392627006396651f)*alu46)+0.009618384763598442f)*alu46)+0.055503472685813904f)*alu46)+0.24022644758224487f)*alu46)+0.6931471824645996f)*alu46)+1.0f)*__builtin_bit_cast(float, (int)(((alu47+127)<<23)))*__builtin_bit_cast(float, (int)((((cast1-alu47)+127)<<23)))));
  float alu49 = ((alu40<-150.0f)?0.0f:alu48);
  float alu50 = (alu41?((float)(__builtin_nanf(""))):alu49);
  float alu51 = ((alu36|alu38)?0.0f:((alu16-((((*(buf1+1))/alu50)/alu50)*alu50))+-0.0010000000474974513f));
  *((float2*)((data0_2+0))) = (float2){alu35,alu51};
}
