
void r_2000_128_5_2(float* restrict data0_1280000, float* restrict data1_1280000, float* restrict data3_20000, float* restrict data2_2, float* restrict data4_256) {
  for (int Lidx1 = 0; Lidx1 < 2000; Lidx1++) {
    for (int Lidx2 = 0; Lidx2 < 128; Lidx2++) {
      int alu0 = ((Lidx1*640)+Lidx2);
      float val0 = (*(data1_1280000+alu0));
      float buf0[5];
      *(buf0+0) = 0.0f;
      *(buf0+1) = 0.0f;
      *(buf0+2) = 0.0f;
      *(buf0+3) = 0.0f;
      *(buf0+4) = 0.0f;
      for (int Ridx0 = 0; Ridx0 < 2; Ridx0++) {
        int alu6 = ((Lidx1*10)+Ridx0);
        float val1 = (*(data3_20000+alu6));
        float val2 = (*(data2_2+Ridx0));
        float alu7 = ((val2<-5.0f)?-5.0f:val2);
        float alu8 = ((2.0f<alu7)?2.0f:alu7);
        float alu9 = (alu8*1.4426950216293335f);
        _Bool alu10 = (alu9!=alu9);
        float alu11 = ((alu9!=((float)(-__builtin_inff())))?alu9:0.0f);
        float alu12 = (alu10?0.0f:alu11);
        float alu13 = ((alu9!=((float)(__builtin_inff())))?alu12:0.0f);
        float alu14 = ((alu13<0.0f)?-0.5f:0.5f);
        int cast0 = ((int)((alu13+alu14)));
        float alu15 = (alu13-((float)(cast0)));
        int alu16 = (cast0>>1);
        float alu17 = (((alu9<128.0f)!=1)?((float)(__builtin_inff())):(((((((((((((0.00015359208919107914f*alu15)+0.0013392627006396651f)*alu15)+0.009618384763598442f)*alu15)+0.055503472685813904f)*alu15)+0.24022644758224487f)*alu15)+0.6931471824645996f)*alu15)+1.0f)*__builtin_bit_cast(float, (int)(((alu16+127)<<23)))*__builtin_bit_cast(float, (int)((((cast0-alu16)+127)<<23)))));
        float alu18 = ((alu9<-150.0f)?0.0f:alu17);
        float alu19 = (alu10?((float)(__builtin_nanf(""))):alu18);
        float val3 = (*(data4_256+((Ridx0<<7)+Lidx2)));
        *(buf0+0) = ((*(buf0+0))+((val1/alu19)*val3));
        float val4 = (*(data3_20000+(alu6+2)));
        *(buf0+1) = ((*(buf0+1))+((val4/alu19)*val3));
        float val5 = (*(data3_20000+(alu6+4)));
        *(buf0+2) = ((*(buf0+2))+((val5/alu19)*val3));
        float val6 = (*(data3_20000+(alu6+6)));
        *(buf0+3) = ((*(buf0+3))+((val6/alu19)*val3));
        float val7 = (*(data3_20000+(alu6+8)));
        *(buf0+4) = ((*(buf0+4))+((val7/alu19)*val3));
      }
      float alu26 = ((0.0f<val0)?-(*(buf0+0)):0.0f);
      *(data0_1280000+alu0) = alu26;
      int alu28 = (alu0+128);
      float val8 = (*(data1_1280000+alu28));
      float alu29 = ((0.0f<val8)?-(*(buf0+1)):0.0f);
      *(data0_1280000+alu28) = alu29;
      int alu31 = (alu0+256);
      float val9 = (*(data1_1280000+alu31));
      float alu32 = ((0.0f<val9)?-(*(buf0+2)):0.0f);
      *(data0_1280000+alu31) = alu32;
      int alu34 = (alu0+384);
      float val10 = (*(data1_1280000+alu34));
      float alu35 = ((0.0f<val10)?-(*(buf0+3)):0.0f);
      *(data0_1280000+alu34) = alu35;
      int alu37 = (alu0+512);
      float val11 = (*(data1_1280000+alu37));
      float alu38 = ((0.0f<val11)?-(*(buf0+4)):0.0f);
      *(data0_1280000+alu37) = alu38;
    }
  }
}
