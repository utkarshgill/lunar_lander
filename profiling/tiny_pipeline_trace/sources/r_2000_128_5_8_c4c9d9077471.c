
void r_2000_128_5_8(float* restrict data0_1280000, float* restrict data1_80000, float* restrict data2_1024, float* restrict data3_128) {
  for (int Lidx1 = 0; Lidx1 < 2000; Lidx1++) {
    for (int Lidx2 = 0; Lidx2 < 128; Lidx2++) {
      int alu0 = ((Lidx1*640)+Lidx2);
      float buf0[5];
      *(buf0+0) = 0.0f;
      *(buf0+1) = 0.0f;
      *(buf0+2) = 0.0f;
      *(buf0+3) = 0.0f;
      *(buf0+4) = 0.0f;
      for (int Ridx0 = 0; Ridx0 < 8; Ridx0++) {
        int alu6 = ((Lidx1*40)+Ridx0);
        float val0 = (*(data1_80000+alu6));
        float val1 = (*(data2_1024+((Lidx2<<3)+Ridx0)));
        *(buf0+0) = ((*(buf0+0))+(val0*val1));
        float val2 = (*(data1_80000+(alu6+8)));
        *(buf0+1) = ((*(buf0+1))+(val2*val1));
        float val3 = (*(data1_80000+(alu6+16)));
        *(buf0+2) = ((*(buf0+2))+(val3*val1));
        float val4 = (*(data1_80000+(alu6+24)));
        *(buf0+3) = ((*(buf0+3))+(val4*val1));
        float val5 = (*(data1_80000+(alu6+32)));
        *(buf0+4) = ((*(buf0+4))+(val5*val1));
      }
      float val6 = (*(data3_128+Lidx2));
      *(data0_1280000+alu0) = ((*(buf0+0))+val6);
      *(data0_1280000+(alu0+128)) = ((*(buf0+1))+val6);
      *(data0_1280000+(alu0+256)) = ((*(buf0+2))+val6);
      *(data0_1280000+(alu0+384)) = ((*(buf0+3))+val6);
      *(data0_1280000+(alu0+512)) = ((*(buf0+4))+val6);
    }
  }
}
