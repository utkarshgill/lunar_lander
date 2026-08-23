typedef float float4 __attribute__((aligned(16),ext_vector_type(4)));
void r_2500_32_4_4_32_4(float* restrict data0_1280000, float* restrict data1_1280000, float* restrict data2_16384) {
  for (int Lidx1 = 0; Lidx1 < 2500; Lidx1++) {
    int alu0 = (Lidx1<<9);
    for (int Lidx2 = 0; Lidx2 < 32; Lidx2++) {
      int alu1 = (Lidx2<<2);
      int alu2 = (alu0+alu1);
      float buf0[16];
      *(buf0+0) = 0.0f;
      *(buf0+1) = 0.0f;
      *(buf0+2) = 0.0f;
      *(buf0+3) = 0.0f;
      *(buf0+4) = 0.0f;
      *(buf0+5) = 0.0f;
      *(buf0+6) = 0.0f;
      *(buf0+7) = 0.0f;
      *(buf0+8) = 0.0f;
      *(buf0+9) = 0.0f;
      *(buf0+10) = 0.0f;
      *(buf0+11) = 0.0f;
      *(buf0+12) = 0.0f;
      *(buf0+13) = 0.0f;
      *(buf0+14) = 0.0f;
      *(buf0+15) = 0.0f;
      for (int Ridx0 = 0; Ridx0 < 32; Ridx0++) {
        int alu19 = ((Ridx0<<2)+alu0);
        float4 val0 = (*((float4*)((data1_1280000+alu19))));
        int alu20 = ((Ridx0<<9)+alu1);
        float4 val1 = (*((float4*)((data2_16384+alu20))));
        float4 val2 = (*((float4*)((data2_16384+(alu20+128)))));
        float4 val3 = (*((float4*)((data2_16384+(alu20+256)))));
        float4 val4 = (*((float4*)((data2_16384+(alu20+384)))));
        *(buf0+0) = ((*(buf0+0))+(val0[0]*val1[0])+(val0[1]*val2[0])+(val0[2]*val3[0])+(val0[3]*val4[0]));
        *(buf0+1) = ((*(buf0+1))+(val0[0]*val1[1])+(val0[1]*val2[1])+(val0[2]*val3[1])+(val0[3]*val4[1]));
        *(buf0+2) = ((*(buf0+2))+(val0[0]*val1[2])+(val0[1]*val2[2])+(val0[2]*val3[2])+(val0[3]*val4[2]));
        *(buf0+3) = ((*(buf0+3))+(val0[0]*val1[3])+(val0[1]*val2[3])+(val0[2]*val3[3])+(val0[3]*val4[3]));
        float4 val5 = (*((float4*)((data1_1280000+(alu19+128)))));
        *(buf0+4) = ((*(buf0+4))+(val5[0]*val1[0])+(val5[1]*val2[0])+(val5[2]*val3[0])+(val5[3]*val4[0]));
        *(buf0+5) = ((*(buf0+5))+(val5[0]*val1[1])+(val5[1]*val2[1])+(val5[2]*val3[1])+(val5[3]*val4[1]));
        *(buf0+6) = ((*(buf0+6))+(val5[0]*val1[2])+(val5[1]*val2[2])+(val5[2]*val3[2])+(val5[3]*val4[2]));
        *(buf0+7) = ((*(buf0+7))+(val5[0]*val1[3])+(val5[1]*val2[3])+(val5[2]*val3[3])+(val5[3]*val4[3]));
        float4 val6 = (*((float4*)((data1_1280000+(alu19+256)))));
        *(buf0+8) = ((*(buf0+8))+(val6[0]*val1[0])+(val6[1]*val2[0])+(val6[2]*val3[0])+(val6[3]*val4[0]));
        *(buf0+9) = ((*(buf0+9))+(val6[0]*val1[1])+(val6[1]*val2[1])+(val6[2]*val3[1])+(val6[3]*val4[1]));
        *(buf0+10) = ((*(buf0+10))+(val6[0]*val1[2])+(val6[1]*val2[2])+(val6[2]*val3[2])+(val6[3]*val4[2]));
        *(buf0+11) = ((*(buf0+11))+(val6[0]*val1[3])+(val6[1]*val2[3])+(val6[2]*val3[3])+(val6[3]*val4[3]));
        float4 val7 = (*((float4*)((data1_1280000+(alu19+384)))));
        *(buf0+12) = ((*(buf0+12))+(val7[0]*val1[0])+(val7[1]*val2[0])+(val7[2]*val3[0])+(val7[3]*val4[0]));
        *(buf0+13) = ((*(buf0+13))+(val7[0]*val1[1])+(val7[1]*val2[1])+(val7[2]*val3[1])+(val7[3]*val4[1]));
        *(buf0+14) = ((*(buf0+14))+(val7[0]*val1[2])+(val7[1]*val2[2])+(val7[2]*val3[2])+(val7[3]*val4[2]));
        *(buf0+15) = ((*(buf0+15))+(val7[0]*val1[3])+(val7[1]*val2[3])+(val7[2]*val3[3])+(val7[3]*val4[3]));
      }
      *((float4*)((data0_1280000+alu2))) = (float4){(*(buf0+0)),(*(buf0+1)),(*(buf0+2)),(*(buf0+3))};
      *((float4*)((data0_1280000+(alu2+128)))) = (float4){(*(buf0+4)),(*(buf0+5)),(*(buf0+6)),(*(buf0+7))};
      *((float4*)((data0_1280000+(alu2+256)))) = (float4){(*(buf0+8)),(*(buf0+9)),(*(buf0+10)),(*(buf0+11))};
      *((float4*)((data0_1280000+(alu2+384)))) = (float4){(*(buf0+12)),(*(buf0+13)),(*(buf0+14)),(*(buf0+15))};
    }
  }
}
