; Exact runtime path sampled from the Lunar Lander matrix worker:
; aten::mm -> APL_sgemm -> libBLAS+0x8e894 -> libBLAS+0x91320
; The hottest sampled instruction was libBLAS+0x537590.
; Static shared-cache address: 0x18149C590.

0x18149C578   zero     {za}
0x18149C57C   cmp      x20, #0x1
0x18149C580   b.lt     0x18149c5b8
0x18149C584   mov      x13, x7
0x18149C588   mov      x4, x15
0x18149C58C   mov      x5, x20
0x18149C590   ld1w     { z16.s, z24.s }, pn8/z, [x13]
0x18149C594   ld1w     { z17.s, z25.s }, pn8/z, [x4]
0x18149C598   fmopa    za0.s, p0/m, p0/m, z17.s, z16.s
0x18149C59C   fmopa    za1.s, p0/m, p0/m, z17.s, z24.s
0x18149C5A0   fmopa    za2.s, p0/m, p0/m, z25.s, z16.s
0x18149C5A4   fmopa    za3.s, p0/m, p0/m, z25.s, z24.s
0x18149C5A8   add      x4, x4, x9
0x18149C5AC   add      x13, x13, x10
0x18149C5B0   subs     x5, x5, #0x1
0x18149C5B4   b.ne     0x18149c590
0x18149C5B8   mov      x13, #0x0
0x18149C5BC   mov      x4, x0
0x18149C5C0   mov      x5, x24
0x18149C5C4   mov      { z0.b - z3.b }, za0h.b[w13, 0x0:0x3]
0x18149C5C8   st1w     { z0.s, z1.s }, pn9, [x4]
0x18149C5CC   st1w     { z2.s, z3.s }, pn9, [x5]
0x18149C5D0   add      x13, x13, #0x4
0x18149C5D4   add      x5, x5, x12
0x18149C5D8   add      x4, x4, x12
0x18149C5DC   cmp      x13, #0x40
0x18149C5E0   b.ne     0x18149c5c4
