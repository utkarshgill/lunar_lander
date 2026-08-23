	.build_version macos, 26, 0	sdk_version 26, 5
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_r_2000_8_5_4_4_128             ; -- Begin function r_2000_8_5_4_4_128
	.p2align	2
_r_2000_8_5_4_4_128:                    ; @r_2000_8_5_4_4_128
	.cfi_startproc
; %bb.0:
	stp	d11, d10, [sp, #-32]!           ; 16-byte Folded Spill
	stp	d9, d8, [sp, #16]               ; 16-byte Folded Spill
	.cfi_def_cfa_offset 32
	.cfi_offset b8, -8
	.cfi_offset b9, -16
	.cfi_offset b10, -24
	.cfi_offset b11, -32
	mov	x8, #0                          ; =0x0
	add	x9, x2, #32
	mov	w10, #2560                      ; =0xa00
LBB0_1:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_2 Depth 2
                                        ;       Child Loop BB0_3 Depth 3
	mov	x11, #0                         ; =0x0
	madd	x12, x8, x10, x0
	mov	x13, x9
LBB0_2:                                 ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_3 Depth 3
	movi.2d	v0, #0000000000000000
	lsl	x14, x11, #4
	movi.2d	v1, #0000000000000000
	mov	x15, x1
	movi.2d	v17, #0000000000000000
	mov	x16, x13
	movi.2d	v25, #0000000000000000
	mov	w17, #128                       ; =0x80
	movi.2d	v27, #0000000000000000
	movi.2d	v26, #0000000000000000
	movi.2d	v18, #0000000000000000
	movi.2d	v19, #0000000000000000
	movi.2d	v20, #0000000000000000
	movi.2d	v21, #0000000000000000
	movi.2d	v4, #0000000000000000
	movi.2d	v2, #0000000000000000
	movi.2d	v22, #0000000000000000
	movi.2d	v23, #0000000000000000
	movi.2d	v24, #0000000000000000
	movi.2d	v16, #0000000000000000
	movi.2d	v5, #0000000000000000
	movi.2d	v6, #0000000000000000
	movi.2d	v7, #0000000000000000
	movi.2d	v3, #0000000000000000
LBB0_3:                                 ;   Parent Loop BB0_1 Depth=1
                                        ;     Parent Loop BB0_2 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldur	q28, [x16, #-32]
	ldr	s29, [x15, #512]
	ldr	s30, [x15, #1024]
	ldr	s31, [x15, #1536]
	ldr	s8, [x15, #2048]
	ld1r.4s	{ v9 }, [x15], #4
	fmla.4s	v0, v28, v9
	fmla.4s	v1, v28, v29[0]
	fmla.4s	v17, v28, v30[0]
	fmla.4s	v25, v28, v31[0]
	fmla.4s	v27, v28, v8[0]
	ldp	q28, q10, [x16, #-16]
	fmla.4s	v26, v28, v9
	fmla.4s	v18, v28, v29[0]
	fmla.4s	v19, v28, v30[0]
	fmla.4s	v20, v28, v31[0]
	fmla.4s	v21, v28, v8[0]
	fmla.4s	v4, v10, v9
	fmla.4s	v2, v10, v29[0]
	fmla.4s	v22, v10, v30[0]
	fmla.4s	v23, v10, v31[0]
	fmla.4s	v24, v10, v8[0]
	ldr	q28, [x16, #16]
	fmla.4s	v16, v28, v9
	fmla.4s	v5, v28, v29[0]
	fmla.4s	v6, v28, v30[0]
	fmla.4s	v7, v28, v31[0]
	add	x16, x16, #512
	fmla.4s	v3, v28, v8[0]
	subs	x17, x17, #1
	b.ne	LBB0_3
; %bb.4:                                ;   in Loop: Header=BB0_2 Depth=2
	add	x14, x12, x14, lsl #2
	str	q17, [x14, #1024]
	str	q25, [x14, #1536]
	str	q27, [x14, #2048]
	stp	q0, q26, [x14]
	stp	q1, q18, [x14, #512]
	str	q19, [x14, #1040]
	str	q20, [x14, #1552]
	str	q21, [x14, #2064]
	str	q22, [x14, #1056]
	str	q23, [x14, #1568]
	str	q24, [x14, #2080]
	stp	q4, q16, [x14, #32]
	stp	q2, q5, [x14, #544]
	str	q6, [x14, #1072]
	str	q7, [x14, #1584]
	add	x11, x11, #1
	add	x13, x13, #64
	str	q3, [x14, #2096]
	cmp	x11, #8
	b.ne	LBB0_2
; %bb.5:                                ;   in Loop: Header=BB0_1 Depth=1
	add	x8, x8, #1
	add	x1, x1, #2560
	cmp	x8, #2000
	b.ne	LBB0_1
; %bb.6:
	ldp	d9, d8, [sp, #16]               ; 16-byte Folded Reload
	ldp	d11, d10, [sp], #32             ; 16-byte Folded Reload
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols
