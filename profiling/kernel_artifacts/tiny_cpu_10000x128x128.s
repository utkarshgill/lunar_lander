	.build_version macos, 26, 0	sdk_version 26, 5
	.section	__TEXT,__text,regular,pure_instructions
	.globl	_r_2500_32_4_4_32_4             ; -- Begin function r_2500_32_4_4_32_4
	.p2align	2
_r_2500_32_4_4_32_4:                    ; @r_2500_32_4_4_32_4
	.cfi_startproc
; %bb.0:
	mov	x8, #0                          ; =0x0
LBB0_1:                                 ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_2 Depth 2
                                        ;       Child Loop BB0_3 Depth 3
	mov	x9, #0                          ; =0x0
	add	x10, x0, x8, lsl #11
	mov	x11, x2
LBB0_2:                                 ;   Parent Loop BB0_1 Depth=1
                                        ; =>  This Loop Header: Depth=2
                                        ;       Child Loop BB0_3 Depth 3
	lsl	x12, x9, #2
	movi.2d	v0, #0000000000000000
	mov	x13, x11
	mov	x14, x1
	mov	w15, #32                        ; =0x20
	movi.2d	v1, #0000000000000000
	movi.2d	v2, #0000000000000000
	movi.2d	v3, #0000000000000000
LBB0_3:                                 ;   Parent Loop BB0_1 Depth=1
                                        ;     Parent Loop BB0_2 Depth=2
                                        ; =>    This Inner Loop Header: Depth=3
	ldr	q4, [x14]
	ldr	q5, [x13]
	ldr	q6, [x13, #512]
	ldr	q7, [x13, #1024]
	ldr	q16, [x13, #1536]
	fmla.4s	v0, v5, v4[0]
	fmla.4s	v0, v6, v4[1]
	fmla.4s	v0, v7, v4[2]
	fmla.4s	v0, v16, v4[3]
	ldr	q4, [x14, #512]
	fmla.4s	v1, v5, v4[0]
	fmla.4s	v1, v6, v4[1]
	fmla.4s	v1, v7, v4[2]
	fmla.4s	v1, v16, v4[3]
	ldr	q4, [x14, #1024]
	fmla.4s	v2, v5, v4[0]
	fmla.4s	v2, v6, v4[1]
	fmla.4s	v2, v7, v4[2]
	fmla.4s	v2, v16, v4[3]
	ldr	q4, [x14, #1536]
	fmla.4s	v3, v5, v4[0]
	fmla.4s	v3, v6, v4[1]
	fmla.4s	v3, v7, v4[2]
	fmla.4s	v3, v16, v4[3]
	add	x14, x14, #16
	add	x13, x13, #2048
	subs	x15, x15, #1
	b.ne	LBB0_3
; %bb.4:                                ;   in Loop: Header=BB0_2 Depth=2
	add	x12, x10, x12, lsl #2
	str	q0, [x12]
	str	q1, [x12, #512]
	str	q2, [x12, #1024]
	str	q3, [x12, #1536]
	add	x9, x9, #1
	add	x11, x11, #16
	cmp	x9, #32
	b.ne	LBB0_2
; %bb.5:                                ;   in Loop: Header=BB0_1 Depth=1
	add	x8, x8, #1
	add	x1, x1, #2048
	cmp	x8, #2500
	b.ne	LBB0_1
; %bb.6:
	ret
	.cfi_endproc
                                        ; -- End function
.subsections_via_symbols
