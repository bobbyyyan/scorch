	.file	"scorch-spmm-csr-raw.cpp"
	.text
	.section	.rodata
	.type	_ZStL19piecewise_construct, @object
	.size	_ZStL19piecewise_construct, 1
_ZStL19piecewise_construct:
	.zero	1
	.section	.text._ZnwmPv,"axG",@progbits,_ZnwmPv,comdat
	.weak	_ZnwmPv
	.type	_ZnwmPv, @function
_ZnwmPv:
.LFB379:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE379:
	.size	_ZnwmPv, .-_ZnwmPv
	.section	.text._ZdlPvS_,"axG",@progbits,_ZdlPvS_,comdat
	.weak	_ZdlPvS_
	.type	_ZdlPvS_, @function
_ZdlPvS_:
.LFB381:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE381:
	.size	_ZdlPvS_, .-_ZdlPvS_
	.section	.text._ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE12_Vector_implD2Ev,"axG",@progbits,_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE12_Vector_implD5Ev,comdat
	.align 2
	.weak	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE12_Vector_implD2Ev
	.type	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE12_Vector_implD2Ev, @function
_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE12_Vector_implD2Ev:
.LFB875:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSaISt6vectorIPfSaIS0_EEED2Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE875:
	.size	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE12_Vector_implD2Ev, .-_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE12_Vector_implD2Ev
	.weak	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE12_Vector_implD1Ev
	.set	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE12_Vector_implD1Ev,_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE12_Vector_implD2Ev
	.section	.text._ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EEC2Ev,"axG",@progbits,_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EEC5Ev,comdat
	.align 2
	.weak	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EEC2Ev
	.type	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EEC2Ev, @function
_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EEC2Ev:
.LFB877:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE12_Vector_implC1Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE877:
	.size	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EEC2Ev, .-_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EEC2Ev
	.weak	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EEC1Ev
	.set	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EEC1Ev,_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EEC2Ev
	.section	.text._ZNSt6vectorIS_IPfSaIS0_EESaIS2_EEC2Ev,"axG",@progbits,_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EEC5Ev,comdat
	.align 2
	.weak	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EEC2Ev
	.type	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EEC2Ev, @function
_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EEC2Ev:
.LFB879:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EEC2Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE879:
	.size	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EEC2Ev, .-_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EEC2Ev
	.weak	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EEC1Ev
	.set	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EEC1Ev,_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EEC2Ev
	.section	.text._ZN11TensorIndexC2Ev,"axG",@progbits,_ZN11TensorIndexC5Ev,comdat
	.align 2
	.weak	_ZN11TensorIndexC2Ev
	.type	_ZN11TensorIndexC2Ev, @function
_ZN11TensorIndexC2Ev:
.LFB881:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EEC1Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE881:
	.size	_ZN11TensorIndexC2Ev, .-_ZN11TensorIndexC2Ev
	.weak	_ZN11TensorIndexC1Ev
	.set	_ZN11TensorIndexC1Ev,_ZN11TensorIndexC2Ev
	.section	.text._ZN11TensorIndexD2Ev,"axG",@progbits,_ZN11TensorIndexD5Ev,comdat
	.align 2
	.weak	_ZN11TensorIndexD2Ev
	.type	_ZN11TensorIndexD2Ev, @function
_ZN11TensorIndexD2Ev:
.LFB884:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EED1Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE884:
	.size	_ZN11TensorIndexD2Ev, .-_ZN11TensorIndexD2Ev
	.weak	_ZN11TensorIndexD1Ev
	.set	_ZN11TensorIndexD1Ev,_ZN11TensorIndexD2Ev
	.section	.text._ZN13TensorStorageC2Ev,"axG",@progbits,_ZN13TensorStorageC5Ev,comdat
	.align 2
	.weak	_ZN13TensorStorageC2Ev
	.type	_ZN13TensorStorageC2Ev, @function
_ZN13TensorStorageC2Ev:
.LFB886:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZN11TensorIndexC1Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE886:
	.size	_ZN13TensorStorageC2Ev, .-_ZN13TensorStorageC2Ev
	.weak	_ZN13TensorStorageC1Ev
	.set	_ZN13TensorStorageC1Ev,_ZN13TensorStorageC2Ev
	.section	.text._ZN13TensorStorageD2Ev,"axG",@progbits,_ZN13TensorStorageD5Ev,comdat
	.align 2
	.weak	_ZN13TensorStorageD2Ev
	.type	_ZN13TensorStorageD2Ev, @function
_ZN13TensorStorageD2Ev:
.LFB889:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZN11TensorIndexD1Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE889:
	.size	_ZN13TensorStorageD2Ev, .-_ZN13TensorStorageD2Ev
	.weak	_ZN13TensorStorageD1Ev
	.set	_ZN13TensorStorageD1Ev,_ZN13TensorStorageD2Ev
	.section	.text._ZNSt12_Vector_baseIiSaIiEE12_Vector_implD2Ev,"axG",@progbits,_ZNSt12_Vector_baseIiSaIiEE12_Vector_implD5Ev,comdat
	.align 2
	.weak	_ZNSt12_Vector_baseIiSaIiEE12_Vector_implD2Ev
	.type	_ZNSt12_Vector_baseIiSaIiEE12_Vector_implD2Ev, @function
_ZNSt12_Vector_baseIiSaIiEE12_Vector_implD2Ev:
.LFB894:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSaIiED2Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE894:
	.size	_ZNSt12_Vector_baseIiSaIiEE12_Vector_implD2Ev, .-_ZNSt12_Vector_baseIiSaIiEE12_Vector_implD2Ev
	.weak	_ZNSt12_Vector_baseIiSaIiEE12_Vector_implD1Ev
	.set	_ZNSt12_Vector_baseIiSaIiEE12_Vector_implD1Ev,_ZNSt12_Vector_baseIiSaIiEE12_Vector_implD2Ev
	.section	.text._ZNSt12_Vector_baseIiSaIiEEC2Ev,"axG",@progbits,_ZNSt12_Vector_baseIiSaIiEEC5Ev,comdat
	.align 2
	.weak	_ZNSt12_Vector_baseIiSaIiEEC2Ev
	.type	_ZNSt12_Vector_baseIiSaIiEEC2Ev, @function
_ZNSt12_Vector_baseIiSaIiEEC2Ev:
.LFB896:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIiSaIiEE12_Vector_implC1Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE896:
	.size	_ZNSt12_Vector_baseIiSaIiEEC2Ev, .-_ZNSt12_Vector_baseIiSaIiEEC2Ev
	.weak	_ZNSt12_Vector_baseIiSaIiEEC1Ev
	.set	_ZNSt12_Vector_baseIiSaIiEEC1Ev,_ZNSt12_Vector_baseIiSaIiEEC2Ev
	.section	.text._ZNSt6vectorIiSaIiEEC2Ev,"axG",@progbits,_ZNSt6vectorIiSaIiEEC5Ev,comdat
	.align 2
	.weak	_ZNSt6vectorIiSaIiEEC2Ev
	.type	_ZNSt6vectorIiSaIiEEC2Ev, @function
_ZNSt6vectorIiSaIiEEC2Ev:
.LFB898:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIiSaIiEEC2Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE898:
	.size	_ZNSt6vectorIiSaIiEEC2Ev, .-_ZNSt6vectorIiSaIiEEC2Ev
	.weak	_ZNSt6vectorIiSaIiEEC1Ev
	.set	_ZNSt6vectorIiSaIiEEC1Ev,_ZNSt6vectorIiSaIiEEC2Ev
	.section	.text._ZN6TensorC2Ev,"axG",@progbits,_ZN6TensorC5Ev,comdat
	.align 2
	.weak	_ZN6TensorC2Ev
	.type	_ZN6TensorC2Ev, @function
_ZN6TensorC2Ev:
.LFB900:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZN13TensorStorageC1Ev
	movq	-8(%rbp), %rax
	addq	$32, %rax
	movq	%rax, %rdi
	call	_ZNSt6vectorIiSaIiEEC1Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE900:
	.size	_ZN6TensorC2Ev, .-_ZN6TensorC2Ev
	.weak	_ZN6TensorC1Ev
	.set	_ZN6TensorC1Ev,_ZN6TensorC2Ev
	.section	.text._ZN6TensorD2Ev,"axG",@progbits,_ZN6TensorD5Ev,comdat
	.align 2
	.weak	_ZN6TensorD2Ev
	.type	_ZN6TensorD2Ev, @function
_ZN6TensorD2Ev:
.LFB903:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	addq	$32, %rax
	movq	%rax, %rdi
	call	_ZNSt6vectorIiSaIiEED1Ev
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZN13TensorStorageD1Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE903:
	.size	_ZN6TensorD2Ev, .-_ZN6TensorD2Ev
	.weak	_ZN6TensorD1Ev
	.set	_ZN6TensorD1Ev,_ZN6TensorD2Ev
	.section	.text._ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implD2Ev,"axG",@progbits,_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implD5Ev,comdat
	.align 2
	.weak	_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implD2Ev
	.type	_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implD2Ev, @function
_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implD2Ev:
.LFB908:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSaIPfED2Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE908:
	.size	_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implD2Ev, .-_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implD2Ev
	.weak	_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implD1Ev
	.set	_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implD1Ev,_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implD2Ev
	.section	.text._ZNSt12_Vector_baseIPfSaIS0_EEC2Ev,"axG",@progbits,_ZNSt12_Vector_baseIPfSaIS0_EEC5Ev,comdat
	.align 2
	.weak	_ZNSt12_Vector_baseIPfSaIS0_EEC2Ev
	.type	_ZNSt12_Vector_baseIPfSaIS0_EEC2Ev, @function
_ZNSt12_Vector_baseIPfSaIS0_EEC2Ev:
.LFB910:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implC1Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE910:
	.size	_ZNSt12_Vector_baseIPfSaIS0_EEC2Ev, .-_ZNSt12_Vector_baseIPfSaIS0_EEC2Ev
	.weak	_ZNSt12_Vector_baseIPfSaIS0_EEC1Ev
	.set	_ZNSt12_Vector_baseIPfSaIS0_EEC1Ev,_ZNSt12_Vector_baseIPfSaIS0_EEC2Ev
	.section	.text._ZNSt6vectorIPfSaIS0_EEC2Ev,"axG",@progbits,_ZNSt6vectorIPfSaIS0_EEC5Ev,comdat
	.align 2
	.weak	_ZNSt6vectorIPfSaIS0_EEC2Ev
	.type	_ZNSt6vectorIPfSaIS0_EEC2Ev, @function
_ZNSt6vectorIPfSaIS0_EEC2Ev:
.LFB912:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIPfSaIS0_EEC2Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE912:
	.size	_ZNSt6vectorIPfSaIS0_EEC2Ev, .-_ZNSt6vectorIPfSaIS0_EEC2Ev
	.weak	_ZNSt6vectorIPfSaIS0_EEC1Ev
	.set	_ZNSt6vectorIPfSaIS0_EEC1Ev,_ZNSt6vectorIPfSaIS0_EEC2Ev
	.text
	.globl	_Z8evaluateSt6vectorIiSaIiEES1_PKiS3_PKfS1_S5_
	.type	_Z8evaluateSt6vectorIiSaIiEES1_PKiS3_PKfS1_S5_, @function
_Z8evaluateSt6vectorIiSaIiEES1_PKiS3_PKfS1_S5_:
.LFB868:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA868
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$2232, %rsp
	.cfi_offset 13, -24
	.cfi_offset 12, -32
	.cfi_offset 3, -40
	movq	%rdi, -2200(%rbp)
	movq	%rsi, -2208(%rbp)
	movq	%rdx, -2216(%rbp)
	movq	%rcx, -2224(%rbp)
	movq	%r8, -2232(%rbp)
	movq	%r9, -2240(%rbp)
	movq	16(%rbp), %rax
	movq	%rax, -2248(%rbp)
	movq	24(%rbp), %rax
	movq	%rax, -2256(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -40(%rbp)
	xorl	%eax, %eax
	movq	-2208(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	_ZNSt6vectorIiSaIiEEixEm
	movl	(%rax), %eax
	movl	%eax, -2164(%rbp)
	movq	-2208(%rbp), %rax
	movl	$1, %esi
	movq	%rax, %rdi
	call	_ZNSt6vectorIiSaIiEEixEm
	movl	(%rax), %eax
	movl	%eax, -2160(%rbp)
	movq	-2216(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	_ZNSt6vectorIiSaIiEEixEm
	movl	(%rax), %eax
	movl	%eax, -2156(%rbp)
	movq	-2248(%rbp), %rax
	movl	$0, %esi
	movq	%rax, %rdi
	call	_ZNSt6vectorIiSaIiEEixEm
	movl	(%rax), %eax
	movl	%eax, -2152(%rbp)
	movq	-2248(%rbp), %rax
	movl	$1, %esi
	movq	%rax, %rdi
	call	_ZNSt6vectorIiSaIiEEixEm
	movl	(%rax), %eax
	movl	%eax, -2148(%rbp)
	movl	-2164(%rbp), %eax
	imull	-2160(%rbp), %eax
	movl	%eax, -2144(%rbp)
	movl	-2144(%rbp), %eax
	cltq
	salq	$2, %rax
	movq	%rax, %rdi
	call	malloc@PLT
	movq	%rax, -2104(%rbp)
	movl	$512, -2140(%rbp)
	movl	$0, -2184(%rbp)
.L29:
	movl	-2184(%rbp), %eax
	cmpl	-2156(%rbp), %eax
	setl	%al
	movzbl	%al, %eax
	testq	%rax, %rax
	je	.L20
	movl	-2184(%rbp), %eax
	movl	%eax, -2136(%rbp)
	movl	-2184(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-2224(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, -2132(%rbp)
	movl	-2184(%rbp), %eax
	cltq
	addq	$1, %rax
	leaq	0(,%rax,4), %rdx
	movq	-2224(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, -2128(%rbp)
	movl	$0, -2180(%rbp)
.L28:
	movl	-2180(%rbp), %eax
	cmpl	-2148(%rbp), %eax
	jge	.L21
	leaq	-2096(%rbp), %rdx
	movl	$0, %eax
	movl	$256, %ecx
	movq	%rdx, %rdi
	rep stosq
	movl	-2132(%rbp), %eax
	movl	%eax, -2176(%rbp)
.L25:
	movl	-2176(%rbp), %eax
	cmpl	-2128(%rbp), %eax
	jge	.L22
	movl	-2176(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-2232(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movl	%eax, -2124(%rbp)
	movl	$0, -2172(%rbp)
.L24:
	cmpl	$511, -2172(%rbp)
	jg	.L23
	movl	-2180(%rbp), %edx
	movl	-2172(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -2120(%rbp)
	movl	-2124(%rbp), %eax
	imull	-2148(%rbp), %eax
	movl	%eax, %edx
	movl	-2120(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -2116(%rbp)
	movl	-2172(%rbp), %eax
	cltq
	movss	-2096(%rbp,%rax,4), %xmm1
	movl	-2176(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-2240(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm2
	movl	-2116(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-2256(%rbp), %rax
	addq	%rdx, %rax
	movss	(%rax), %xmm0
	mulss	%xmm2, %xmm0
	addss	%xmm1, %xmm0
	movl	-2172(%rbp), %eax
	cltq
	movss	%xmm0, -2096(%rbp,%rax,4)
	addl	$1, -2172(%rbp)
	jmp	.L24
.L23:
	addl	$1, -2176(%rbp)
	jmp	.L25
.L22:
	movl	$0, -2168(%rbp)
.L27:
	cmpl	$511, -2168(%rbp)
	jg	.L26
	movl	-2180(%rbp), %edx
	movl	-2168(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -2112(%rbp)
	movl	-2136(%rbp), %eax
	imull	-2160(%rbp), %eax
	movl	%eax, %edx
	movl	-2112(%rbp), %eax
	addl	%edx, %eax
	movl	%eax, -2108(%rbp)
	movl	-2108(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-2104(%rbp), %rax
	addq	%rax, %rdx
	movl	-2168(%rbp), %eax
	cltq
	movss	-2096(%rbp,%rax,4), %xmm0
	movss	%xmm0, (%rdx)
	addl	$1, -2168(%rbp)
	jmp	.L27
.L26:
	addl	$512, -2180(%rbp)
	jmp	.L28
.L21:
	addl	$1, -2184(%rbp)
	jmp	.L29
.L20:
	movq	-2200(%rbp), %rax
	movq	%rax, %rdi
	call	_ZN6TensorC1Ev
	movq	$0, -2096(%rbp)
	movq	$0, -2088(%rbp)
	movq	$0, -2080(%rbp)
	leaq	-2096(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt6vectorIPfSaIS0_EEC1Ev
	movq	$0, -2072(%rbp)
	movq	$0, -2064(%rbp)
	movq	$0, -2056(%rbp)
	leaq	-2096(%rbp), %rax
	addq	$24, %rax
	movq	%rax, %rdi
	call	_ZNSt6vectorIPfSaIS0_EEC1Ev
	leaq	-2096(%rbp), %rax
	movq	%rax, %r12
	movl	$2, %r13d
	movq	-2200(%rbp), %rax
	movq	%r12, %rsi
	movq	%r13, %rdi
	movq	%r12, %rcx
	movq	%r13, %rbx
	movq	%rbx, %rdx
	movq	%rax, %rdi
.LEHB0:
	call	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EEaSESt16initializer_listIS2_E
.LEHE0:
	leaq	-2096(%rbp), %rbx
	addq	$48, %rbx
.L31:
	leaq	-2096(%rbp), %rax
	cmpq	%rax, %rbx
	je	.L30
	subq	$24, %rbx
	movq	%rbx, %rdi
	call	_ZNSt6vectorIPfSaIS0_EED1Ev
	jmp	.L31
.L30:
	movq	-2200(%rbp), %rax
	movq	-2104(%rbp), %rdx
	movq	%rdx, 24(%rax)
	nop
	movq	-40(%rbp), %rax
	xorq	%fs:40, %rax
	je	.L36
	jmp	.L38
.L37:
	endbr64
	movq	%rax, %r12
	leaq	-2096(%rbp), %rbx
	addq	$48, %rbx
.L35:
	leaq	-2096(%rbp), %rax
	cmpq	%rax, %rbx
	je	.L34
	subq	$24, %rbx
	movq	%rbx, %rdi
	call	_ZNSt6vectorIPfSaIS0_EED1Ev
	jmp	.L35
.L34:
	movq	%r12, %rbx
	movq	-2200(%rbp), %rax
	movq	%rax, %rdi
	call	_ZN6TensorD1Ev
	movq	%rbx, %rax
	movq	%rax, %rdi
.LEHB1:
	call	_Unwind_Resume@PLT
.LEHE1:
.L38:
	call	__stack_chk_fail@PLT
.L36:
	movq	-2200(%rbp), %rax
	addq	$2232, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE868:
	.globl	__gxx_personality_v0
	.section	.gcc_except_table,"a",@progbits
.LLSDA868:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE868-.LLSDACSB868
.LLSDACSB868:
	.uleb128 .LEHB0-.LFB868
	.uleb128 .LEHE0-.LEHB0
	.uleb128 .L37-.LFB868
	.uleb128 0
	.uleb128 .LEHB1-.LFB868
	.uleb128 .LEHE1-.LEHB1
	.uleb128 0
	.uleb128 0
.LLSDACSE868:
	.text
	.size	_Z8evaluateSt6vectorIiSaIiEES1_PKiS3_PKfS1_S5_, .-_Z8evaluateSt6vectorIiSaIiEES1_PKiS3_PKfS1_S5_
	.section	.text._ZNSt6vectorIiSaIiEEixEm,"axG",@progbits,_ZNSt6vectorIiSaIiEEixEm,comdat
	.align 2
	.weak	_ZNSt6vectorIiSaIiEEixEm
	.type	_ZNSt6vectorIiSaIiEEixEm, @function
_ZNSt6vectorIiSaIiEEixEm:
.LFB937:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	(%rax), %rax
	movq	-16(%rbp), %rdx
	salq	$2, %rdx
	addq	%rdx, %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE937:
	.size	_ZNSt6vectorIiSaIiEEixEm, .-_ZNSt6vectorIiSaIiEEixEm
	.section	.text._ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE12_Vector_implC2Ev,"axG",@progbits,_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE12_Vector_implC5Ev,comdat
	.align 2
	.weak	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE12_Vector_implC2Ev
	.type	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE12_Vector_implC2Ev, @function
_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE12_Vector_implC2Ev:
.LFB939:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSaISt6vectorIPfSaIS0_EEEC2Ev
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE17_Vector_impl_dataC2Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE939:
	.size	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE12_Vector_implC2Ev, .-_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE12_Vector_implC2Ev
	.weak	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE12_Vector_implC1Ev
	.set	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE12_Vector_implC1Ev,_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE12_Vector_implC2Ev
	.section	.text._ZNSaISt6vectorIPfSaIS0_EEED2Ev,"axG",@progbits,_ZNSaISt6vectorIPfSaIS0_EEED5Ev,comdat
	.align 2
	.weak	_ZNSaISt6vectorIPfSaIS0_EEED2Ev
	.type	_ZNSaISt6vectorIPfSaIS0_EEED2Ev, @function
_ZNSaISt6vectorIPfSaIS0_EEED2Ev:
.LFB942:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEED2Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE942:
	.size	_ZNSaISt6vectorIPfSaIS0_EEED2Ev, .-_ZNSaISt6vectorIPfSaIS0_EEED2Ev
	.weak	_ZNSaISt6vectorIPfSaIS0_EEED1Ev
	.set	_ZNSaISt6vectorIPfSaIS0_EEED1Ev,_ZNSaISt6vectorIPfSaIS0_EEED2Ev
	.section	.text._ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EED2Ev,"axG",@progbits,_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EED5Ev,comdat
	.align 2
	.weak	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EED2Ev
	.type	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EED2Ev, @function
_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EED2Ev:
.LFB945:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA945
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	16(%rax), %rdx
	movq	-8(%rbp), %rax
	movq	(%rax), %rax
	subq	%rax, %rdx
	movq	%rdx, %rax
	sarq	$3, %rax
	movq	%rax, %rdx
	movabsq	$-6148914691236517205, %rax
	imulq	%rdx, %rax
	movq	%rax, %rdx
	movq	-8(%rbp), %rax
	movq	(%rax), %rcx
	movq	-8(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE13_M_deallocateEPS3_m
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE12_Vector_implD1Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE945:
	.section	.gcc_except_table
.LLSDA945:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE945-.LLSDACSB945
.LLSDACSB945:
.LLSDACSE945:
	.section	.text._ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EED2Ev,"axG",@progbits,_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EED5Ev,comdat
	.size	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EED2Ev, .-_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EED2Ev
	.weak	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EED1Ev
	.set	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EED1Ev,_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EED2Ev
	.section	.text._ZNSt6vectorIS_IPfSaIS0_EESaIS2_EED2Ev,"axG",@progbits,_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EED5Ev,comdat
	.align 2
	.weak	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EED2Ev
	.type	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EED2Ev, @function
_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EED2Ev:
.LFB948:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA948
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE19_M_get_Tp_allocatorEv
	movq	%rax, %rdx
	movq	-8(%rbp), %rax
	movq	8(%rax), %rcx
	movq	-8(%rbp), %rax
	movq	(%rax), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZSt8_DestroyIPSt6vectorIPfSaIS1_EES3_EvT_S5_RSaIT0_E
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EED2Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE948:
	.section	.gcc_except_table
.LLSDA948:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE948-.LLSDACSB948
.LLSDACSB948:
.LLSDACSE948:
	.section	.text._ZNSt6vectorIS_IPfSaIS0_EESaIS2_EED2Ev,"axG",@progbits,_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EED5Ev,comdat
	.size	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EED2Ev, .-_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EED2Ev
	.weak	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EED1Ev
	.set	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EED1Ev,_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EED2Ev
	.section	.text._ZNSt12_Vector_baseIiSaIiEE12_Vector_implC2Ev,"axG",@progbits,_ZNSt12_Vector_baseIiSaIiEE12_Vector_implC5Ev,comdat
	.align 2
	.weak	_ZNSt12_Vector_baseIiSaIiEE12_Vector_implC2Ev
	.type	_ZNSt12_Vector_baseIiSaIiEE12_Vector_implC2Ev, @function
_ZNSt12_Vector_baseIiSaIiEE12_Vector_implC2Ev:
.LFB951:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSaIiEC2Ev
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIiSaIiEE17_Vector_impl_dataC2Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE951:
	.size	_ZNSt12_Vector_baseIiSaIiEE12_Vector_implC2Ev, .-_ZNSt12_Vector_baseIiSaIiEE12_Vector_implC2Ev
	.weak	_ZNSt12_Vector_baseIiSaIiEE12_Vector_implC1Ev
	.set	_ZNSt12_Vector_baseIiSaIiEE12_Vector_implC1Ev,_ZNSt12_Vector_baseIiSaIiEE12_Vector_implC2Ev
	.section	.text._ZNSaIiED2Ev,"axG",@progbits,_ZNSaIiED5Ev,comdat
	.align 2
	.weak	_ZNSaIiED2Ev
	.type	_ZNSaIiED2Ev, @function
_ZNSaIiED2Ev:
.LFB954:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZN9__gnu_cxx13new_allocatorIiED2Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE954:
	.size	_ZNSaIiED2Ev, .-_ZNSaIiED2Ev
	.weak	_ZNSaIiED1Ev
	.set	_ZNSaIiED1Ev,_ZNSaIiED2Ev
	.section	.text._ZNSt12_Vector_baseIiSaIiEED2Ev,"axG",@progbits,_ZNSt12_Vector_baseIiSaIiEED5Ev,comdat
	.align 2
	.weak	_ZNSt12_Vector_baseIiSaIiEED2Ev
	.type	_ZNSt12_Vector_baseIiSaIiEED2Ev, @function
_ZNSt12_Vector_baseIiSaIiEED2Ev:
.LFB957:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA957
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	16(%rax), %rdx
	movq	-8(%rbp), %rax
	movq	(%rax), %rax
	subq	%rax, %rdx
	movq	%rdx, %rax
	sarq	$2, %rax
	movq	%rax, %rdx
	movq	-8(%rbp), %rax
	movq	(%rax), %rcx
	movq	-8(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIiSaIiEE13_M_deallocateEPim
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIiSaIiEE12_Vector_implD1Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE957:
	.section	.gcc_except_table
.LLSDA957:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE957-.LLSDACSB957
.LLSDACSB957:
.LLSDACSE957:
	.section	.text._ZNSt12_Vector_baseIiSaIiEED2Ev,"axG",@progbits,_ZNSt12_Vector_baseIiSaIiEED5Ev,comdat
	.size	_ZNSt12_Vector_baseIiSaIiEED2Ev, .-_ZNSt12_Vector_baseIiSaIiEED2Ev
	.weak	_ZNSt12_Vector_baseIiSaIiEED1Ev
	.set	_ZNSt12_Vector_baseIiSaIiEED1Ev,_ZNSt12_Vector_baseIiSaIiEED2Ev
	.section	.text._ZNSt6vectorIiSaIiEED2Ev,"axG",@progbits,_ZNSt6vectorIiSaIiEED5Ev,comdat
	.align 2
	.weak	_ZNSt6vectorIiSaIiEED2Ev
	.type	_ZNSt6vectorIiSaIiEED2Ev, @function
_ZNSt6vectorIiSaIiEED2Ev:
.LFB960:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA960
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIiSaIiEE19_M_get_Tp_allocatorEv
	movq	%rax, %rdx
	movq	-8(%rbp), %rax
	movq	8(%rax), %rcx
	movq	-8(%rbp), %rax
	movq	(%rax), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZSt8_DestroyIPiiEvT_S1_RSaIT0_E
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIiSaIiEED2Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE960:
	.section	.gcc_except_table
.LLSDA960:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE960-.LLSDACSB960
.LLSDACSB960:
.LLSDACSE960:
	.section	.text._ZNSt6vectorIiSaIiEED2Ev,"axG",@progbits,_ZNSt6vectorIiSaIiEED5Ev,comdat
	.size	_ZNSt6vectorIiSaIiEED2Ev, .-_ZNSt6vectorIiSaIiEED2Ev
	.weak	_ZNSt6vectorIiSaIiEED1Ev
	.set	_ZNSt6vectorIiSaIiEED1Ev,_ZNSt6vectorIiSaIiEED2Ev
	.section	.text._ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implC2Ev,"axG",@progbits,_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implC5Ev,comdat
	.align 2
	.weak	_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implC2Ev
	.type	_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implC2Ev, @function
_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implC2Ev:
.LFB963:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSaIPfEC2Ev
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIPfSaIS0_EE17_Vector_impl_dataC2Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE963:
	.size	_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implC2Ev, .-_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implC2Ev
	.weak	_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implC1Ev
	.set	_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implC1Ev,_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implC2Ev
	.section	.text._ZNSaIPfED2Ev,"axG",@progbits,_ZNSaIPfED5Ev,comdat
	.align 2
	.weak	_ZNSaIPfED2Ev
	.type	_ZNSaIPfED2Ev, @function
_ZNSaIPfED2Ev:
.LFB966:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZN9__gnu_cxx13new_allocatorIPfED2Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE966:
	.size	_ZNSaIPfED2Ev, .-_ZNSaIPfED2Ev
	.weak	_ZNSaIPfED1Ev
	.set	_ZNSaIPfED1Ev,_ZNSaIPfED2Ev
	.section	.text._ZNSt12_Vector_baseIPfSaIS0_EED2Ev,"axG",@progbits,_ZNSt12_Vector_baseIPfSaIS0_EED5Ev,comdat
	.align 2
	.weak	_ZNSt12_Vector_baseIPfSaIS0_EED2Ev
	.type	_ZNSt12_Vector_baseIPfSaIS0_EED2Ev, @function
_ZNSt12_Vector_baseIPfSaIS0_EED2Ev:
.LFB969:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA969
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	16(%rax), %rdx
	movq	-8(%rbp), %rax
	movq	(%rax), %rax
	subq	%rax, %rdx
	movq	%rdx, %rax
	sarq	$3, %rax
	movq	%rax, %rdx
	movq	-8(%rbp), %rax
	movq	(%rax), %rcx
	movq	-8(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIPfSaIS0_EE13_M_deallocateEPS0_m
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implD1Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE969:
	.section	.gcc_except_table
.LLSDA969:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE969-.LLSDACSB969
.LLSDACSB969:
.LLSDACSE969:
	.section	.text._ZNSt12_Vector_baseIPfSaIS0_EED2Ev,"axG",@progbits,_ZNSt12_Vector_baseIPfSaIS0_EED5Ev,comdat
	.size	_ZNSt12_Vector_baseIPfSaIS0_EED2Ev, .-_ZNSt12_Vector_baseIPfSaIS0_EED2Ev
	.weak	_ZNSt12_Vector_baseIPfSaIS0_EED1Ev
	.set	_ZNSt12_Vector_baseIPfSaIS0_EED1Ev,_ZNSt12_Vector_baseIPfSaIS0_EED2Ev
	.section	.text._ZNSt6vectorIPfSaIS0_EED2Ev,"axG",@progbits,_ZNSt6vectorIPfSaIS0_EED5Ev,comdat
	.align 2
	.weak	_ZNSt6vectorIPfSaIS0_EED2Ev
	.type	_ZNSt6vectorIPfSaIS0_EED2Ev, @function
_ZNSt6vectorIPfSaIS0_EED2Ev:
.LFB972:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA972
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIPfSaIS0_EE19_M_get_Tp_allocatorEv
	movq	%rax, %rdx
	movq	-8(%rbp), %rax
	movq	8(%rax), %rcx
	movq	-8(%rbp), %rax
	movq	(%rax), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZSt8_DestroyIPPfS0_EvT_S2_RSaIT0_E
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIPfSaIS0_EED2Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE972:
	.section	.gcc_except_table
.LLSDA972:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE972-.LLSDACSB972
.LLSDACSB972:
.LLSDACSE972:
	.section	.text._ZNSt6vectorIPfSaIS0_EED2Ev,"axG",@progbits,_ZNSt6vectorIPfSaIS0_EED5Ev,comdat
	.size	_ZNSt6vectorIPfSaIS0_EED2Ev, .-_ZNSt6vectorIPfSaIS0_EED2Ev
	.weak	_ZNSt6vectorIPfSaIS0_EED1Ev
	.set	_ZNSt6vectorIPfSaIS0_EED1Ev,_ZNSt6vectorIPfSaIS0_EED2Ev
	.section	.text._ZNSt6vectorIS_IPfSaIS0_EESaIS2_EEaSESt16initializer_listIS2_E,"axG",@progbits,_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EEaSESt16initializer_listIS2_E,comdat
	.align 2
	.weak	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EEaSESt16initializer_listIS2_E
	.type	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EEaSESt16initializer_listIS2_E, @function
_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EEaSESt16initializer_listIS2_E:
.LFB974:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$56, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -40(%rbp)
	movq	%rdx, %rcx
	movq	%rsi, %rax
	movq	%rdi, %rdx
	movq	%rcx, %rdx
	movq	%rax, -64(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	leaq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt16initializer_listISt6vectorIPfSaIS1_EEE3endEv
	movq	%rax, %rbx
	leaq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt16initializer_listISt6vectorIPfSaIS1_EEE5beginEv
	movq	%rax, %rcx
	movq	-40(%rbp), %rax
	movq	%rbx, %rdx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE13_M_assign_auxIPKS2_EEvT_S8_St20forward_iterator_tag
	movq	-40(%rbp), %rax
	movq	-24(%rbp), %rdx
	xorq	%fs:40, %rdx
	je	.L55
	call	__stack_chk_fail@PLT
.L55:
	addq	$56, %rsp
	popq	%rbx
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE974:
	.size	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EEaSESt16initializer_listIS2_E, .-_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EEaSESt16initializer_listIS2_E
	.section	.text._ZNSaISt6vectorIPfSaIS0_EEEC2Ev,"axG",@progbits,_ZNSaISt6vectorIPfSaIS0_EEEC5Ev,comdat
	.align 2
	.weak	_ZNSaISt6vectorIPfSaIS0_EEEC2Ev
	.type	_ZNSaISt6vectorIPfSaIS0_EEEC2Ev, @function
_ZNSaISt6vectorIPfSaIS0_EEEC2Ev:
.LFB982:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEEC2Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE982:
	.size	_ZNSaISt6vectorIPfSaIS0_EEEC2Ev, .-_ZNSaISt6vectorIPfSaIS0_EEEC2Ev
	.weak	_ZNSaISt6vectorIPfSaIS0_EEEC1Ev
	.set	_ZNSaISt6vectorIPfSaIS0_EEEC1Ev,_ZNSaISt6vectorIPfSaIS0_EEEC2Ev
	.section	.text._ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE17_Vector_impl_dataC2Ev,"axG",@progbits,_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE17_Vector_impl_dataC5Ev,comdat
	.align 2
	.weak	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE17_Vector_impl_dataC2Ev
	.type	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE17_Vector_impl_dataC2Ev, @function
_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE17_Vector_impl_dataC2Ev:
.LFB985:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	$0, (%rax)
	movq	-8(%rbp), %rax
	movq	$0, 8(%rax)
	movq	-8(%rbp), %rax
	movq	$0, 16(%rax)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE985:
	.size	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE17_Vector_impl_dataC2Ev, .-_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE17_Vector_impl_dataC2Ev
	.weak	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE17_Vector_impl_dataC1Ev
	.set	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE17_Vector_impl_dataC1Ev,_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE17_Vector_impl_dataC2Ev
	.section	.text._ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEED2Ev,"axG",@progbits,_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEED5Ev,comdat
	.align 2
	.weak	_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEED2Ev
	.type	_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEED2Ev, @function
_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEED2Ev:
.LFB988:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE988:
	.size	_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEED2Ev, .-_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEED2Ev
	.weak	_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEED1Ev
	.set	_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEED1Ev,_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEED2Ev
	.section	.text._ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE13_M_deallocateEPS3_m,"axG",@progbits,_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE13_M_deallocateEPS3_m,comdat
	.align 2
	.weak	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE13_M_deallocateEPS3_m
	.type	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE13_M_deallocateEPS3_m, @function
_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE13_M_deallocateEPS3_m:
.LFB990:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	cmpq	$0, -16(%rbp)
	je	.L61
	movq	-8(%rbp), %rax
	movq	-24(%rbp), %rdx
	movq	-16(%rbp), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZNSt16allocator_traitsISaISt6vectorIPfSaIS1_EEEE10deallocateERS4_PS3_m
.L61:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE990:
	.size	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE13_M_deallocateEPS3_m, .-_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE13_M_deallocateEPS3_m
	.section	.text._ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE19_M_get_Tp_allocatorEv,"axG",@progbits,_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE19_M_get_Tp_allocatorEv,comdat
	.align 2
	.weak	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE19_M_get_Tp_allocatorEv
	.type	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE19_M_get_Tp_allocatorEv, @function
_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE19_M_get_Tp_allocatorEv:
.LFB991:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE991:
	.size	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE19_M_get_Tp_allocatorEv, .-_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE19_M_get_Tp_allocatorEv
	.section	.text._ZSt8_DestroyIPSt6vectorIPfSaIS1_EES3_EvT_S5_RSaIT0_E,"axG",@progbits,_ZSt8_DestroyIPSt6vectorIPfSaIS1_EES3_EvT_S5_RSaIT0_E,comdat
	.weak	_ZSt8_DestroyIPSt6vectorIPfSaIS1_EES3_EvT_S5_RSaIT0_E
	.type	_ZSt8_DestroyIPSt6vectorIPfSaIS1_EES3_EvT_S5_RSaIT0_E, @function
_ZSt8_DestroyIPSt6vectorIPfSaIS1_EES3_EvT_S5_RSaIT0_E:
.LFB992:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-16(%rbp), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZSt8_DestroyIPSt6vectorIPfSaIS1_EEEvT_S5_
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE992:
	.size	_ZSt8_DestroyIPSt6vectorIPfSaIS1_EES3_EvT_S5_RSaIT0_E, .-_ZSt8_DestroyIPSt6vectorIPfSaIS1_EES3_EvT_S5_RSaIT0_E
	.section	.text._ZNSaIiEC2Ev,"axG",@progbits,_ZNSaIiEC5Ev,comdat
	.align 2
	.weak	_ZNSaIiEC2Ev
	.type	_ZNSaIiEC2Ev, @function
_ZNSaIiEC2Ev:
.LFB994:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZN9__gnu_cxx13new_allocatorIiEC2Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE994:
	.size	_ZNSaIiEC2Ev, .-_ZNSaIiEC2Ev
	.weak	_ZNSaIiEC1Ev
	.set	_ZNSaIiEC1Ev,_ZNSaIiEC2Ev
	.section	.text._ZNSt12_Vector_baseIiSaIiEE17_Vector_impl_dataC2Ev,"axG",@progbits,_ZNSt12_Vector_baseIiSaIiEE17_Vector_impl_dataC5Ev,comdat
	.align 2
	.weak	_ZNSt12_Vector_baseIiSaIiEE17_Vector_impl_dataC2Ev
	.type	_ZNSt12_Vector_baseIiSaIiEE17_Vector_impl_dataC2Ev, @function
_ZNSt12_Vector_baseIiSaIiEE17_Vector_impl_dataC2Ev:
.LFB997:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	$0, (%rax)
	movq	-8(%rbp), %rax
	movq	$0, 8(%rax)
	movq	-8(%rbp), %rax
	movq	$0, 16(%rax)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE997:
	.size	_ZNSt12_Vector_baseIiSaIiEE17_Vector_impl_dataC2Ev, .-_ZNSt12_Vector_baseIiSaIiEE17_Vector_impl_dataC2Ev
	.weak	_ZNSt12_Vector_baseIiSaIiEE17_Vector_impl_dataC1Ev
	.set	_ZNSt12_Vector_baseIiSaIiEE17_Vector_impl_dataC1Ev,_ZNSt12_Vector_baseIiSaIiEE17_Vector_impl_dataC2Ev
	.section	.text._ZN9__gnu_cxx13new_allocatorIiED2Ev,"axG",@progbits,_ZN9__gnu_cxx13new_allocatorIiED5Ev,comdat
	.align 2
	.weak	_ZN9__gnu_cxx13new_allocatorIiED2Ev
	.type	_ZN9__gnu_cxx13new_allocatorIiED2Ev, @function
_ZN9__gnu_cxx13new_allocatorIiED2Ev:
.LFB1000:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1000:
	.size	_ZN9__gnu_cxx13new_allocatorIiED2Ev, .-_ZN9__gnu_cxx13new_allocatorIiED2Ev
	.weak	_ZN9__gnu_cxx13new_allocatorIiED1Ev
	.set	_ZN9__gnu_cxx13new_allocatorIiED1Ev,_ZN9__gnu_cxx13new_allocatorIiED2Ev
	.section	.text._ZNSt12_Vector_baseIiSaIiEE13_M_deallocateEPim,"axG",@progbits,_ZNSt12_Vector_baseIiSaIiEE13_M_deallocateEPim,comdat
	.align 2
	.weak	_ZNSt12_Vector_baseIiSaIiEE13_M_deallocateEPim
	.type	_ZNSt12_Vector_baseIiSaIiEE13_M_deallocateEPim, @function
_ZNSt12_Vector_baseIiSaIiEE13_M_deallocateEPim:
.LFB1002:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	cmpq	$0, -16(%rbp)
	je	.L70
	movq	-8(%rbp), %rax
	movq	-24(%rbp), %rdx
	movq	-16(%rbp), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZNSt16allocator_traitsISaIiEE10deallocateERS0_Pim
.L70:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1002:
	.size	_ZNSt12_Vector_baseIiSaIiEE13_M_deallocateEPim, .-_ZNSt12_Vector_baseIiSaIiEE13_M_deallocateEPim
	.section	.text._ZNSt12_Vector_baseIiSaIiEE19_M_get_Tp_allocatorEv,"axG",@progbits,_ZNSt12_Vector_baseIiSaIiEE19_M_get_Tp_allocatorEv,comdat
	.align 2
	.weak	_ZNSt12_Vector_baseIiSaIiEE19_M_get_Tp_allocatorEv
	.type	_ZNSt12_Vector_baseIiSaIiEE19_M_get_Tp_allocatorEv, @function
_ZNSt12_Vector_baseIiSaIiEE19_M_get_Tp_allocatorEv:
.LFB1003:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1003:
	.size	_ZNSt12_Vector_baseIiSaIiEE19_M_get_Tp_allocatorEv, .-_ZNSt12_Vector_baseIiSaIiEE19_M_get_Tp_allocatorEv
	.section	.text._ZSt8_DestroyIPiiEvT_S1_RSaIT0_E,"axG",@progbits,_ZSt8_DestroyIPiiEvT_S1_RSaIT0_E,comdat
	.weak	_ZSt8_DestroyIPiiEvT_S1_RSaIT0_E
	.type	_ZSt8_DestroyIPiiEvT_S1_RSaIT0_E, @function
_ZSt8_DestroyIPiiEvT_S1_RSaIT0_E:
.LFB1004:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-16(%rbp), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZSt8_DestroyIPiEvT_S1_
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1004:
	.size	_ZSt8_DestroyIPiiEvT_S1_RSaIT0_E, .-_ZSt8_DestroyIPiiEvT_S1_RSaIT0_E
	.section	.text._ZNSaIPfEC2Ev,"axG",@progbits,_ZNSaIPfEC5Ev,comdat
	.align 2
	.weak	_ZNSaIPfEC2Ev
	.type	_ZNSaIPfEC2Ev, @function
_ZNSaIPfEC2Ev:
.LFB1006:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZN9__gnu_cxx13new_allocatorIPfEC2Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1006:
	.size	_ZNSaIPfEC2Ev, .-_ZNSaIPfEC2Ev
	.weak	_ZNSaIPfEC1Ev
	.set	_ZNSaIPfEC1Ev,_ZNSaIPfEC2Ev
	.section	.text._ZNSt12_Vector_baseIPfSaIS0_EE17_Vector_impl_dataC2Ev,"axG",@progbits,_ZNSt12_Vector_baseIPfSaIS0_EE17_Vector_impl_dataC5Ev,comdat
	.align 2
	.weak	_ZNSt12_Vector_baseIPfSaIS0_EE17_Vector_impl_dataC2Ev
	.type	_ZNSt12_Vector_baseIPfSaIS0_EE17_Vector_impl_dataC2Ev, @function
_ZNSt12_Vector_baseIPfSaIS0_EE17_Vector_impl_dataC2Ev:
.LFB1009:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	$0, (%rax)
	movq	-8(%rbp), %rax
	movq	$0, 8(%rax)
	movq	-8(%rbp), %rax
	movq	$0, 16(%rax)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1009:
	.size	_ZNSt12_Vector_baseIPfSaIS0_EE17_Vector_impl_dataC2Ev, .-_ZNSt12_Vector_baseIPfSaIS0_EE17_Vector_impl_dataC2Ev
	.weak	_ZNSt12_Vector_baseIPfSaIS0_EE17_Vector_impl_dataC1Ev
	.set	_ZNSt12_Vector_baseIPfSaIS0_EE17_Vector_impl_dataC1Ev,_ZNSt12_Vector_baseIPfSaIS0_EE17_Vector_impl_dataC2Ev
	.section	.text._ZN9__gnu_cxx13new_allocatorIPfED2Ev,"axG",@progbits,_ZN9__gnu_cxx13new_allocatorIPfED5Ev,comdat
	.align 2
	.weak	_ZN9__gnu_cxx13new_allocatorIPfED2Ev
	.type	_ZN9__gnu_cxx13new_allocatorIPfED2Ev, @function
_ZN9__gnu_cxx13new_allocatorIPfED2Ev:
.LFB1012:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1012:
	.size	_ZN9__gnu_cxx13new_allocatorIPfED2Ev, .-_ZN9__gnu_cxx13new_allocatorIPfED2Ev
	.weak	_ZN9__gnu_cxx13new_allocatorIPfED1Ev
	.set	_ZN9__gnu_cxx13new_allocatorIPfED1Ev,_ZN9__gnu_cxx13new_allocatorIPfED2Ev
	.section	.text._ZNSt12_Vector_baseIPfSaIS0_EE13_M_deallocateEPS0_m,"axG",@progbits,_ZNSt12_Vector_baseIPfSaIS0_EE13_M_deallocateEPS0_m,comdat
	.align 2
	.weak	_ZNSt12_Vector_baseIPfSaIS0_EE13_M_deallocateEPS0_m
	.type	_ZNSt12_Vector_baseIPfSaIS0_EE13_M_deallocateEPS0_m, @function
_ZNSt12_Vector_baseIPfSaIS0_EE13_M_deallocateEPS0_m:
.LFB1014:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	cmpq	$0, -16(%rbp)
	je	.L79
	movq	-8(%rbp), %rax
	movq	-24(%rbp), %rdx
	movq	-16(%rbp), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZNSt16allocator_traitsISaIPfEE10deallocateERS1_PS0_m
.L79:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1014:
	.size	_ZNSt12_Vector_baseIPfSaIS0_EE13_M_deallocateEPS0_m, .-_ZNSt12_Vector_baseIPfSaIS0_EE13_M_deallocateEPS0_m
	.section	.text._ZNSt12_Vector_baseIPfSaIS0_EE19_M_get_Tp_allocatorEv,"axG",@progbits,_ZNSt12_Vector_baseIPfSaIS0_EE19_M_get_Tp_allocatorEv,comdat
	.align 2
	.weak	_ZNSt12_Vector_baseIPfSaIS0_EE19_M_get_Tp_allocatorEv
	.type	_ZNSt12_Vector_baseIPfSaIS0_EE19_M_get_Tp_allocatorEv, @function
_ZNSt12_Vector_baseIPfSaIS0_EE19_M_get_Tp_allocatorEv:
.LFB1015:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1015:
	.size	_ZNSt12_Vector_baseIPfSaIS0_EE19_M_get_Tp_allocatorEv, .-_ZNSt12_Vector_baseIPfSaIS0_EE19_M_get_Tp_allocatorEv
	.section	.text._ZSt8_DestroyIPPfS0_EvT_S2_RSaIT0_E,"axG",@progbits,_ZSt8_DestroyIPPfS0_EvT_S2_RSaIT0_E,comdat
	.weak	_ZSt8_DestroyIPPfS0_EvT_S2_RSaIT0_E
	.type	_ZSt8_DestroyIPPfS0_EvT_S2_RSaIT0_E, @function
_ZSt8_DestroyIPPfS0_EvT_S2_RSaIT0_E:
.LFB1016:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-16(%rbp), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZSt8_DestroyIPPfEvT_S2_
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1016:
	.size	_ZSt8_DestroyIPPfS0_EvT_S2_RSaIT0_E, .-_ZSt8_DestroyIPPfS0_EvT_S2_RSaIT0_E
	.section	.text._ZNKSt16initializer_listISt6vectorIPfSaIS1_EEE5beginEv,"axG",@progbits,_ZNKSt16initializer_listISt6vectorIPfSaIS1_EEE5beginEv,comdat
	.align 2
	.weak	_ZNKSt16initializer_listISt6vectorIPfSaIS1_EEE5beginEv
	.type	_ZNKSt16initializer_listISt6vectorIPfSaIS1_EEE5beginEv, @function
_ZNKSt16initializer_listISt6vectorIPfSaIS1_EEE5beginEv:
.LFB1017:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	(%rax), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1017:
	.size	_ZNKSt16initializer_listISt6vectorIPfSaIS1_EEE5beginEv, .-_ZNKSt16initializer_listISt6vectorIPfSaIS1_EEE5beginEv
	.section	.text._ZNKSt16initializer_listISt6vectorIPfSaIS1_EEE3endEv,"axG",@progbits,_ZNKSt16initializer_listISt6vectorIPfSaIS1_EEE3endEv,comdat
	.align 2
	.weak	_ZNKSt16initializer_listISt6vectorIPfSaIS1_EEE3endEv
	.type	_ZNKSt16initializer_listISt6vectorIPfSaIS1_EEE3endEv, @function
_ZNKSt16initializer_listISt6vectorIPfSaIS1_EEE3endEv:
.LFB1018:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$24, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt16initializer_listISt6vectorIPfSaIS1_EEE5beginEv
	movq	%rax, %rbx
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt16initializer_listISt6vectorIPfSaIS1_EEE4sizeEv
	movq	%rax, %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	addq	%rbx, %rax
	addq	$24, %rsp
	popq	%rbx
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1018:
	.size	_ZNKSt16initializer_listISt6vectorIPfSaIS1_EEE3endEv, .-_ZNKSt16initializer_listISt6vectorIPfSaIS1_EEE3endEv
	.section	.text._ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE13_M_assign_auxIPKS2_EEvT_S8_St20forward_iterator_tag,"axG",@progbits,_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE13_M_assign_auxIPKS2_EEvT_S8_St20forward_iterator_tag,comdat
	.align 2
	.weak	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE13_M_assign_auxIPKS2_EEvT_S8_St20forward_iterator_tag
	.type	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE13_M_assign_auxIPKS2_EEvT_S8_St20forward_iterator_tag, @function
_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE13_M_assign_auxIPKS2_EEvT_S8_St20forward_iterator_tag:
.LFB1019:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$80, %rsp
	movq	%rdi, -56(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%rdx, -72(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	-72(%rbp), %rdx
	movq	-64(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZSt8distanceIPKSt6vectorIPfSaIS1_EEENSt15iterator_traitsIT_E15difference_typeES7_S7_
	movq	%rax, -32(%rbp)
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt6vectorIS_IPfSaIS0_EESaIS2_EE8capacityEv
	cmpq	%rax, -32(%rbp)
	seta	%al
	testb	%al, %al
	je	.L88
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE19_M_get_Tp_allocatorEv
	movq	%rax, %rdx
	movq	-32(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE17_S_check_init_lenEmRKS3_
	movq	-72(%rbp), %rcx
	movq	-64(%rbp), %rdx
	movq	-32(%rbp), %rsi
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE20_M_allocate_and_copyIPKS2_EEPS2_mT_S9_
	movq	%rax, -16(%rbp)
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE19_M_get_Tp_allocatorEv
	movq	%rax, %rdx
	movq	-56(%rbp), %rax
	movq	8(%rax), %rcx
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZSt8_DestroyIPSt6vectorIPfSaIS1_EES3_EvT_S5_RSaIT0_E
	movq	-56(%rbp), %rax
	movq	-56(%rbp), %rdx
	movq	16(%rdx), %rcx
	movq	-56(%rbp), %rdx
	movq	(%rdx), %rdx
	subq	%rdx, %rcx
	movq	%rcx, %rdx
	movq	%rdx, %rcx
	sarq	$3, %rcx
	movabsq	$-6148914691236517205, %rdx
	imulq	%rcx, %rdx
	movq	%rdx, %rsi
	movq	-56(%rbp), %rdx
	movq	(%rdx), %rcx
	movq	%rsi, %rdx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE13_M_deallocateEPS3_m
	movq	-56(%rbp), %rax
	movq	-16(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	-56(%rbp), %rax
	movq	(%rax), %rcx
	movq	-32(%rbp), %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	leaq	(%rcx,%rax), %rdx
	movq	-56(%rbp), %rax
	movq	%rdx, 8(%rax)
	movq	-56(%rbp), %rax
	movq	8(%rax), %rdx
	movq	-56(%rbp), %rax
	movq	%rdx, 16(%rax)
	jmp	.L92
.L88:
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt6vectorIS_IPfSaIS0_EESaIS2_EE4sizeEv
	cmpq	%rax, -32(%rbp)
	setbe	%al
	testb	%al, %al
	je	.L90
	movq	-56(%rbp), %rax
	movq	(%rax), %rdx
	movq	-72(%rbp), %rcx
	movq	-64(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZSt4copyIPKSt6vectorIPfSaIS1_EEPS3_ET0_T_S8_S7_
	movq	%rax, %rdx
	movq	-56(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE15_M_erase_at_endEPS2_
	jmp	.L92
.L90:
	movq	-64(%rbp), %rax
	movq	%rax, -40(%rbp)
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt6vectorIS_IPfSaIS0_EESaIS2_EE4sizeEv
	movq	%rax, %rdx
	leaq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZSt7advanceIPKSt6vectorIPfSaIS1_EEmEvRT_T0_
	movq	-56(%rbp), %rax
	movq	(%rax), %rdx
	movq	-40(%rbp), %rcx
	movq	-64(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZSt4copyIPKSt6vectorIPfSaIS1_EEPS3_ET0_T_S8_S7_
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt6vectorIS_IPfSaIS0_EESaIS2_EE4sizeEv
	movq	-32(%rbp), %rdx
	subq	%rax, %rdx
	movq	%rdx, %rax
	movq	%rax, -24(%rbp)
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE19_M_get_Tp_allocatorEv
	movq	%rax, %rcx
	movq	-56(%rbp), %rax
	movq	8(%rax), %rdx
	movq	-40(%rbp), %rax
	movq	-72(%rbp), %rsi
	movq	%rax, %rdi
	call	_ZSt22__uninitialized_copy_aIPKSt6vectorIPfSaIS1_EEPS3_S3_ET0_T_S8_S7_RSaIT1_E
	movq	-56(%rbp), %rdx
	movq	%rax, 8(%rdx)
.L92:
	nop
	movq	-8(%rbp), %rax
	xorq	%fs:40, %rax
	je	.L91
	call	__stack_chk_fail@PLT
.L91:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1019:
	.size	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE13_M_assign_auxIPKS2_EEvT_S8_St20forward_iterator_tag, .-_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE13_M_assign_auxIPKS2_EEvT_S8_St20forward_iterator_tag
	.section	.text._ZNSaISt6vectorIPfSaIS0_EEEC2ERKS3_,"axG",@progbits,_ZNSaISt6vectorIPfSaIS0_EEEC5ERKS3_,comdat
	.align 2
	.weak	_ZNSaISt6vectorIPfSaIS0_EEEC2ERKS3_
	.type	_ZNSaISt6vectorIPfSaIS0_EEEC2ERKS3_, @function
_ZNSaISt6vectorIPfSaIS0_EEEC2ERKS3_:
.LFB1022:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEEC2ERKS5_
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1022:
	.size	_ZNSaISt6vectorIPfSaIS0_EEEC2ERKS3_, .-_ZNSaISt6vectorIPfSaIS0_EEEC2ERKS3_
	.weak	_ZNSaISt6vectorIPfSaIS0_EEEC1ERKS3_
	.set	_ZNSaISt6vectorIPfSaIS0_EEEC1ERKS3_,_ZNSaISt6vectorIPfSaIS0_EEEC2ERKS3_
	.section	.text._ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEEC2Ev,"axG",@progbits,_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEEC5Ev,comdat
	.align 2
	.weak	_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEEC2Ev
	.type	_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEEC2Ev, @function
_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEEC2Ev:
.LFB1035:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1035:
	.size	_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEEC2Ev, .-_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEEC2Ev
	.weak	_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEEC1Ev
	.set	_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEEC1Ev,_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEEC2Ev
	.section	.text._ZNSt16allocator_traitsISaISt6vectorIPfSaIS1_EEEE10deallocateERS4_PS3_m,"axG",@progbits,_ZNSt16allocator_traitsISaISt6vectorIPfSaIS1_EEEE10deallocateERS4_PS3_m,comdat
	.weak	_ZNSt16allocator_traitsISaISt6vectorIPfSaIS1_EEEE10deallocateERS4_PS3_m
	.type	_ZNSt16allocator_traitsISaISt6vectorIPfSaIS1_EEEE10deallocateERS4_PS3_m, @function
_ZNSt16allocator_traitsISaISt6vectorIPfSaIS1_EEEE10deallocateERS4_PS3_m:
.LFB1037:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-24(%rbp), %rdx
	movq	-16(%rbp), %rcx
	movq	-8(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEE10deallocateEPS4_m
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1037:
	.size	_ZNSt16allocator_traitsISaISt6vectorIPfSaIS1_EEEE10deallocateERS4_PS3_m, .-_ZNSt16allocator_traitsISaISt6vectorIPfSaIS1_EEEE10deallocateERS4_PS3_m
	.section	.text._ZSt8_DestroyIPSt6vectorIPfSaIS1_EEEvT_S5_,"axG",@progbits,_ZSt8_DestroyIPSt6vectorIPfSaIS1_EEEvT_S5_,comdat
	.weak	_ZSt8_DestroyIPSt6vectorIPfSaIS1_EEEvT_S5_
	.type	_ZSt8_DestroyIPSt6vectorIPfSaIS1_EEEvT_S5_, @function
_ZSt8_DestroyIPSt6vectorIPfSaIS1_EEEvT_S5_:
.LFB1038:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSt12_Destroy_auxILb0EE9__destroyIPSt6vectorIPfSaIS3_EEEEvT_S7_
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1038:
	.size	_ZSt8_DestroyIPSt6vectorIPfSaIS1_EEEvT_S5_, .-_ZSt8_DestroyIPSt6vectorIPfSaIS1_EEEvT_S5_
	.section	.text._ZN9__gnu_cxx13new_allocatorIiEC2Ev,"axG",@progbits,_ZN9__gnu_cxx13new_allocatorIiEC5Ev,comdat
	.align 2
	.weak	_ZN9__gnu_cxx13new_allocatorIiEC2Ev
	.type	_ZN9__gnu_cxx13new_allocatorIiEC2Ev, @function
_ZN9__gnu_cxx13new_allocatorIiEC2Ev:
.LFB1040:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1040:
	.size	_ZN9__gnu_cxx13new_allocatorIiEC2Ev, .-_ZN9__gnu_cxx13new_allocatorIiEC2Ev
	.weak	_ZN9__gnu_cxx13new_allocatorIiEC1Ev
	.set	_ZN9__gnu_cxx13new_allocatorIiEC1Ev,_ZN9__gnu_cxx13new_allocatorIiEC2Ev
	.section	.text._ZNSt16allocator_traitsISaIiEE10deallocateERS0_Pim,"axG",@progbits,_ZNSt16allocator_traitsISaIiEE10deallocateERS0_Pim,comdat
	.weak	_ZNSt16allocator_traitsISaIiEE10deallocateERS0_Pim
	.type	_ZNSt16allocator_traitsISaIiEE10deallocateERS0_Pim, @function
_ZNSt16allocator_traitsISaIiEE10deallocateERS0_Pim:
.LFB1042:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-24(%rbp), %rdx
	movq	-16(%rbp), %rcx
	movq	-8(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZN9__gnu_cxx13new_allocatorIiE10deallocateEPim
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1042:
	.size	_ZNSt16allocator_traitsISaIiEE10deallocateERS0_Pim, .-_ZNSt16allocator_traitsISaIiEE10deallocateERS0_Pim
	.section	.text._ZSt8_DestroyIPiEvT_S1_,"axG",@progbits,_ZSt8_DestroyIPiEvT_S1_,comdat
	.weak	_ZSt8_DestroyIPiEvT_S1_
	.type	_ZSt8_DestroyIPiEvT_S1_, @function
_ZSt8_DestroyIPiEvT_S1_:
.LFB1043:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSt12_Destroy_auxILb1EE9__destroyIPiEEvT_S3_
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1043:
	.size	_ZSt8_DestroyIPiEvT_S1_, .-_ZSt8_DestroyIPiEvT_S1_
	.section	.text._ZN9__gnu_cxx13new_allocatorIPfEC2Ev,"axG",@progbits,_ZN9__gnu_cxx13new_allocatorIPfEC5Ev,comdat
	.align 2
	.weak	_ZN9__gnu_cxx13new_allocatorIPfEC2Ev
	.type	_ZN9__gnu_cxx13new_allocatorIPfEC2Ev, @function
_ZN9__gnu_cxx13new_allocatorIPfEC2Ev:
.LFB1045:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1045:
	.size	_ZN9__gnu_cxx13new_allocatorIPfEC2Ev, .-_ZN9__gnu_cxx13new_allocatorIPfEC2Ev
	.weak	_ZN9__gnu_cxx13new_allocatorIPfEC1Ev
	.set	_ZN9__gnu_cxx13new_allocatorIPfEC1Ev,_ZN9__gnu_cxx13new_allocatorIPfEC2Ev
	.section	.text._ZNSt16allocator_traitsISaIPfEE10deallocateERS1_PS0_m,"axG",@progbits,_ZNSt16allocator_traitsISaIPfEE10deallocateERS1_PS0_m,comdat
	.weak	_ZNSt16allocator_traitsISaIPfEE10deallocateERS1_PS0_m
	.type	_ZNSt16allocator_traitsISaIPfEE10deallocateERS1_PS0_m, @function
_ZNSt16allocator_traitsISaIPfEE10deallocateERS1_PS0_m:
.LFB1047:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-24(%rbp), %rdx
	movq	-16(%rbp), %rcx
	movq	-8(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZN9__gnu_cxx13new_allocatorIPfE10deallocateEPS1_m
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1047:
	.size	_ZNSt16allocator_traitsISaIPfEE10deallocateERS1_PS0_m, .-_ZNSt16allocator_traitsISaIPfEE10deallocateERS1_PS0_m
	.section	.text._ZSt8_DestroyIPPfEvT_S2_,"axG",@progbits,_ZSt8_DestroyIPPfEvT_S2_,comdat
	.weak	_ZSt8_DestroyIPPfEvT_S2_
	.type	_ZSt8_DestroyIPPfEvT_S2_, @function
_ZSt8_DestroyIPPfEvT_S2_:
.LFB1048:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSt12_Destroy_auxILb1EE9__destroyIPPfEEvT_S4_
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1048:
	.size	_ZSt8_DestroyIPPfEvT_S2_, .-_ZSt8_DestroyIPPfEvT_S2_
	.section	.text._ZNKSt16initializer_listISt6vectorIPfSaIS1_EEE4sizeEv,"axG",@progbits,_ZNKSt16initializer_listISt6vectorIPfSaIS1_EEE4sizeEv,comdat
	.align 2
	.weak	_ZNKSt16initializer_listISt6vectorIPfSaIS1_EEE4sizeEv
	.type	_ZNKSt16initializer_listISt6vectorIPfSaIS1_EEE4sizeEv, @function
_ZNKSt16initializer_listISt6vectorIPfSaIS1_EEE4sizeEv:
.LFB1049:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1049:
	.size	_ZNKSt16initializer_listISt6vectorIPfSaIS1_EEE4sizeEv, .-_ZNKSt16initializer_listISt6vectorIPfSaIS1_EEE4sizeEv
	.section	.text._ZSt8distanceIPKSt6vectorIPfSaIS1_EEENSt15iterator_traitsIT_E15difference_typeES7_S7_,"axG",@progbits,_ZSt8distanceIPKSt6vectorIPfSaIS1_EEENSt15iterator_traitsIT_E15difference_typeES7_S7_,comdat
	.weak	_ZSt8distanceIPKSt6vectorIPfSaIS1_EEENSt15iterator_traitsIT_E15difference_typeES7_S7_
	.type	_ZSt8distanceIPKSt6vectorIPfSaIS1_EEENSt15iterator_traitsIT_E15difference_typeES7_S7_, @function
_ZSt8distanceIPKSt6vectorIPfSaIS1_EEENSt15iterator_traitsIT_E15difference_typeES7_S7_:
.LFB1050:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	leaq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt19__iterator_categoryIPKSt6vectorIPfSaIS1_EEENSt15iterator_traitsIT_E17iterator_categoryERKS7_
	movq	-24(%rbp), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZSt10__distanceIPKSt6vectorIPfSaIS1_EEENSt15iterator_traitsIT_E15difference_typeES7_S7_St26random_access_iterator_tag
	movq	-8(%rbp), %rcx
	xorq	%fs:40, %rcx
	je	.L107
	call	__stack_chk_fail@PLT
.L107:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1050:
	.size	_ZSt8distanceIPKSt6vectorIPfSaIS1_EEENSt15iterator_traitsIT_E15difference_typeES7_S7_, .-_ZSt8distanceIPKSt6vectorIPfSaIS1_EEENSt15iterator_traitsIT_E15difference_typeES7_S7_
	.section	.text._ZNKSt6vectorIS_IPfSaIS0_EESaIS2_EE8capacityEv,"axG",@progbits,_ZNKSt6vectorIS_IPfSaIS0_EESaIS2_EE8capacityEv,comdat
	.align 2
	.weak	_ZNKSt6vectorIS_IPfSaIS0_EESaIS2_EE8capacityEv
	.type	_ZNKSt6vectorIS_IPfSaIS0_EESaIS2_EE8capacityEv, @function
_ZNKSt6vectorIS_IPfSaIS0_EESaIS2_EE8capacityEv:
.LFB1051:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	16(%rax), %rdx
	movq	-8(%rbp), %rax
	movq	(%rax), %rax
	subq	%rax, %rdx
	movq	%rdx, %rax
	sarq	$3, %rax
	movq	%rax, %rdx
	movabsq	$-6148914691236517205, %rax
	imulq	%rdx, %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1051:
	.size	_ZNKSt6vectorIS_IPfSaIS0_EESaIS2_EE8capacityEv, .-_ZNKSt6vectorIS_IPfSaIS0_EESaIS2_EE8capacityEv
	.section	.rodata
	.align 8
.LC0:
	.string	"cannot create std::vector larger than max_size()"
	.section	.text._ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE17_S_check_init_lenEmRKS3_,"axG",@progbits,_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE17_S_check_init_lenEmRKS3_,comdat
	.weak	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE17_S_check_init_lenEmRKS3_
	.type	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE17_S_check_init_lenEmRKS3_, @function
_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE17_S_check_init_lenEmRKS3_:
.LFB1052:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$40, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -24(%rbp)
	xorl	%eax, %eax
	movq	-48(%rbp), %rdx
	leaq	-25(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSaISt6vectorIPfSaIS0_EEEC1ERKS3_
	leaq	-25(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE11_S_max_sizeERKS3_
	cmpq	%rax, -40(%rbp)
	seta	%bl
	leaq	-25(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSaISt6vectorIPfSaIS0_EEED1Ev
	testb	%bl, %bl
	je	.L111
	leaq	.LC0(%rip), %rdi
	call	_ZSt20__throw_length_errorPKc@PLT
.L111:
	movq	-40(%rbp), %rax
	movq	-24(%rbp), %rcx
	xorq	%fs:40, %rcx
	je	.L113
	call	__stack_chk_fail@PLT
.L113:
	addq	$40, %rsp
	popq	%rbx
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1052:
	.size	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE17_S_check_init_lenEmRKS3_, .-_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE17_S_check_init_lenEmRKS3_
	.section	.text._ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE20_M_allocate_and_copyIPKS2_EEPS2_mT_S9_,"axG",@progbits,_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE20_M_allocate_and_copyIPKS2_EEPS2_mT_S9_,comdat
	.align 2
	.weak	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE20_M_allocate_and_copyIPKS2_EEPS2_mT_S9_
	.type	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE20_M_allocate_and_copyIPKS2_EEPS2_mT_S9_, @function
_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE20_M_allocate_and_copyIPKS2_EEPS2_mT_S9_:
.LFB1053:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA1053
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$56, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%rcx, -64(%rbp)
	movq	-40(%rbp), %rax
	movq	-48(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
.LEHB2:
	call	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE11_M_allocateEm
.LEHE2:
	movq	%rax, -24(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE19_M_get_Tp_allocatorEv
	movq	%rax, %rcx
	movq	-24(%rbp), %rdx
	movq	-64(%rbp), %rsi
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
.LEHB3:
	call	_ZSt22__uninitialized_copy_aIPKSt6vectorIPfSaIS1_EEPS3_S3_ET0_T_S8_S7_RSaIT1_E
.LEHE3:
	movq	-24(%rbp), %rax
	jmp	.L120
.L118:
	endbr64
	movq	%rax, %rdi
	call	__cxa_begin_catch@PLT
	movq	-40(%rbp), %rax
	movq	-48(%rbp), %rdx
	movq	-24(%rbp), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
.LEHB4:
	call	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE13_M_deallocateEPS3_m
	call	__cxa_rethrow@PLT
.LEHE4:
.L119:
	endbr64
	movq	%rax, %rbx
	call	__cxa_end_catch@PLT
	movq	%rbx, %rax
	movq	%rax, %rdi
.LEHB5:
	call	_Unwind_Resume@PLT
.LEHE5:
.L120:
	addq	$56, %rsp
	popq	%rbx
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1053:
	.section	.gcc_except_table
	.align 4
.LLSDA1053:
	.byte	0xff
	.byte	0x9b
	.uleb128 .LLSDATT1053-.LLSDATTD1053
.LLSDATTD1053:
	.byte	0x1
	.uleb128 .LLSDACSE1053-.LLSDACSB1053
.LLSDACSB1053:
	.uleb128 .LEHB2-.LFB1053
	.uleb128 .LEHE2-.LEHB2
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB3-.LFB1053
	.uleb128 .LEHE3-.LEHB3
	.uleb128 .L118-.LFB1053
	.uleb128 0x1
	.uleb128 .LEHB4-.LFB1053
	.uleb128 .LEHE4-.LEHB4
	.uleb128 .L119-.LFB1053
	.uleb128 0
	.uleb128 .LEHB5-.LFB1053
	.uleb128 .LEHE5-.LEHB5
	.uleb128 0
	.uleb128 0
.LLSDACSE1053:
	.byte	0x1
	.byte	0
	.align 4
	.long	0

.LLSDATT1053:
	.section	.text._ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE20_M_allocate_and_copyIPKS2_EEPS2_mT_S9_,"axG",@progbits,_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE20_M_allocate_and_copyIPKS2_EEPS2_mT_S9_,comdat
	.size	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE20_M_allocate_and_copyIPKS2_EEPS2_mT_S9_, .-_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE20_M_allocate_and_copyIPKS2_EEPS2_mT_S9_
	.section	.text._ZNKSt6vectorIS_IPfSaIS0_EESaIS2_EE4sizeEv,"axG",@progbits,_ZNKSt6vectorIS_IPfSaIS0_EESaIS2_EE4sizeEv,comdat
	.align 2
	.weak	_ZNKSt6vectorIS_IPfSaIS0_EESaIS2_EE4sizeEv
	.type	_ZNKSt6vectorIS_IPfSaIS0_EESaIS2_EE4sizeEv, @function
_ZNKSt6vectorIS_IPfSaIS0_EESaIS2_EE4sizeEv:
.LFB1054:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdx
	movq	-8(%rbp), %rax
	movq	(%rax), %rax
	subq	%rax, %rdx
	movq	%rdx, %rax
	sarq	$3, %rax
	movq	%rax, %rdx
	movabsq	$-6148914691236517205, %rax
	imulq	%rdx, %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1054:
	.size	_ZNKSt6vectorIS_IPfSaIS0_EESaIS2_EE4sizeEv, .-_ZNKSt6vectorIS_IPfSaIS0_EESaIS2_EE4sizeEv
	.section	.text._ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE15_M_erase_at_endEPS2_,"axG",@progbits,_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE15_M_erase_at_endEPS2_,comdat
	.align 2
	.weak	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE15_M_erase_at_endEPS2_
	.type	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE15_M_erase_at_endEPS2_, @function
_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE15_M_erase_at_endEPS2_:
.LFB1055:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA1055
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	subq	-32(%rbp), %rax
	sarq	$3, %rax
	movq	%rax, %rdx
	movabsq	$-6148914691236517205, %rax
	imulq	%rdx, %rax
	movq	%rax, -8(%rbp)
	cmpq	$0, -8(%rbp)
	je	.L125
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE19_M_get_Tp_allocatorEv
	movq	%rax, %rdx
	movq	-24(%rbp), %rax
	movq	8(%rax), %rcx
	movq	-32(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZSt8_DestroyIPSt6vectorIPfSaIS1_EES3_EvT_S5_RSaIT0_E
	movq	-24(%rbp), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, 8(%rax)
.L125:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1055:
	.section	.gcc_except_table
.LLSDA1055:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE1055-.LLSDACSB1055
.LLSDACSB1055:
.LLSDACSE1055:
	.section	.text._ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE15_M_erase_at_endEPS2_,"axG",@progbits,_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE15_M_erase_at_endEPS2_,comdat
	.size	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE15_M_erase_at_endEPS2_, .-_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE15_M_erase_at_endEPS2_
	.section	.text._ZSt4copyIPKSt6vectorIPfSaIS1_EEPS3_ET0_T_S8_S7_,"axG",@progbits,_ZSt4copyIPKSt6vectorIPfSaIS1_EEPS3_ET0_T_S8_S7_,comdat
	.weak	_ZSt4copyIPKSt6vectorIPfSaIS1_EEPS3_ET0_T_S8_S7_
	.type	_ZSt4copyIPKSt6vectorIPfSaIS1_EEPS3_ET0_T_S8_S7_, @function
_ZSt4copyIPKSt6vectorIPfSaIS1_EEPS3_ET0_T_S8_S7_:
.LFB1056:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$40, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__miter_baseIPKSt6vectorIPfSaIS1_EEET_S6_
	movq	%rax, %rbx
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__miter_baseIPKSt6vectorIPfSaIS1_EEET_S6_
	movq	%rax, %rcx
	movq	-40(%rbp), %rax
	movq	%rax, %rdx
	movq	%rbx, %rsi
	movq	%rcx, %rdi
	call	_ZSt14__copy_move_a2ILb0EPKSt6vectorIPfSaIS1_EEPS3_ET1_T0_S8_S7_
	addq	$40, %rsp
	popq	%rbx
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1056:
	.size	_ZSt4copyIPKSt6vectorIPfSaIS1_EEPS3_ET0_T_S8_S7_, .-_ZSt4copyIPKSt6vectorIPfSaIS1_EEPS3_ET0_T_S8_S7_
	.section	.text._ZSt7advanceIPKSt6vectorIPfSaIS1_EEmEvRT_T0_,"axG",@progbits,_ZSt7advanceIPKSt6vectorIPfSaIS1_EEmEvRT_T0_,comdat
	.weak	_ZSt7advanceIPKSt6vectorIPfSaIS1_EEmEvRT_T0_
	.type	_ZSt7advanceIPKSt6vectorIPfSaIS1_EEmEvRT_T0_, @function
_ZSt7advanceIPKSt6vectorIPfSaIS1_EEmEvRT_T0_:
.LFB1057:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	-48(%rbp), %rax
	movq	%rax, -16(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt19__iterator_categoryIPKSt6vectorIPfSaIS1_EEENSt15iterator_traitsIT_E17iterator_categoryERKS7_
	movq	-16(%rbp), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZSt9__advanceIPKSt6vectorIPfSaIS1_EElEvRT_T0_St26random_access_iterator_tag
	nop
	movq	-8(%rbp), %rax
	xorq	%fs:40, %rax
	je	.L129
	call	__stack_chk_fail@PLT
.L129:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1057:
	.size	_ZSt7advanceIPKSt6vectorIPfSaIS1_EEmEvRT_T0_, .-_ZSt7advanceIPKSt6vectorIPfSaIS1_EEmEvRT_T0_
	.section	.text._ZSt22__uninitialized_copy_aIPKSt6vectorIPfSaIS1_EEPS3_S3_ET0_T_S8_S7_RSaIT1_E,"axG",@progbits,_ZSt22__uninitialized_copy_aIPKSt6vectorIPfSaIS1_EEPS3_S3_ET0_T_S8_S7_RSaIT1_E,comdat
	.weak	_ZSt22__uninitialized_copy_aIPKSt6vectorIPfSaIS1_EEPS3_S3_ET0_T_S8_S7_RSaIT1_E
	.type	_ZSt22__uninitialized_copy_aIPKSt6vectorIPfSaIS1_EEPS3_S3_ET0_T_S8_S7_RSaIT1_E, @function
_ZSt22__uninitialized_copy_aIPKSt6vectorIPfSaIS1_EEPS3_S3_ET0_T_S8_S7_RSaIT1_E:
.LFB1058:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	%rcx, -32(%rbp)
	movq	-24(%rbp), %rdx
	movq	-16(%rbp), %rcx
	movq	-8(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZSt18uninitialized_copyIPKSt6vectorIPfSaIS1_EEPS3_ET0_T_S8_S7_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1058:
	.size	_ZSt22__uninitialized_copy_aIPKSt6vectorIPfSaIS1_EEPS3_S3_ET0_T_S8_S7_RSaIT1_E, .-_ZSt22__uninitialized_copy_aIPKSt6vectorIPfSaIS1_EEPS3_S3_ET0_T_S8_S7_RSaIT1_E
	.section	.text._ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEEC2ERKS5_,"axG",@progbits,_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEEC5ERKS5_,comdat
	.align 2
	.weak	_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEEC2ERKS5_
	.type	_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEEC2ERKS5_, @function
_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEEC2ERKS5_:
.LFB1060:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1060:
	.size	_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEEC2ERKS5_, .-_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEEC2ERKS5_
	.weak	_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEEC1ERKS5_
	.set	_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEEC1ERKS5_,_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEEC2ERKS5_
	.section	.text._ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEE10deallocateEPS4_m,"axG",@progbits,_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEE10deallocateEPS4_m,comdat
	.align 2
	.weak	_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEE10deallocateEPS4_m
	.type	_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEE10deallocateEPS4_m, @function
_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEE10deallocateEPS4_m:
.LFB1065:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, %rdi
	call	_ZdlPv@PLT
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1065:
	.size	_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEE10deallocateEPS4_m, .-_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEE10deallocateEPS4_m
	.section	.text._ZNSt12_Destroy_auxILb0EE9__destroyIPSt6vectorIPfSaIS3_EEEEvT_S7_,"axG",@progbits,_ZNSt12_Destroy_auxILb0EE9__destroyIPSt6vectorIPfSaIS3_EEEEvT_S7_,comdat
	.weak	_ZNSt12_Destroy_auxILb0EE9__destroyIPSt6vectorIPfSaIS3_EEEEvT_S7_
	.type	_ZNSt12_Destroy_auxILb0EE9__destroyIPSt6vectorIPfSaIS3_EEEEvT_S7_, @function
_ZNSt12_Destroy_auxILb0EE9__destroyIPSt6vectorIPfSaIS3_EEEEvT_S7_:
.LFB1066:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
.L136:
	movq	-8(%rbp), %rax
	cmpq	-16(%rbp), %rax
	je	.L137
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt11__addressofISt6vectorIPfSaIS1_EEEPT_RS4_
	movq	%rax, %rdi
	call	_ZSt8_DestroyISt6vectorIPfSaIS1_EEEvPT_
	addq	$24, -8(%rbp)
	jmp	.L136
.L137:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1066:
	.size	_ZNSt12_Destroy_auxILb0EE9__destroyIPSt6vectorIPfSaIS3_EEEEvT_S7_, .-_ZNSt12_Destroy_auxILb0EE9__destroyIPSt6vectorIPfSaIS3_EEEEvT_S7_
	.section	.text._ZN9__gnu_cxx13new_allocatorIiE10deallocateEPim,"axG",@progbits,_ZN9__gnu_cxx13new_allocatorIiE10deallocateEPim,comdat
	.align 2
	.weak	_ZN9__gnu_cxx13new_allocatorIiE10deallocateEPim
	.type	_ZN9__gnu_cxx13new_allocatorIiE10deallocateEPim, @function
_ZN9__gnu_cxx13new_allocatorIiE10deallocateEPim:
.LFB1067:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, %rdi
	call	_ZdlPv@PLT
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1067:
	.size	_ZN9__gnu_cxx13new_allocatorIiE10deallocateEPim, .-_ZN9__gnu_cxx13new_allocatorIiE10deallocateEPim
	.section	.text._ZNSt12_Destroy_auxILb1EE9__destroyIPiEEvT_S3_,"axG",@progbits,_ZNSt12_Destroy_auxILb1EE9__destroyIPiEEvT_S3_,comdat
	.weak	_ZNSt12_Destroy_auxILb1EE9__destroyIPiEEvT_S3_
	.type	_ZNSt12_Destroy_auxILb1EE9__destroyIPiEEvT_S3_, @function
_ZNSt12_Destroy_auxILb1EE9__destroyIPiEEvT_S3_:
.LFB1068:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1068:
	.size	_ZNSt12_Destroy_auxILb1EE9__destroyIPiEEvT_S3_, .-_ZNSt12_Destroy_auxILb1EE9__destroyIPiEEvT_S3_
	.section	.text._ZN9__gnu_cxx13new_allocatorIPfE10deallocateEPS1_m,"axG",@progbits,_ZN9__gnu_cxx13new_allocatorIPfE10deallocateEPS1_m,comdat
	.align 2
	.weak	_ZN9__gnu_cxx13new_allocatorIPfE10deallocateEPS1_m
	.type	_ZN9__gnu_cxx13new_allocatorIPfE10deallocateEPS1_m, @function
_ZN9__gnu_cxx13new_allocatorIPfE10deallocateEPS1_m:
.LFB1069:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, %rdi
	call	_ZdlPv@PLT
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1069:
	.size	_ZN9__gnu_cxx13new_allocatorIPfE10deallocateEPS1_m, .-_ZN9__gnu_cxx13new_allocatorIPfE10deallocateEPS1_m
	.section	.text._ZNSt12_Destroy_auxILb1EE9__destroyIPPfEEvT_S4_,"axG",@progbits,_ZNSt12_Destroy_auxILb1EE9__destroyIPPfEEvT_S4_,comdat
	.weak	_ZNSt12_Destroy_auxILb1EE9__destroyIPPfEEvT_S4_
	.type	_ZNSt12_Destroy_auxILb1EE9__destroyIPPfEEvT_S4_, @function
_ZNSt12_Destroy_auxILb1EE9__destroyIPPfEEvT_S4_:
.LFB1070:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1070:
	.size	_ZNSt12_Destroy_auxILb1EE9__destroyIPPfEEvT_S4_, .-_ZNSt12_Destroy_auxILb1EE9__destroyIPPfEEvT_S4_
	.section	.text._ZSt19__iterator_categoryIPKSt6vectorIPfSaIS1_EEENSt15iterator_traitsIT_E17iterator_categoryERKS7_,"axG",@progbits,_ZSt19__iterator_categoryIPKSt6vectorIPfSaIS1_EEENSt15iterator_traitsIT_E17iterator_categoryERKS7_,comdat
	.weak	_ZSt19__iterator_categoryIPKSt6vectorIPfSaIS1_EEENSt15iterator_traitsIT_E17iterator_categoryERKS7_
	.type	_ZSt19__iterator_categoryIPKSt6vectorIPfSaIS1_EEENSt15iterator_traitsIT_E17iterator_categoryERKS7_, @function
_ZSt19__iterator_categoryIPKSt6vectorIPfSaIS1_EEENSt15iterator_traitsIT_E17iterator_categoryERKS7_:
.LFB1071:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1071:
	.size	_ZSt19__iterator_categoryIPKSt6vectorIPfSaIS1_EEENSt15iterator_traitsIT_E17iterator_categoryERKS7_, .-_ZSt19__iterator_categoryIPKSt6vectorIPfSaIS1_EEENSt15iterator_traitsIT_E17iterator_categoryERKS7_
	.section	.text._ZSt10__distanceIPKSt6vectorIPfSaIS1_EEENSt15iterator_traitsIT_E15difference_typeES7_S7_St26random_access_iterator_tag,"axG",@progbits,_ZSt10__distanceIPKSt6vectorIPfSaIS1_EEENSt15iterator_traitsIT_E15difference_typeES7_S7_St26random_access_iterator_tag,comdat
	.weak	_ZSt10__distanceIPKSt6vectorIPfSaIS1_EEENSt15iterator_traitsIT_E15difference_typeES7_S7_St26random_access_iterator_tag
	.type	_ZSt10__distanceIPKSt6vectorIPfSaIS1_EEENSt15iterator_traitsIT_E15difference_typeES7_S7_St26random_access_iterator_tag, @function
_ZSt10__distanceIPKSt6vectorIPfSaIS1_EEENSt15iterator_traitsIT_E15difference_typeES7_S7_St26random_access_iterator_tag:
.LFB1072:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rax
	subq	-8(%rbp), %rax
	sarq	$3, %rax
	movq	%rax, %rdx
	movabsq	$-6148914691236517205, %rax
	imulq	%rdx, %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1072:
	.size	_ZSt10__distanceIPKSt6vectorIPfSaIS1_EEENSt15iterator_traitsIT_E15difference_typeES7_S7_St26random_access_iterator_tag, .-_ZSt10__distanceIPKSt6vectorIPfSaIS1_EEENSt15iterator_traitsIT_E15difference_typeES7_S7_St26random_access_iterator_tag
	.section	.text._ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE11_S_max_sizeERKS3_,"axG",@progbits,_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE11_S_max_sizeERKS3_,comdat
	.weak	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE11_S_max_sizeERKS3_
	.type	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE11_S_max_sizeERKS3_, @function
_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE11_S_max_sizeERKS3_:
.LFB1073:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movabsq	$384307168202282325, %rax
	movq	%rax, -24(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt16allocator_traitsISaISt6vectorIPfSaIS1_EEEE8max_sizeERKS4_
	movq	%rax, -16(%rbp)
	leaq	-16(%rbp), %rdx
	leaq	-24(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZSt3minImERKT_S2_S2_
	movq	(%rax), %rax
	movq	-8(%rbp), %rcx
	xorq	%fs:40, %rcx
	je	.L148
	call	__stack_chk_fail@PLT
.L148:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1073:
	.size	_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE11_S_max_sizeERKS3_, .-_ZNSt6vectorIS_IPfSaIS0_EESaIS2_EE11_S_max_sizeERKS3_
	.section	.text._ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE11_M_allocateEm,"axG",@progbits,_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE11_M_allocateEm,comdat
	.align 2
	.weak	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE11_M_allocateEm
	.type	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE11_M_allocateEm, @function
_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE11_M_allocateEm:
.LFB1074:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	cmpq	$0, -16(%rbp)
	je	.L150
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSt16allocator_traitsISaISt6vectorIPfSaIS1_EEEE8allocateERS4_m
	jmp	.L152
.L150:
	movl	$0, %eax
.L152:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1074:
	.size	_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE11_M_allocateEm, .-_ZNSt12_Vector_baseISt6vectorIPfSaIS1_EESaIS3_EE11_M_allocateEm
	.section	.text._ZSt12__miter_baseIPKSt6vectorIPfSaIS1_EEET_S6_,"axG",@progbits,_ZSt12__miter_baseIPKSt6vectorIPfSaIS1_EEET_S6_,comdat
	.weak	_ZSt12__miter_baseIPKSt6vectorIPfSaIS1_EEET_S6_
	.type	_ZSt12__miter_baseIPKSt6vectorIPfSaIS1_EEET_S6_, @function
_ZSt12__miter_baseIPKSt6vectorIPfSaIS1_EEET_S6_:
.LFB1075:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1075:
	.size	_ZSt12__miter_baseIPKSt6vectorIPfSaIS1_EEET_S6_, .-_ZSt12__miter_baseIPKSt6vectorIPfSaIS1_EEET_S6_
	.section	.text._ZSt14__copy_move_a2ILb0EPKSt6vectorIPfSaIS1_EEPS3_ET1_T0_S8_S7_,"axG",@progbits,_ZSt14__copy_move_a2ILb0EPKSt6vectorIPfSaIS1_EEPS3_ET1_T0_S8_S7_,comdat
	.weak	_ZSt14__copy_move_a2ILb0EPKSt6vectorIPfSaIS1_EEPS3_ET1_T0_S8_S7_
	.type	_ZSt14__copy_move_a2ILb0EPKSt6vectorIPfSaIS1_EEPS3_ET1_T0_S8_S7_, @function
_ZSt14__copy_move_a2ILb0EPKSt6vectorIPfSaIS1_EEPS3_ET1_T0_S8_S7_:
.LFB1076:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r12
	pushq	%rbx
	subq	$32, %rsp
	.cfi_offset 12, -24
	.cfi_offset 3, -32
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__niter_baseIPSt6vectorIPfSaIS1_EEET_S5_
	movq	%rax, %r12
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__niter_baseIPKSt6vectorIPfSaIS1_EEET_S6_
	movq	%rax, %rbx
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__niter_baseIPKSt6vectorIPfSaIS1_EEET_S6_
	movq	%r12, %rdx
	movq	%rbx, %rsi
	movq	%rax, %rdi
	call	_ZSt13__copy_move_aILb0EPKSt6vectorIPfSaIS1_EEPS3_ET1_T0_S8_S7_
	movq	%rax, %rdx
	leaq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZSt12__niter_wrapIPSt6vectorIPfSaIS1_EEET_RKS5_S5_
	addq	$32, %rsp
	popq	%rbx
	popq	%r12
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1076:
	.size	_ZSt14__copy_move_a2ILb0EPKSt6vectorIPfSaIS1_EEPS3_ET1_T0_S8_S7_, .-_ZSt14__copy_move_a2ILb0EPKSt6vectorIPfSaIS1_EEPS3_ET1_T0_S8_S7_
	.section	.text._ZSt9__advanceIPKSt6vectorIPfSaIS1_EElEvRT_T0_St26random_access_iterator_tag,"axG",@progbits,_ZSt9__advanceIPKSt6vectorIPfSaIS1_EElEvRT_T0_St26random_access_iterator_tag,comdat
	.weak	_ZSt9__advanceIPKSt6vectorIPfSaIS1_EElEvRT_T0_St26random_access_iterator_tag
	.type	_ZSt9__advanceIPKSt6vectorIPfSaIS1_EElEvRT_T0_St26random_access_iterator_tag, @function
_ZSt9__advanceIPKSt6vectorIPfSaIS1_EElEvRT_T0_St26random_access_iterator_tag:
.LFB1077:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	(%rax), %rcx
	movq	-16(%rbp), %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	leaq	(%rcx,%rax), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, (%rax)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1077:
	.size	_ZSt9__advanceIPKSt6vectorIPfSaIS1_EElEvRT_T0_St26random_access_iterator_tag, .-_ZSt9__advanceIPKSt6vectorIPfSaIS1_EElEvRT_T0_St26random_access_iterator_tag
	.section	.text._ZSt18uninitialized_copyIPKSt6vectorIPfSaIS1_EEPS3_ET0_T_S8_S7_,"axG",@progbits,_ZSt18uninitialized_copyIPKSt6vectorIPfSaIS1_EEPS3_ET0_T_S8_S7_,comdat
	.weak	_ZSt18uninitialized_copyIPKSt6vectorIPfSaIS1_EEPS3_ET0_T_S8_S7_
	.type	_ZSt18uninitialized_copyIPKSt6vectorIPfSaIS1_EEPS3_ET0_T_S8_S7_, @function
_ZSt18uninitialized_copyIPKSt6vectorIPfSaIS1_EEPS3_ET0_T_S8_S7_:
.LFB1078:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movb	$1, -1(%rbp)
	movq	-40(%rbp), %rdx
	movq	-32(%rbp), %rcx
	movq	-24(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZNSt20__uninitialized_copyILb0EE13__uninit_copyIPKSt6vectorIPfSaIS3_EEPS5_EET0_T_SA_S9_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1078:
	.size	_ZSt18uninitialized_copyIPKSt6vectorIPfSaIS1_EEPS3_ET0_T_S8_S7_, .-_ZSt18uninitialized_copyIPKSt6vectorIPfSaIS1_EEPS3_ET0_T_S8_S7_
	.section	.text._ZSt11__addressofISt6vectorIPfSaIS1_EEEPT_RS4_,"axG",@progbits,_ZSt11__addressofISt6vectorIPfSaIS1_EEEPT_RS4_,comdat
	.weak	_ZSt11__addressofISt6vectorIPfSaIS1_EEEPT_RS4_
	.type	_ZSt11__addressofISt6vectorIPfSaIS1_EEEPT_RS4_, @function
_ZSt11__addressofISt6vectorIPfSaIS1_EEEPT_RS4_:
.LFB1079:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1079:
	.size	_ZSt11__addressofISt6vectorIPfSaIS1_EEEPT_RS4_, .-_ZSt11__addressofISt6vectorIPfSaIS1_EEEPT_RS4_
	.section	.text._ZSt8_DestroyISt6vectorIPfSaIS1_EEEvPT_,"axG",@progbits,_ZSt8_DestroyISt6vectorIPfSaIS1_EEEvPT_,comdat
	.weak	_ZSt8_DestroyISt6vectorIPfSaIS1_EEEvPT_
	.type	_ZSt8_DestroyISt6vectorIPfSaIS1_EEEvPT_, @function
_ZSt8_DestroyISt6vectorIPfSaIS1_EEEvPT_:
.LFB1080:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt6vectorIPfSaIS0_EED1Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1080:
	.size	_ZSt8_DestroyISt6vectorIPfSaIS1_EEEvPT_, .-_ZSt8_DestroyISt6vectorIPfSaIS1_EEEvPT_
	.section	.text._ZNSt16allocator_traitsISaISt6vectorIPfSaIS1_EEEE8max_sizeERKS4_,"axG",@progbits,_ZNSt16allocator_traitsISaISt6vectorIPfSaIS1_EEEE8max_sizeERKS4_,comdat
	.weak	_ZNSt16allocator_traitsISaISt6vectorIPfSaIS1_EEEE8max_sizeERKS4_
	.type	_ZNSt16allocator_traitsISaISt6vectorIPfSaIS1_EEEE8max_sizeERKS4_, @function
_ZNSt16allocator_traitsISaISt6vectorIPfSaIS1_EEEE8max_sizeERKS4_:
.LFB1081:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNK9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEE8max_sizeEv
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1081:
	.size	_ZNSt16allocator_traitsISaISt6vectorIPfSaIS1_EEEE8max_sizeERKS4_, .-_ZNSt16allocator_traitsISaISt6vectorIPfSaIS1_EEEE8max_sizeERKS4_
	.section	.text._ZSt3minImERKT_S2_S2_,"axG",@progbits,_ZSt3minImERKT_S2_S2_,comdat
	.weak	_ZSt3minImERKT_S2_S2_
	.type	_ZSt3minImERKT_S2_S2_, @function
_ZSt3minImERKT_S2_S2_:
.LFB1082:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	(%rax), %rdx
	movq	-8(%rbp), %rax
	movq	(%rax), %rax
	cmpq	%rax, %rdx
	jnb	.L166
	movq	-16(%rbp), %rax
	jmp	.L167
.L166:
	movq	-8(%rbp), %rax
.L167:
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1082:
	.size	_ZSt3minImERKT_S2_S2_, .-_ZSt3minImERKT_S2_S2_
	.section	.text._ZNSt16allocator_traitsISaISt6vectorIPfSaIS1_EEEE8allocateERS4_m,"axG",@progbits,_ZNSt16allocator_traitsISaISt6vectorIPfSaIS1_EEEE8allocateERS4_m,comdat
	.weak	_ZNSt16allocator_traitsISaISt6vectorIPfSaIS1_EEEE8allocateERS4_m
	.type	_ZNSt16allocator_traitsISaISt6vectorIPfSaIS1_EEEE8allocateERS4_m, @function
_ZNSt16allocator_traitsISaISt6vectorIPfSaIS1_EEEE8allocateERS4_m:
.LFB1083:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rcx
	movq	-8(%rbp), %rax
	movl	$0, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEE8allocateEmPKv
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1083:
	.size	_ZNSt16allocator_traitsISaISt6vectorIPfSaIS1_EEEE8allocateERS4_m, .-_ZNSt16allocator_traitsISaISt6vectorIPfSaIS1_EEEE8allocateERS4_m
	.section	.text._ZSt12__niter_baseIPKSt6vectorIPfSaIS1_EEET_S6_,"axG",@progbits,_ZSt12__niter_baseIPKSt6vectorIPfSaIS1_EEET_S6_,comdat
	.weak	_ZSt12__niter_baseIPKSt6vectorIPfSaIS1_EEET_S6_
	.type	_ZSt12__niter_baseIPKSt6vectorIPfSaIS1_EEET_S6_, @function
_ZSt12__niter_baseIPKSt6vectorIPfSaIS1_EEET_S6_:
.LFB1084:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1084:
	.size	_ZSt12__niter_baseIPKSt6vectorIPfSaIS1_EEET_S6_, .-_ZSt12__niter_baseIPKSt6vectorIPfSaIS1_EEET_S6_
	.section	.text._ZSt12__niter_baseIPSt6vectorIPfSaIS1_EEET_S5_,"axG",@progbits,_ZSt12__niter_baseIPSt6vectorIPfSaIS1_EEET_S5_,comdat
	.weak	_ZSt12__niter_baseIPSt6vectorIPfSaIS1_EEET_S5_
	.type	_ZSt12__niter_baseIPSt6vectorIPfSaIS1_EEET_S5_, @function
_ZSt12__niter_baseIPSt6vectorIPfSaIS1_EEET_S5_:
.LFB1085:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1085:
	.size	_ZSt12__niter_baseIPSt6vectorIPfSaIS1_EEET_S5_, .-_ZSt12__niter_baseIPSt6vectorIPfSaIS1_EEET_S5_
	.section	.text._ZSt13__copy_move_aILb0EPKSt6vectorIPfSaIS1_EEPS3_ET1_T0_S8_S7_,"axG",@progbits,_ZSt13__copy_move_aILb0EPKSt6vectorIPfSaIS1_EEPS3_ET1_T0_S8_S7_,comdat
	.weak	_ZSt13__copy_move_aILb0EPKSt6vectorIPfSaIS1_EEPS3_ET1_T0_S8_S7_
	.type	_ZSt13__copy_move_aILb0EPKSt6vectorIPfSaIS1_EEPS3_ET1_T0_S8_S7_, @function
_ZSt13__copy_move_aILb0EPKSt6vectorIPfSaIS1_EEPS3_ET1_T0_S8_S7_:
.LFB1086:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movb	$0, -1(%rbp)
	movq	-40(%rbp), %rdx
	movq	-32(%rbp), %rcx
	movq	-24(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZNSt11__copy_moveILb0ELb0ESt26random_access_iterator_tagE8__copy_mIPKSt6vectorIPfSaIS4_EEPS6_EET0_T_SB_SA_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1086:
	.size	_ZSt13__copy_move_aILb0EPKSt6vectorIPfSaIS1_EEPS3_ET1_T0_S8_S7_, .-_ZSt13__copy_move_aILb0EPKSt6vectorIPfSaIS1_EEPS3_ET1_T0_S8_S7_
	.section	.text._ZSt12__niter_wrapIPSt6vectorIPfSaIS1_EEET_RKS5_S5_,"axG",@progbits,_ZSt12__niter_wrapIPSt6vectorIPfSaIS1_EEET_RKS5_S5_,comdat
	.weak	_ZSt12__niter_wrapIPSt6vectorIPfSaIS1_EEET_RKS5_S5_
	.type	_ZSt12__niter_wrapIPSt6vectorIPfSaIS1_EEET_RKS5_S5_, @function
_ZSt12__niter_wrapIPSt6vectorIPfSaIS1_EEET_RKS5_S5_:
.LFB1087:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1087:
	.size	_ZSt12__niter_wrapIPSt6vectorIPfSaIS1_EEET_RKS5_S5_, .-_ZSt12__niter_wrapIPSt6vectorIPfSaIS1_EEET_RKS5_S5_
	.section	.text._ZNSt20__uninitialized_copyILb0EE13__uninit_copyIPKSt6vectorIPfSaIS3_EEPS5_EET0_T_SA_S9_,"axG",@progbits,_ZNSt20__uninitialized_copyILb0EE13__uninit_copyIPKSt6vectorIPfSaIS3_EEPS5_EET0_T_SA_S9_,comdat
	.weak	_ZNSt20__uninitialized_copyILb0EE13__uninit_copyIPKSt6vectorIPfSaIS3_EEPS5_EET0_T_SA_S9_
	.type	_ZNSt20__uninitialized_copyILb0EE13__uninit_copyIPKSt6vectorIPfSaIS3_EEPS5_EET0_T_SA_S9_, @function
_ZNSt20__uninitialized_copyILb0EE13__uninit_copyIPKSt6vectorIPfSaIS3_EEPS5_EET0_T_SA_S9_:
.LFB1088:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA1088
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$56, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	-56(%rbp), %rax
	movq	%rax, -24(%rbp)
.L180:
	movq	-40(%rbp), %rax
	cmpq	-48(%rbp), %rax
	je	.L179
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt11__addressofISt6vectorIPfSaIS1_EEEPT_RS4_
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	movq	%rax, %rsi
	movq	%rdx, %rdi
.LEHB6:
	call	_ZSt10_ConstructISt6vectorIPfSaIS1_EEJRKS3_EEvPT_DpOT0_
.LEHE6:
	addq	$24, -40(%rbp)
	addq	$24, -24(%rbp)
	jmp	.L180
.L179:
	movq	-24(%rbp), %rax
	jmp	.L186
.L184:
	endbr64
	movq	%rax, %rdi
	call	__cxa_begin_catch@PLT
	movq	-24(%rbp), %rdx
	movq	-56(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
.LEHB7:
	call	_ZSt8_DestroyIPSt6vectorIPfSaIS1_EEEvT_S5_
	call	__cxa_rethrow@PLT
.LEHE7:
.L185:
	endbr64
	movq	%rax, %rbx
	call	__cxa_end_catch@PLT
	movq	%rbx, %rax
	movq	%rax, %rdi
.LEHB8:
	call	_Unwind_Resume@PLT
.LEHE8:
.L186:
	addq	$56, %rsp
	popq	%rbx
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1088:
	.section	.gcc_except_table
	.align 4
.LLSDA1088:
	.byte	0xff
	.byte	0x9b
	.uleb128 .LLSDATT1088-.LLSDATTD1088
.LLSDATTD1088:
	.byte	0x1
	.uleb128 .LLSDACSE1088-.LLSDACSB1088
.LLSDACSB1088:
	.uleb128 .LEHB6-.LFB1088
	.uleb128 .LEHE6-.LEHB6
	.uleb128 .L184-.LFB1088
	.uleb128 0x1
	.uleb128 .LEHB7-.LFB1088
	.uleb128 .LEHE7-.LEHB7
	.uleb128 .L185-.LFB1088
	.uleb128 0
	.uleb128 .LEHB8-.LFB1088
	.uleb128 .LEHE8-.LEHB8
	.uleb128 0
	.uleb128 0
.LLSDACSE1088:
	.byte	0x1
	.byte	0
	.align 4
	.long	0

.LLSDATT1088:
	.section	.text._ZNSt20__uninitialized_copyILb0EE13__uninit_copyIPKSt6vectorIPfSaIS3_EEPS5_EET0_T_SA_S9_,"axG",@progbits,_ZNSt20__uninitialized_copyILb0EE13__uninit_copyIPKSt6vectorIPfSaIS3_EEPS5_EET0_T_SA_S9_,comdat
	.size	_ZNSt20__uninitialized_copyILb0EE13__uninit_copyIPKSt6vectorIPfSaIS3_EEPS5_EET0_T_SA_S9_, .-_ZNSt20__uninitialized_copyILb0EE13__uninit_copyIPKSt6vectorIPfSaIS3_EEPS5_EET0_T_SA_S9_
	.section	.text._ZNK9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEE8max_sizeEv,"axG",@progbits,_ZNK9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEE8max_sizeEv,comdat
	.align 2
	.weak	_ZNK9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEE8max_sizeEv
	.type	_ZNK9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEE8max_sizeEv, @function
_ZNK9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEE8max_sizeEv:
.LFB1089:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movabsq	$384307168202282325, %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1089:
	.size	_ZNK9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEE8max_sizeEv, .-_ZNK9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEE8max_sizeEv
	.section	.text._ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEE8allocateEmPKv,"axG",@progbits,_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEE8allocateEmPKv,comdat
	.align 2
	.weak	_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEE8allocateEmPKv
	.type	_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEE8allocateEmPKv, @function
_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEE8allocateEmPKv:
.LFB1090:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNK9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEE8max_sizeEv
	cmpq	%rax, -16(%rbp)
	seta	%al
	testb	%al, %al
	je	.L190
	call	_ZSt17__throw_bad_allocv@PLT
.L190:
	movq	-16(%rbp), %rdx
	movq	%rdx, %rax
	addq	%rax, %rax
	addq	%rdx, %rax
	salq	$3, %rax
	movq	%rax, %rdi
	call	_Znwm@PLT
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1090:
	.size	_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEE8allocateEmPKv, .-_ZN9__gnu_cxx13new_allocatorISt6vectorIPfSaIS2_EEE8allocateEmPKv
	.section	.text._ZNSt11__copy_moveILb0ELb0ESt26random_access_iterator_tagE8__copy_mIPKSt6vectorIPfSaIS4_EEPS6_EET0_T_SB_SA_,"axG",@progbits,_ZNSt11__copy_moveILb0ELb0ESt26random_access_iterator_tagE8__copy_mIPKSt6vectorIPfSaIS4_EEPS6_EET0_T_SB_SA_,comdat
	.weak	_ZNSt11__copy_moveILb0ELb0ESt26random_access_iterator_tagE8__copy_mIPKSt6vectorIPfSaIS4_EEPS6_EET0_T_SB_SA_
	.type	_ZNSt11__copy_moveILb0ELb0ESt26random_access_iterator_tagE8__copy_mIPKSt6vectorIPfSaIS4_EEPS6_EET0_T_SB_SA_, @function
_ZNSt11__copy_moveILb0ELb0ESt26random_access_iterator_tagE8__copy_mIPKSt6vectorIPfSaIS4_EEPS6_EET0_T_SB_SA_:
.LFB1091:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	-32(%rbp), %rax
	subq	-24(%rbp), %rax
	sarq	$3, %rax
	movq	%rax, %rdx
	movabsq	$-6148914691236517205, %rax
	imulq	%rdx, %rax
	movq	%rax, -8(%rbp)
.L194:
	cmpq	$0, -8(%rbp)
	jle	.L193
	movq	-24(%rbp), %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSt6vectorIPfSaIS0_EEaSERKS2_
	addq	$24, -24(%rbp)
	addq	$24, -40(%rbp)
	subq	$1, -8(%rbp)
	jmp	.L194
.L193:
	movq	-40(%rbp), %rax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1091:
	.size	_ZNSt11__copy_moveILb0ELb0ESt26random_access_iterator_tagE8__copy_mIPKSt6vectorIPfSaIS4_EEPS6_EET0_T_SB_SA_, .-_ZNSt11__copy_moveILb0ELb0ESt26random_access_iterator_tagE8__copy_mIPKSt6vectorIPfSaIS4_EEPS6_EET0_T_SB_SA_
	.section	.text._ZSt10_ConstructISt6vectorIPfSaIS1_EEJRKS3_EEvPT_DpOT0_,"axG",@progbits,_ZSt10_ConstructISt6vectorIPfSaIS1_EEJRKS3_EEvPT_DpOT0_,comdat
	.weak	_ZSt10_ConstructISt6vectorIPfSaIS1_EEJRKS3_EEvPT_DpOT0_
	.type	_ZSt10_ConstructISt6vectorIPfSaIS1_EEJRKS3_EEvPT_DpOT0_, @function
_ZSt10_ConstructISt6vectorIPfSaIS1_EEJRKS3_EEvPT_DpOT0_:
.LFB1092:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA1092
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$24, %rsp
	.cfi_offset 13, -24
	.cfi_offset 12, -32
	.cfi_offset 3, -40
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	-48(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt7forwardIRKSt6vectorIPfSaIS1_EEEOT_RNSt16remove_referenceIS6_E4typeE
	movq	%rax, %r13
	movq	-40(%rbp), %rbx
	movq	%rbx, %rsi
	movl	$24, %edi
	call	_ZnwmPv
	movq	%rax, %r12
	movq	%r13, %rsi
	movq	%r12, %rdi
.LEHB9:
	call	_ZNSt6vectorIPfSaIS0_EEC1ERKS2_
.LEHE9:
	jmp	.L199
.L198:
	endbr64
	movq	%rax, %r13
	movq	%rbx, %rsi
	movq	%r12, %rdi
	call	_ZdlPvS_
	movq	%r13, %rax
	movq	%rax, %rdi
.LEHB10:
	call	_Unwind_Resume@PLT
.LEHE10:
.L199:
	addq	$24, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1092:
	.section	.gcc_except_table
.LLSDA1092:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE1092-.LLSDACSB1092
.LLSDACSB1092:
	.uleb128 .LEHB9-.LFB1092
	.uleb128 .LEHE9-.LEHB9
	.uleb128 .L198-.LFB1092
	.uleb128 0
	.uleb128 .LEHB10-.LFB1092
	.uleb128 .LEHE10-.LEHB10
	.uleb128 0
	.uleb128 0
.LLSDACSE1092:
	.section	.text._ZSt10_ConstructISt6vectorIPfSaIS1_EEJRKS3_EEvPT_DpOT0_,"axG",@progbits,_ZSt10_ConstructISt6vectorIPfSaIS1_EEJRKS3_EEvPT_DpOT0_,comdat
	.size	_ZSt10_ConstructISt6vectorIPfSaIS1_EEJRKS3_EEvPT_DpOT0_, .-_ZSt10_ConstructISt6vectorIPfSaIS1_EEJRKS3_EEvPT_DpOT0_
	.section	.text._ZN9__gnu_cxx14__alloc_traitsISaIPfES1_E27_S_propagate_on_copy_assignEv,"axG",@progbits,_ZN9__gnu_cxx14__alloc_traitsISaIPfES1_E27_S_propagate_on_copy_assignEv,comdat
	.weak	_ZN9__gnu_cxx14__alloc_traitsISaIPfES1_E27_S_propagate_on_copy_assignEv
	.type	_ZN9__gnu_cxx14__alloc_traitsISaIPfES1_E27_S_propagate_on_copy_assignEv, @function
_ZN9__gnu_cxx14__alloc_traitsISaIPfES1_E27_S_propagate_on_copy_assignEv:
.LFB1094:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	$0, %eax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1094:
	.size	_ZN9__gnu_cxx14__alloc_traitsISaIPfES1_E27_S_propagate_on_copy_assignEv, .-_ZN9__gnu_cxx14__alloc_traitsISaIPfES1_E27_S_propagate_on_copy_assignEv
	.section	.text._ZN9__gnu_cxx14__alloc_traitsISaIPfES1_E15_S_always_equalEv,"axG",@progbits,_ZN9__gnu_cxx14__alloc_traitsISaIPfES1_E15_S_always_equalEv,comdat
	.weak	_ZN9__gnu_cxx14__alloc_traitsISaIPfES1_E15_S_always_equalEv
	.type	_ZN9__gnu_cxx14__alloc_traitsISaIPfES1_E15_S_always_equalEv, @function
_ZN9__gnu_cxx14__alloc_traitsISaIPfES1_E15_S_always_equalEv:
.LFB1095:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movl	$1, %eax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1095:
	.size	_ZN9__gnu_cxx14__alloc_traitsISaIPfES1_E15_S_always_equalEv, .-_ZN9__gnu_cxx14__alloc_traitsISaIPfES1_E15_S_always_equalEv
	.section	.text._ZNSt6vectorIPfSaIS0_EEaSERKS2_,"axG",@progbits,_ZNSt6vectorIPfSaIS0_EEaSERKS2_,comdat
	.align 2
	.weak	_ZNSt6vectorIPfSaIS0_EEaSERKS2_
	.type	_ZNSt6vectorIPfSaIS0_EEaSERKS2_, @function
_ZNSt6vectorIPfSaIS0_EEaSERKS2_:
.LFB1093:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$32, %rsp
	.cfi_offset 14, -24
	.cfi_offset 13, -32
	.cfi_offset 12, -40
	.cfi_offset 3, -48
	movq	%rdi, -56(%rbp)
	movq	%rsi, -64(%rbp)
	movq	-64(%rbp), %rax
	cmpq	-56(%rbp), %rax
	je	.L205
	call	_ZN9__gnu_cxx14__alloc_traitsISaIPfES1_E27_S_propagate_on_copy_assignEv
	testb	%al, %al
	je	.L206
	call	_ZN9__gnu_cxx14__alloc_traitsISaIPfES1_E15_S_always_equalEv
	xorl	$1, %eax
	testb	%al, %al
	je	.L207
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt12_Vector_baseIPfSaIS0_EE19_M_get_Tp_allocatorEv
	movq	%rax, %rbx
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIPfSaIS0_EE19_M_get_Tp_allocatorEv
	movq	%rbx, %rsi
	movq	%rax, %rdi
	call	_ZStneRKSaIPfES2_
	testb	%al, %al
	je	.L207
	movl	$1, %eax
	jmp	.L208
.L207:
	movl	$0, %eax
.L208:
	testb	%al, %al
	je	.L209
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt6vectorIPfSaIS0_EE5clearEv
	movq	-56(%rbp), %rax
	movq	-56(%rbp), %rdx
	movq	16(%rdx), %rcx
	movq	-56(%rbp), %rdx
	movq	(%rdx), %rdx
	subq	%rdx, %rcx
	movq	%rcx, %rdx
	sarq	$3, %rdx
	movq	%rdx, %rsi
	movq	-56(%rbp), %rdx
	movq	(%rdx), %rcx
	movq	%rsi, %rdx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIPfSaIS0_EE13_M_deallocateEPS0_m
	movq	-56(%rbp), %rax
	movq	$0, (%rax)
	movq	-56(%rbp), %rax
	movq	$0, 8(%rax)
	movq	-56(%rbp), %rax
	movq	$0, 16(%rax)
.L209:
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt12_Vector_baseIPfSaIS0_EE19_M_get_Tp_allocatorEv
	movq	%rax, %rbx
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIPfSaIS0_EE19_M_get_Tp_allocatorEv
	movq	%rbx, %rsi
	movq	%rax, %rdi
	call	_ZSt15__alloc_on_copyISaIPfEEvRT_RKS2_
.L206:
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt6vectorIPfSaIS0_EE4sizeEv
	movq	%rax, -48(%rbp)
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt6vectorIPfSaIS0_EE8capacityEv
	cmpq	%rax, -48(%rbp)
	seta	%al
	testb	%al, %al
	je	.L210
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt6vectorIPfSaIS0_EE3endEv
	movq	%rax, %rbx
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt6vectorIPfSaIS0_EE5beginEv
	movq	%rax, %rdx
	movq	-48(%rbp), %rsi
	movq	-56(%rbp), %rax
	movq	%rbx, %rcx
	movq	%rax, %rdi
	call	_ZNSt6vectorIPfSaIS0_EE20_M_allocate_and_copyIN9__gnu_cxx17__normal_iteratorIPKS0_S2_EEEEPS0_mT_SA_
	movq	%rax, -40(%rbp)
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIPfSaIS0_EE19_M_get_Tp_allocatorEv
	movq	%rax, %rdx
	movq	-56(%rbp), %rax
	movq	8(%rax), %rcx
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZSt8_DestroyIPPfS0_EvT_S2_RSaIT0_E
	movq	-56(%rbp), %rax
	movq	-56(%rbp), %rdx
	movq	16(%rdx), %rcx
	movq	-56(%rbp), %rdx
	movq	(%rdx), %rdx
	subq	%rdx, %rcx
	movq	%rcx, %rdx
	sarq	$3, %rdx
	movq	%rdx, %rsi
	movq	-56(%rbp), %rdx
	movq	(%rdx), %rcx
	movq	%rsi, %rdx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIPfSaIS0_EE13_M_deallocateEPS0_m
	movq	-56(%rbp), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, (%rax)
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	movq	-48(%rbp), %rdx
	salq	$3, %rdx
	addq	%rax, %rdx
	movq	-56(%rbp), %rax
	movq	%rdx, 16(%rax)
	jmp	.L211
.L210:
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt6vectorIPfSaIS0_EE4sizeEv
	cmpq	%rax, -48(%rbp)
	setbe	%al
	testb	%al, %al
	je	.L212
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIPfSaIS0_EE19_M_get_Tp_allocatorEv
	movq	%rax, %r12
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt6vectorIPfSaIS0_EE3endEv
	movq	%rax, %rbx
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt6vectorIPfSaIS0_EE5beginEv
	movq	%rax, %r14
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt6vectorIPfSaIS0_EE3endEv
	movq	%rax, %r13
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt6vectorIPfSaIS0_EE5beginEv
	movq	%r14, %rdx
	movq	%r13, %rsi
	movq	%rax, %rdi
	call	_ZSt4copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEENS1_IPS2_S7_EEET0_T_SC_SB_
	movq	%r12, %rdx
	movq	%rbx, %rsi
	movq	%rax, %rdi
	call	_ZSt8_DestroyIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS2_SaIS2_EEEES2_EvT_S8_RSaIT0_E
	jmp	.L211
.L212:
	movq	-56(%rbp), %rax
	movq	(%rax), %rbx
	movq	-64(%rbp), %rax
	movq	(%rax), %r12
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt6vectorIPfSaIS0_EE4sizeEv
	salq	$3, %rax
	leaq	(%r12,%rax), %rcx
	movq	-64(%rbp), %rax
	movq	(%rax), %rax
	movq	%rbx, %rdx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZSt4copyIPPfS1_ET0_T_S3_S2_
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIPfSaIS0_EE19_M_get_Tp_allocatorEv
	movq	%rax, %r13
	movq	-56(%rbp), %rax
	movq	8(%rax), %r12
	movq	-64(%rbp), %rax
	movq	8(%rax), %rbx
	movq	-64(%rbp), %rax
	movq	(%rax), %r14
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt6vectorIPfSaIS0_EE4sizeEv
	salq	$3, %rax
	addq	%r14, %rax
	movq	%r13, %rcx
	movq	%r12, %rdx
	movq	%rbx, %rsi
	movq	%rax, %rdi
	call	_ZSt22__uninitialized_copy_aIPPfS1_S0_ET0_T_S3_S2_RSaIT1_E
.L211:
	movq	-56(%rbp), %rax
	movq	(%rax), %rax
	movq	-48(%rbp), %rdx
	salq	$3, %rdx
	addq	%rax, %rdx
	movq	-56(%rbp), %rax
	movq	%rdx, 8(%rax)
.L205:
	movq	-56(%rbp), %rax
	addq	$32, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1093:
	.size	_ZNSt6vectorIPfSaIS0_EEaSERKS2_, .-_ZNSt6vectorIPfSaIS0_EEaSERKS2_
	.section	.text._ZSt7forwardIRKSt6vectorIPfSaIS1_EEEOT_RNSt16remove_referenceIS6_E4typeE,"axG",@progbits,_ZSt7forwardIRKSt6vectorIPfSaIS1_EEEOT_RNSt16remove_referenceIS6_E4typeE,comdat
	.weak	_ZSt7forwardIRKSt6vectorIPfSaIS1_EEEOT_RNSt16remove_referenceIS6_E4typeE
	.type	_ZSt7forwardIRKSt6vectorIPfSaIS1_EEEOT_RNSt16remove_referenceIS6_E4typeE, @function
_ZSt7forwardIRKSt6vectorIPfSaIS1_EEEOT_RNSt16remove_referenceIS6_E4typeE:
.LFB1096:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1096:
	.size	_ZSt7forwardIRKSt6vectorIPfSaIS1_EEEOT_RNSt16remove_referenceIS6_E4typeE, .-_ZSt7forwardIRKSt6vectorIPfSaIS1_EEEOT_RNSt16remove_referenceIS6_E4typeE
	.section	.text._ZNSt6vectorIPfSaIS0_EEC2ERKS2_,"axG",@progbits,_ZNSt6vectorIPfSaIS0_EEC5ERKS2_,comdat
	.align 2
	.weak	_ZNSt6vectorIPfSaIS0_EEC2ERKS2_
	.type	_ZNSt6vectorIPfSaIS0_EEC2ERKS2_, @function
_ZNSt6vectorIPfSaIS0_EEC2ERKS2_:
.LFB1098:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA1098
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$40, %rsp
	.cfi_offset 13, -24
	.cfi_offset 12, -32
	.cfi_offset 3, -40
	movq	%rdi, -56(%rbp)
	movq	%rsi, -64(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -40(%rbp)
	xorl	%eax, %eax
	movq	-56(%rbp), %rbx
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt12_Vector_baseIPfSaIS0_EE19_M_get_Tp_allocatorEv
	movq	%rax, %rdx
	leaq	-41(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
.LEHB11:
	call	_ZN9__gnu_cxx14__alloc_traitsISaIPfES1_E17_S_select_on_copyERKS2_
.LEHE11:
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt6vectorIPfSaIS0_EE4sizeEv
	movq	%rax, %rcx
	leaq	-41(%rbp), %rax
	movq	%rax, %rdx
	movq	%rcx, %rsi
	movq	%rbx, %rdi
.LEHB12:
	call	_ZNSt12_Vector_baseIPfSaIS0_EEC2EmRKS1_
.LEHE12:
	leaq	-41(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSaIPfED1Ev
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIPfSaIS0_EE19_M_get_Tp_allocatorEv
	movq	%rax, %r13
	movq	-56(%rbp), %rax
	movq	(%rax), %rbx
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt6vectorIPfSaIS0_EE3endEv
	movq	%rax, %r12
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNKSt6vectorIPfSaIS0_EE5beginEv
	movq	%r13, %rcx
	movq	%rbx, %rdx
	movq	%r12, %rsi
	movq	%rax, %rdi
.LEHB13:
	call	_ZSt22__uninitialized_copy_aIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_S2_ET0_T_SB_SA_RSaIT1_E
.LEHE13:
	movq	-56(%rbp), %rdx
	movq	%rax, 8(%rdx)
	nop
	movq	-40(%rbp), %rax
	xorq	%fs:40, %rax
	je	.L219
	jmp	.L222
.L220:
	endbr64
	movq	%rax, %rbx
	leaq	-41(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSaIPfED1Ev
	movq	%rbx, %rax
	movq	%rax, %rdi
.LEHB14:
	call	_Unwind_Resume@PLT
.L221:
	endbr64
	movq	%rax, %rbx
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIPfSaIS0_EED2Ev
	movq	%rbx, %rax
	movq	%rax, %rdi
	call	_Unwind_Resume@PLT
.LEHE14:
.L222:
	call	__stack_chk_fail@PLT
.L219:
	addq	$40, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1098:
	.section	.gcc_except_table
.LLSDA1098:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE1098-.LLSDACSB1098
.LLSDACSB1098:
	.uleb128 .LEHB11-.LFB1098
	.uleb128 .LEHE11-.LEHB11
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB12-.LFB1098
	.uleb128 .LEHE12-.LEHB12
	.uleb128 .L220-.LFB1098
	.uleb128 0
	.uleb128 .LEHB13-.LFB1098
	.uleb128 .LEHE13-.LEHB13
	.uleb128 .L221-.LFB1098
	.uleb128 0
	.uleb128 .LEHB14-.LFB1098
	.uleb128 .LEHE14-.LEHB14
	.uleb128 0
	.uleb128 0
.LLSDACSE1098:
	.section	.text._ZNSt6vectorIPfSaIS0_EEC2ERKS2_,"axG",@progbits,_ZNSt6vectorIPfSaIS0_EEC5ERKS2_,comdat
	.size	_ZNSt6vectorIPfSaIS0_EEC2ERKS2_, .-_ZNSt6vectorIPfSaIS0_EEC2ERKS2_
	.weak	_ZNSt6vectorIPfSaIS0_EEC1ERKS2_
	.set	_ZNSt6vectorIPfSaIS0_EEC1ERKS2_,_ZNSt6vectorIPfSaIS0_EEC2ERKS2_
	.section	.text._ZNKSt12_Vector_baseIPfSaIS0_EE19_M_get_Tp_allocatorEv,"axG",@progbits,_ZNKSt12_Vector_baseIPfSaIS0_EE19_M_get_Tp_allocatorEv,comdat
	.align 2
	.weak	_ZNKSt12_Vector_baseIPfSaIS0_EE19_M_get_Tp_allocatorEv
	.type	_ZNKSt12_Vector_baseIPfSaIS0_EE19_M_get_Tp_allocatorEv, @function
_ZNKSt12_Vector_baseIPfSaIS0_EE19_M_get_Tp_allocatorEv:
.LFB1100:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1100:
	.size	_ZNKSt12_Vector_baseIPfSaIS0_EE19_M_get_Tp_allocatorEv, .-_ZNKSt12_Vector_baseIPfSaIS0_EE19_M_get_Tp_allocatorEv
	.section	.text._ZStneRKSaIPfES2_,"axG",@progbits,_ZStneRKSaIPfES2_,comdat
	.weak	_ZStneRKSaIPfES2_
	.type	_ZStneRKSaIPfES2_, @function
_ZStneRKSaIPfES2_:
.LFB1101:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movl	$0, %eax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1101:
	.size	_ZStneRKSaIPfES2_, .-_ZStneRKSaIPfES2_
	.section	.text._ZNSt6vectorIPfSaIS0_EE5clearEv,"axG",@progbits,_ZNSt6vectorIPfSaIS0_EE5clearEv,comdat
	.align 2
	.weak	_ZNSt6vectorIPfSaIS0_EE5clearEv
	.type	_ZNSt6vectorIPfSaIS0_EE5clearEv, @function
_ZNSt6vectorIPfSaIS0_EE5clearEv:
.LFB1102:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	(%rax), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSt6vectorIPfSaIS0_EE15_M_erase_at_endEPS0_
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1102:
	.size	_ZNSt6vectorIPfSaIS0_EE5clearEv, .-_ZNSt6vectorIPfSaIS0_EE5clearEv
	.section	.text._ZSt15__alloc_on_copyISaIPfEEvRT_RKS2_,"axG",@progbits,_ZSt15__alloc_on_copyISaIPfEEvRT_RKS2_,comdat
	.weak	_ZSt15__alloc_on_copyISaIPfEEvRT_RKS2_
	.type	_ZSt15__alloc_on_copyISaIPfEEvRT_RKS2_, @function
_ZSt15__alloc_on_copyISaIPfEEvRT_RKS2_:
.LFB1103:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	-32(%rbp), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZSt18__do_alloc_on_copyISaIPfEEvRT_RKS2_St17integral_constantIbLb0EE
	nop
	movq	-8(%rbp), %rax
	xorq	%fs:40, %rax
	je	.L229
	call	__stack_chk_fail@PLT
.L229:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1103:
	.size	_ZSt15__alloc_on_copyISaIPfEEvRT_RKS2_, .-_ZSt15__alloc_on_copyISaIPfEEvRT_RKS2_
	.section	.text._ZNKSt6vectorIPfSaIS0_EE4sizeEv,"axG",@progbits,_ZNKSt6vectorIPfSaIS0_EE4sizeEv,comdat
	.align 2
	.weak	_ZNKSt6vectorIPfSaIS0_EE4sizeEv
	.type	_ZNKSt6vectorIPfSaIS0_EE4sizeEv, @function
_ZNKSt6vectorIPfSaIS0_EE4sizeEv:
.LFB1104:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	8(%rax), %rdx
	movq	-8(%rbp), %rax
	movq	(%rax), %rax
	subq	%rax, %rdx
	movq	%rdx, %rax
	sarq	$3, %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1104:
	.size	_ZNKSt6vectorIPfSaIS0_EE4sizeEv, .-_ZNKSt6vectorIPfSaIS0_EE4sizeEv
	.section	.text._ZNKSt6vectorIPfSaIS0_EE8capacityEv,"axG",@progbits,_ZNKSt6vectorIPfSaIS0_EE8capacityEv,comdat
	.align 2
	.weak	_ZNKSt6vectorIPfSaIS0_EE8capacityEv
	.type	_ZNKSt6vectorIPfSaIS0_EE8capacityEv, @function
_ZNKSt6vectorIPfSaIS0_EE8capacityEv:
.LFB1105:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	16(%rax), %rdx
	movq	-8(%rbp), %rax
	movq	(%rax), %rax
	subq	%rax, %rdx
	movq	%rdx, %rax
	sarq	$3, %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1105:
	.size	_ZNKSt6vectorIPfSaIS0_EE8capacityEv, .-_ZNKSt6vectorIPfSaIS0_EE8capacityEv
	.section	.text._ZNKSt6vectorIPfSaIS0_EE5beginEv,"axG",@progbits,_ZNKSt6vectorIPfSaIS0_EE5beginEv,comdat
	.align 2
	.weak	_ZNKSt6vectorIPfSaIS0_EE5beginEv
	.type	_ZNKSt6vectorIPfSaIS0_EE5beginEv, @function
_ZNKSt6vectorIPfSaIS0_EE5beginEv:
.LFB1106:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	-40(%rbp), %rax
	movq	(%rax), %rax
	movq	%rax, -24(%rbp)
	leaq	-24(%rbp), %rdx
	leaq	-16(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS1_SaIS1_EEEC1ERKS3_
	movq	-16(%rbp), %rax
	movq	-8(%rbp), %rcx
	xorq	%fs:40, %rcx
	je	.L236
	call	__stack_chk_fail@PLT
.L236:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1106:
	.size	_ZNKSt6vectorIPfSaIS0_EE5beginEv, .-_ZNKSt6vectorIPfSaIS0_EE5beginEv
	.section	.text._ZNKSt6vectorIPfSaIS0_EE3endEv,"axG",@progbits,_ZNKSt6vectorIPfSaIS0_EE3endEv,comdat
	.align 2
	.weak	_ZNKSt6vectorIPfSaIS0_EE3endEv
	.type	_ZNKSt6vectorIPfSaIS0_EE3endEv, @function
_ZNKSt6vectorIPfSaIS0_EE3endEv:
.LFB1107:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	-40(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -24(%rbp)
	leaq	-24(%rbp), %rdx
	leaq	-16(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS1_SaIS1_EEEC1ERKS3_
	movq	-16(%rbp), %rax
	movq	-8(%rbp), %rcx
	xorq	%fs:40, %rcx
	je	.L239
	call	__stack_chk_fail@PLT
.L239:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1107:
	.size	_ZNKSt6vectorIPfSaIS0_EE3endEv, .-_ZNKSt6vectorIPfSaIS0_EE3endEv
	.section	.text._ZNSt6vectorIPfSaIS0_EE20_M_allocate_and_copyIN9__gnu_cxx17__normal_iteratorIPKS0_S2_EEEEPS0_mT_SA_,"axG",@progbits,_ZNSt6vectorIPfSaIS0_EE20_M_allocate_and_copyIN9__gnu_cxx17__normal_iteratorIPKS0_S2_EEEEPS0_mT_SA_,comdat
	.align 2
	.weak	_ZNSt6vectorIPfSaIS0_EE20_M_allocate_and_copyIN9__gnu_cxx17__normal_iteratorIPKS0_S2_EEEEPS0_mT_SA_
	.type	_ZNSt6vectorIPfSaIS0_EE20_M_allocate_and_copyIN9__gnu_cxx17__normal_iteratorIPKS0_S2_EEEEPS0_mT_SA_, @function
_ZNSt6vectorIPfSaIS0_EE20_M_allocate_and_copyIN9__gnu_cxx17__normal_iteratorIPKS0_S2_EEEEPS0_mT_SA_:
.LFB1108:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA1108
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$56, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%rcx, -64(%rbp)
	movq	-40(%rbp), %rax
	movq	-48(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
.LEHB15:
	call	_ZNSt12_Vector_baseIPfSaIS0_EE11_M_allocateEm
.LEHE15:
	movq	%rax, -24(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIPfSaIS0_EE19_M_get_Tp_allocatorEv
	movq	%rax, %rcx
	movq	-24(%rbp), %rdx
	movq	-64(%rbp), %rsi
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
.LEHB16:
	call	_ZSt22__uninitialized_copy_aIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_S2_ET0_T_SB_SA_RSaIT1_E
.LEHE16:
	movq	-24(%rbp), %rax
	jmp	.L246
.L244:
	endbr64
	movq	%rax, %rdi
	call	__cxa_begin_catch@PLT
	movq	-40(%rbp), %rax
	movq	-48(%rbp), %rdx
	movq	-24(%rbp), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
.LEHB17:
	call	_ZNSt12_Vector_baseIPfSaIS0_EE13_M_deallocateEPS0_m
	call	__cxa_rethrow@PLT
.LEHE17:
.L245:
	endbr64
	movq	%rax, %rbx
	call	__cxa_end_catch@PLT
	movq	%rbx, %rax
	movq	%rax, %rdi
.LEHB18:
	call	_Unwind_Resume@PLT
.LEHE18:
.L246:
	addq	$56, %rsp
	popq	%rbx
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1108:
	.section	.gcc_except_table
	.align 4
.LLSDA1108:
	.byte	0xff
	.byte	0x9b
	.uleb128 .LLSDATT1108-.LLSDATTD1108
.LLSDATTD1108:
	.byte	0x1
	.uleb128 .LLSDACSE1108-.LLSDACSB1108
.LLSDACSB1108:
	.uleb128 .LEHB15-.LFB1108
	.uleb128 .LEHE15-.LEHB15
	.uleb128 0
	.uleb128 0
	.uleb128 .LEHB16-.LFB1108
	.uleb128 .LEHE16-.LEHB16
	.uleb128 .L244-.LFB1108
	.uleb128 0x1
	.uleb128 .LEHB17-.LFB1108
	.uleb128 .LEHE17-.LEHB17
	.uleb128 .L245-.LFB1108
	.uleb128 0
	.uleb128 .LEHB18-.LFB1108
	.uleb128 .LEHE18-.LEHB18
	.uleb128 0
	.uleb128 0
.LLSDACSE1108:
	.byte	0x1
	.byte	0
	.align 4
	.long	0

.LLSDATT1108:
	.section	.text._ZNSt6vectorIPfSaIS0_EE20_M_allocate_and_copyIN9__gnu_cxx17__normal_iteratorIPKS0_S2_EEEEPS0_mT_SA_,"axG",@progbits,_ZNSt6vectorIPfSaIS0_EE20_M_allocate_and_copyIN9__gnu_cxx17__normal_iteratorIPKS0_S2_EEEEPS0_mT_SA_,comdat
	.size	_ZNSt6vectorIPfSaIS0_EE20_M_allocate_and_copyIN9__gnu_cxx17__normal_iteratorIPKS0_S2_EEEEPS0_mT_SA_, .-_ZNSt6vectorIPfSaIS0_EE20_M_allocate_and_copyIN9__gnu_cxx17__normal_iteratorIPKS0_S2_EEEEPS0_mT_SA_
	.section	.text._ZNSt6vectorIPfSaIS0_EE5beginEv,"axG",@progbits,_ZNSt6vectorIPfSaIS0_EE5beginEv,comdat
	.align 2
	.weak	_ZNSt6vectorIPfSaIS0_EE5beginEv
	.type	_ZNSt6vectorIPfSaIS0_EE5beginEv, @function
_ZNSt6vectorIPfSaIS0_EE5beginEv:
.LFB1109:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	-24(%rbp), %rdx
	leaq	-16(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEEC1ERKS2_
	movq	-16(%rbp), %rax
	movq	-8(%rbp), %rcx
	xorq	%fs:40, %rcx
	je	.L249
	call	__stack_chk_fail@PLT
.L249:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1109:
	.size	_ZNSt6vectorIPfSaIS0_EE5beginEv, .-_ZNSt6vectorIPfSaIS0_EE5beginEv
	.section	.text._ZSt4copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEENS1_IPS2_S7_EEET0_T_SC_SB_,"axG",@progbits,_ZSt4copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEENS1_IPS2_S7_EEET0_T_SC_SB_,comdat
	.weak	_ZSt4copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEENS1_IPS2_S7_EEET0_T_SC_SB_
	.type	_ZSt4copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEENS1_IPS2_S7_EEET0_T_SC_SB_, @function
_ZSt4copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEENS1_IPS2_S7_EEET0_T_SC_SB_:
.LFB1110:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$40, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__miter_baseIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEET_S9_
	movq	%rax, %rbx
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__miter_baseIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEET_S9_
	movq	%rax, %rcx
	movq	-40(%rbp), %rax
	movq	%rax, %rdx
	movq	%rbx, %rsi
	movq	%rcx, %rdi
	call	_ZSt14__copy_move_a2ILb0EN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEENS1_IPS2_S7_EEET1_T0_SC_SB_
	addq	$40, %rsp
	popq	%rbx
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1110:
	.size	_ZSt4copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEENS1_IPS2_S7_EEET0_T_SC_SB_, .-_ZSt4copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEENS1_IPS2_S7_EEET0_T_SC_SB_
	.section	.text._ZNSt6vectorIPfSaIS0_EE3endEv,"axG",@progbits,_ZNSt6vectorIPfSaIS0_EE3endEv,comdat
	.align 2
	.weak	_ZNSt6vectorIPfSaIS0_EE3endEv
	.type	_ZNSt6vectorIPfSaIS0_EE3endEv, @function
_ZNSt6vectorIPfSaIS0_EE3endEv:
.LFB1111:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	-24(%rbp), %rax
	leaq	8(%rax), %rdx
	leaq	-16(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEEC1ERKS2_
	movq	-16(%rbp), %rax
	movq	-8(%rbp), %rcx
	xorq	%fs:40, %rcx
	je	.L254
	call	__stack_chk_fail@PLT
.L254:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1111:
	.size	_ZNSt6vectorIPfSaIS0_EE3endEv, .-_ZNSt6vectorIPfSaIS0_EE3endEv
	.section	.text._ZSt8_DestroyIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS2_SaIS2_EEEES2_EvT_S8_RSaIT0_E,"axG",@progbits,_ZSt8_DestroyIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS2_SaIS2_EEEES2_EvT_S8_RSaIT0_E,comdat
	.weak	_ZSt8_DestroyIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS2_SaIS2_EEEES2_EvT_S8_RSaIT0_E
	.type	_ZSt8_DestroyIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS2_SaIS2_EEEES2_EvT_S8_RSaIT0_E, @function
_ZSt8_DestroyIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS2_SaIS2_EEEES2_EvT_S8_RSaIT0_E:
.LFB1112:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-16(%rbp), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZSt8_DestroyIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS2_SaIS2_EEEEEvT_S8_
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1112:
	.size	_ZSt8_DestroyIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS2_SaIS2_EEEES2_EvT_S8_RSaIT0_E, .-_ZSt8_DestroyIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS2_SaIS2_EEEES2_EvT_S8_RSaIT0_E
	.section	.text._ZSt4copyIPPfS1_ET0_T_S3_S2_,"axG",@progbits,_ZSt4copyIPPfS1_ET0_T_S3_S2_,comdat
	.weak	_ZSt4copyIPPfS1_ET0_T_S3_S2_
	.type	_ZSt4copyIPPfS1_ET0_T_S3_S2_, @function
_ZSt4copyIPPfS1_ET0_T_S3_S2_:
.LFB1113:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$40, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__miter_baseIPPfET_S2_
	movq	%rax, %rbx
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__miter_baseIPPfET_S2_
	movq	%rax, %rcx
	movq	-40(%rbp), %rax
	movq	%rax, %rdx
	movq	%rbx, %rsi
	movq	%rcx, %rdi
	call	_ZSt14__copy_move_a2ILb0EPPfS1_ET1_T0_S3_S2_
	addq	$40, %rsp
	popq	%rbx
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1113:
	.size	_ZSt4copyIPPfS1_ET0_T_S3_S2_, .-_ZSt4copyIPPfS1_ET0_T_S3_S2_
	.section	.text._ZSt22__uninitialized_copy_aIPPfS1_S0_ET0_T_S3_S2_RSaIT1_E,"axG",@progbits,_ZSt22__uninitialized_copy_aIPPfS1_S0_ET0_T_S3_S2_RSaIT1_E,comdat
	.weak	_ZSt22__uninitialized_copy_aIPPfS1_S0_ET0_T_S3_S2_RSaIT1_E
	.type	_ZSt22__uninitialized_copy_aIPPfS1_S0_ET0_T_S3_S2_RSaIT1_E, @function
_ZSt22__uninitialized_copy_aIPPfS1_S0_ET0_T_S3_S2_RSaIT1_E:
.LFB1114:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	%rcx, -32(%rbp)
	movq	-24(%rbp), %rdx
	movq	-16(%rbp), %rcx
	movq	-8(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZSt18uninitialized_copyIPPfS1_ET0_T_S3_S2_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1114:
	.size	_ZSt22__uninitialized_copy_aIPPfS1_S0_ET0_T_S3_S2_RSaIT1_E, .-_ZSt22__uninitialized_copy_aIPPfS1_S0_ET0_T_S3_S2_RSaIT1_E
	.section	.text._ZN9__gnu_cxx14__alloc_traitsISaIPfES1_E17_S_select_on_copyERKS2_,"axG",@progbits,_ZN9__gnu_cxx14__alloc_traitsISaIPfES1_E17_S_select_on_copyERKS2_,comdat
	.weak	_ZN9__gnu_cxx14__alloc_traitsISaIPfES1_E17_S_select_on_copyERKS2_
	.type	_ZN9__gnu_cxx14__alloc_traitsISaIPfES1_E17_S_select_on_copyERKS2_, @function
_ZN9__gnu_cxx14__alloc_traitsISaIPfES1_E17_S_select_on_copyERKS2_:
.LFB1115:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	-24(%rbp), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSt16allocator_traitsISaIPfEE37select_on_container_copy_constructionERKS1_
	movq	-8(%rbp), %rax
	xorq	%fs:40, %rax
	je	.L262
	call	__stack_chk_fail@PLT
.L262:
	movq	-24(%rbp), %rax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1115:
	.size	_ZN9__gnu_cxx14__alloc_traitsISaIPfES1_E17_S_select_on_copyERKS2_, .-_ZN9__gnu_cxx14__alloc_traitsISaIPfES1_E17_S_select_on_copyERKS2_
	.section	.text._ZNSt12_Vector_baseIPfSaIS0_EEC2EmRKS1_,"axG",@progbits,_ZNSt12_Vector_baseIPfSaIS0_EEC5EmRKS1_,comdat
	.align 2
	.weak	_ZNSt12_Vector_baseIPfSaIS0_EEC2EmRKS1_
	.type	_ZNSt12_Vector_baseIPfSaIS0_EEC2EmRKS1_, @function
_ZNSt12_Vector_baseIPfSaIS0_EEC2EmRKS1_:
.LFB1117:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA1117
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$40, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	-24(%rbp), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implC1ERKS1_
	movq	-32(%rbp), %rdx
	movq	-24(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
.LEHB19:
	call	_ZNSt12_Vector_baseIPfSaIS0_EE17_M_create_storageEm
.LEHE19:
	jmp	.L266
.L265:
	endbr64
	movq	%rax, %rbx
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implD1Ev
	movq	%rbx, %rax
	movq	%rax, %rdi
.LEHB20:
	call	_Unwind_Resume@PLT
.LEHE20:
.L266:
	addq	$40, %rsp
	popq	%rbx
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1117:
	.section	.gcc_except_table
.LLSDA1117:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE1117-.LLSDACSB1117
.LLSDACSB1117:
	.uleb128 .LEHB19-.LFB1117
	.uleb128 .LEHE19-.LEHB19
	.uleb128 .L265-.LFB1117
	.uleb128 0
	.uleb128 .LEHB20-.LFB1117
	.uleb128 .LEHE20-.LEHB20
	.uleb128 0
	.uleb128 0
.LLSDACSE1117:
	.section	.text._ZNSt12_Vector_baseIPfSaIS0_EEC2EmRKS1_,"axG",@progbits,_ZNSt12_Vector_baseIPfSaIS0_EEC5EmRKS1_,comdat
	.size	_ZNSt12_Vector_baseIPfSaIS0_EEC2EmRKS1_, .-_ZNSt12_Vector_baseIPfSaIS0_EEC2EmRKS1_
	.weak	_ZNSt12_Vector_baseIPfSaIS0_EEC1EmRKS1_
	.set	_ZNSt12_Vector_baseIPfSaIS0_EEC1EmRKS1_,_ZNSt12_Vector_baseIPfSaIS0_EEC2EmRKS1_
	.section	.text._ZSt22__uninitialized_copy_aIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_S2_ET0_T_SB_SA_RSaIT1_E,"axG",@progbits,_ZSt22__uninitialized_copy_aIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_S2_ET0_T_SB_SA_RSaIT1_E,comdat
	.weak	_ZSt22__uninitialized_copy_aIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_S2_ET0_T_SB_SA_RSaIT1_E
	.type	_ZSt22__uninitialized_copy_aIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_S2_ET0_T_SB_SA_RSaIT1_E, @function
_ZSt22__uninitialized_copy_aIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_S2_ET0_T_SB_SA_RSaIT1_E:
.LFB1119:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	%rcx, -32(%rbp)
	movq	-24(%rbp), %rdx
	movq	-16(%rbp), %rcx
	movq	-8(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZSt18uninitialized_copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_ET0_T_SB_SA_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1119:
	.size	_ZSt22__uninitialized_copy_aIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_S2_ET0_T_SB_SA_RSaIT1_E, .-_ZSt22__uninitialized_copy_aIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_S2_ET0_T_SB_SA_RSaIT1_E
	.section	.text._ZNSt6vectorIPfSaIS0_EE15_M_erase_at_endEPS0_,"axG",@progbits,_ZNSt6vectorIPfSaIS0_EE15_M_erase_at_endEPS0_,comdat
	.align 2
	.weak	_ZNSt6vectorIPfSaIS0_EE15_M_erase_at_endEPS0_
	.type	_ZNSt6vectorIPfSaIS0_EE15_M_erase_at_endEPS0_, @function
_ZNSt6vectorIPfSaIS0_EE15_M_erase_at_endEPS0_:
.LFB1120:
	.cfi_startproc
	.cfi_personality 0x9b,DW.ref.__gxx_personality_v0
	.cfi_lsda 0x1b,.LLSDA1120
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	subq	-32(%rbp), %rax
	sarq	$3, %rax
	movq	%rax, -8(%rbp)
	cmpq	$0, -8(%rbp)
	je	.L271
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIPfSaIS0_EE19_M_get_Tp_allocatorEv
	movq	%rax, %rdx
	movq	-24(%rbp), %rax
	movq	8(%rax), %rcx
	movq	-32(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZSt8_DestroyIPPfS0_EvT_S2_RSaIT0_E
	movq	-24(%rbp), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, 8(%rax)
.L271:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1120:
	.section	.gcc_except_table
.LLSDA1120:
	.byte	0xff
	.byte	0xff
	.byte	0x1
	.uleb128 .LLSDACSE1120-.LLSDACSB1120
.LLSDACSB1120:
.LLSDACSE1120:
	.section	.text._ZNSt6vectorIPfSaIS0_EE15_M_erase_at_endEPS0_,"axG",@progbits,_ZNSt6vectorIPfSaIS0_EE15_M_erase_at_endEPS0_,comdat
	.size	_ZNSt6vectorIPfSaIS0_EE15_M_erase_at_endEPS0_, .-_ZNSt6vectorIPfSaIS0_EE15_M_erase_at_endEPS0_
	.section	.text._ZSt18__do_alloc_on_copyISaIPfEEvRT_RKS2_St17integral_constantIbLb0EE,"axG",@progbits,_ZSt18__do_alloc_on_copyISaIPfEEvRT_RKS2_St17integral_constantIbLb0EE,comdat
	.weak	_ZSt18__do_alloc_on_copyISaIPfEEvRT_RKS2_St17integral_constantIbLb0EE
	.type	_ZSt18__do_alloc_on_copyISaIPfEEvRT_RKS2_St17integral_constantIbLb0EE, @function
_ZSt18__do_alloc_on_copyISaIPfEEvRT_RKS2_St17integral_constantIbLb0EE:
.LFB1121:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1121:
	.size	_ZSt18__do_alloc_on_copyISaIPfEEvRT_RKS2_St17integral_constantIbLb0EE, .-_ZSt18__do_alloc_on_copyISaIPfEEvRT_RKS2_St17integral_constantIbLb0EE
	.section	.text._ZN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS1_SaIS1_EEEC2ERKS3_,"axG",@progbits,_ZN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS1_SaIS1_EEEC5ERKS3_,comdat
	.align 2
	.weak	_ZN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS1_SaIS1_EEEC2ERKS3_
	.type	_ZN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS1_SaIS1_EEEC2ERKS3_, @function
_ZN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS1_SaIS1_EEEC2ERKS3_:
.LFB1123:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	(%rax), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, (%rax)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1123:
	.size	_ZN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS1_SaIS1_EEEC2ERKS3_, .-_ZN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS1_SaIS1_EEEC2ERKS3_
	.weak	_ZN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS1_SaIS1_EEEC1ERKS3_
	.set	_ZN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS1_SaIS1_EEEC1ERKS3_,_ZN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS1_SaIS1_EEEC2ERKS3_
	.section	.text._ZNSt12_Vector_baseIPfSaIS0_EE11_M_allocateEm,"axG",@progbits,_ZNSt12_Vector_baseIPfSaIS0_EE11_M_allocateEm,comdat
	.align 2
	.weak	_ZNSt12_Vector_baseIPfSaIS0_EE11_M_allocateEm
	.type	_ZNSt12_Vector_baseIPfSaIS0_EE11_M_allocateEm, @function
_ZNSt12_Vector_baseIPfSaIS0_EE11_M_allocateEm:
.LFB1125:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	cmpq	$0, -16(%rbp)
	je	.L275
	movq	-8(%rbp), %rax
	movq	-16(%rbp), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSt16allocator_traitsISaIPfEE8allocateERS1_m
	jmp	.L277
.L275:
	movl	$0, %eax
.L277:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1125:
	.size	_ZNSt12_Vector_baseIPfSaIS0_EE11_M_allocateEm, .-_ZNSt12_Vector_baseIPfSaIS0_EE11_M_allocateEm
	.section	.text._ZN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEEC2ERKS2_,"axG",@progbits,_ZN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEEC5ERKS2_,comdat
	.align 2
	.weak	_ZN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEEC2ERKS2_
	.type	_ZN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEEC2ERKS2_, @function
_ZN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEEC2ERKS2_:
.LFB1127:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	(%rax), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, (%rax)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1127:
	.size	_ZN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEEC2ERKS2_, .-_ZN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEEC2ERKS2_
	.weak	_ZN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEEC1ERKS2_
	.set	_ZN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEEC1ERKS2_,_ZN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEEC2ERKS2_
	.section	.text._ZSt12__miter_baseIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEET_S9_,"axG",@progbits,_ZSt12__miter_baseIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEET_S9_,comdat
	.weak	_ZSt12__miter_baseIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEET_S9_
	.type	_ZSt12__miter_baseIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEET_S9_, @function
_ZSt12__miter_baseIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEET_S9_:
.LFB1129:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1129:
	.size	_ZSt12__miter_baseIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEET_S9_, .-_ZSt12__miter_baseIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEET_S9_
	.section	.text._ZSt14__copy_move_a2ILb0EN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEENS1_IPS2_S7_EEET1_T0_SC_SB_,"axG",@progbits,_ZSt14__copy_move_a2ILb0EN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEENS1_IPS2_S7_EEET1_T0_SC_SB_,comdat
	.weak	_ZSt14__copy_move_a2ILb0EN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEENS1_IPS2_S7_EEET1_T0_SC_SB_
	.type	_ZSt14__copy_move_a2ILb0EN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEENS1_IPS2_S7_EEET1_T0_SC_SB_, @function
_ZSt14__copy_move_a2ILb0EN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEENS1_IPS2_S7_EEET1_T0_SC_SB_:
.LFB1130:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r12
	pushq	%rbx
	subq	$32, %rsp
	.cfi_offset 12, -24
	.cfi_offset 3, -32
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__niter_baseIPPfSt6vectorIS0_SaIS0_EEET_N9__gnu_cxx17__normal_iteratorIS5_T0_EE
	movq	%rax, %r12
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__niter_baseIPKPfSt6vectorIS0_SaIS0_EEET_N9__gnu_cxx17__normal_iteratorIS6_T0_EE
	movq	%rax, %rbx
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__niter_baseIPKPfSt6vectorIS0_SaIS0_EEET_N9__gnu_cxx17__normal_iteratorIS6_T0_EE
	movq	%r12, %rdx
	movq	%rbx, %rsi
	movq	%rax, %rdi
	call	_ZSt13__copy_move_aILb0EPKPfPS0_ET1_T0_S5_S4_
	movq	%rax, %rdx
	movq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZSt12__niter_wrapIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS2_SaIS2_EEEES3_ET_S8_T0_
	addq	$32, %rsp
	popq	%rbx
	popq	%r12
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1130:
	.size	_ZSt14__copy_move_a2ILb0EN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEENS1_IPS2_S7_EEET1_T0_SC_SB_, .-_ZSt14__copy_move_a2ILb0EN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEENS1_IPS2_S7_EEET1_T0_SC_SB_
	.section	.text._ZSt8_DestroyIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS2_SaIS2_EEEEEvT_S8_,"axG",@progbits,_ZSt8_DestroyIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS2_SaIS2_EEEEEvT_S8_,comdat
	.weak	_ZSt8_DestroyIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS2_SaIS2_EEEEEvT_S8_
	.type	_ZSt8_DestroyIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS2_SaIS2_EEEEEvT_S8_, @function
_ZSt8_DestroyIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS2_SaIS2_EEEEEvT_S8_:
.LFB1131:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSt12_Destroy_auxILb1EE9__destroyIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS4_SaIS4_EEEEEEvT_SA_
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1131:
	.size	_ZSt8_DestroyIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS2_SaIS2_EEEEEvT_S8_, .-_ZSt8_DestroyIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS2_SaIS2_EEEEEvT_S8_
	.section	.text._ZSt12__miter_baseIPPfET_S2_,"axG",@progbits,_ZSt12__miter_baseIPPfET_S2_,comdat
	.weak	_ZSt12__miter_baseIPPfET_S2_
	.type	_ZSt12__miter_baseIPPfET_S2_, @function
_ZSt12__miter_baseIPPfET_S2_:
.LFB1132:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1132:
	.size	_ZSt12__miter_baseIPPfET_S2_, .-_ZSt12__miter_baseIPPfET_S2_
	.section	.text._ZSt14__copy_move_a2ILb0EPPfS1_ET1_T0_S3_S2_,"axG",@progbits,_ZSt14__copy_move_a2ILb0EPPfS1_ET1_T0_S3_S2_,comdat
	.weak	_ZSt14__copy_move_a2ILb0EPPfS1_ET1_T0_S3_S2_
	.type	_ZSt14__copy_move_a2ILb0EPPfS1_ET1_T0_S3_S2_, @function
_ZSt14__copy_move_a2ILb0EPPfS1_ET1_T0_S3_S2_:
.LFB1133:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r12
	pushq	%rbx
	subq	$32, %rsp
	.cfi_offset 12, -24
	.cfi_offset 3, -32
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__niter_baseIPPfET_S2_
	movq	%rax, %r12
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__niter_baseIPPfET_S2_
	movq	%rax, %rbx
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__niter_baseIPPfET_S2_
	movq	%r12, %rdx
	movq	%rbx, %rsi
	movq	%rax, %rdi
	call	_ZSt13__copy_move_aILb0EPPfS1_ET1_T0_S3_S2_
	movq	%rax, %rdx
	leaq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZSt12__niter_wrapIPPfET_RKS2_S2_
	addq	$32, %rsp
	popq	%rbx
	popq	%r12
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1133:
	.size	_ZSt14__copy_move_a2ILb0EPPfS1_ET1_T0_S3_S2_, .-_ZSt14__copy_move_a2ILb0EPPfS1_ET1_T0_S3_S2_
	.section	.text._ZSt18uninitialized_copyIPPfS1_ET0_T_S3_S2_,"axG",@progbits,_ZSt18uninitialized_copyIPPfS1_ET0_T_S3_S2_,comdat
	.weak	_ZSt18uninitialized_copyIPPfS1_ET0_T_S3_S2_
	.type	_ZSt18uninitialized_copyIPPfS1_ET0_T_S3_S2_, @function
_ZSt18uninitialized_copyIPPfS1_ET0_T_S3_S2_:
.LFB1134:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movb	$1, -1(%rbp)
	movq	-40(%rbp), %rdx
	movq	-32(%rbp), %rcx
	movq	-24(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZNSt20__uninitialized_copyILb1EE13__uninit_copyIPPfS3_EET0_T_S5_S4_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1134:
	.size	_ZSt18uninitialized_copyIPPfS1_ET0_T_S3_S2_, .-_ZSt18uninitialized_copyIPPfS1_ET0_T_S3_S2_
	.section	.text._ZNSt16allocator_traitsISaIPfEE37select_on_container_copy_constructionERKS1_,"axG",@progbits,_ZNSt16allocator_traitsISaIPfEE37select_on_container_copy_constructionERKS1_,comdat
	.weak	_ZNSt16allocator_traitsISaIPfEE37select_on_container_copy_constructionERKS1_
	.type	_ZNSt16allocator_traitsISaIPfEE37select_on_container_copy_constructionERKS1_, @function
_ZNSt16allocator_traitsISaIPfEE37select_on_container_copy_constructionERKS1_:
.LFB1135:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSaIPfEC1ERKS0_
	movq	-8(%rbp), %rax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1135:
	.size	_ZNSt16allocator_traitsISaIPfEE37select_on_container_copy_constructionERKS1_, .-_ZNSt16allocator_traitsISaIPfEE37select_on_container_copy_constructionERKS1_
	.section	.text._ZNSaIPfEC2ERKS0_,"axG",@progbits,_ZNSaIPfEC5ERKS0_,comdat
	.align 2
	.weak	_ZNSaIPfEC2ERKS0_
	.type	_ZNSaIPfEC2ERKS0_, @function
_ZNSaIPfEC2ERKS0_:
.LFB1137:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZN9__gnu_cxx13new_allocatorIPfEC2ERKS2_
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1137:
	.size	_ZNSaIPfEC2ERKS0_, .-_ZNSaIPfEC2ERKS0_
	.weak	_ZNSaIPfEC1ERKS0_
	.set	_ZNSaIPfEC1ERKS0_,_ZNSaIPfEC2ERKS0_
	.section	.text._ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implC2ERKS1_,"axG",@progbits,_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implC5ERKS1_,comdat
	.align 2
	.weak	_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implC2ERKS1_
	.type	_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implC2ERKS1_, @function
_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implC2ERKS1_:
.LFB1140:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSaIPfEC2ERKS0_
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIPfSaIS0_EE17_Vector_impl_dataC2Ev
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1140:
	.size	_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implC2ERKS1_, .-_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implC2ERKS1_
	.weak	_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implC1ERKS1_
	.set	_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implC1ERKS1_,_ZNSt12_Vector_baseIPfSaIS0_EE12_Vector_implC2ERKS1_
	.section	.text._ZNSt12_Vector_baseIPfSaIS0_EE17_M_create_storageEm,"axG",@progbits,_ZNSt12_Vector_baseIPfSaIS0_EE17_M_create_storageEm,comdat
	.align 2
	.weak	_ZNSt12_Vector_baseIPfSaIS0_EE17_M_create_storageEm
	.type	_ZNSt12_Vector_baseIPfSaIS0_EE17_M_create_storageEm, @function
_ZNSt12_Vector_baseIPfSaIS0_EE17_M_create_storageEm:
.LFB1142:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNSt12_Vector_baseIPfSaIS0_EE11_M_allocateEm
	movq	-8(%rbp), %rdx
	movq	%rax, (%rdx)
	movq	-8(%rbp), %rax
	movq	(%rax), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, 8(%rax)
	movq	-8(%rbp), %rax
	movq	(%rax), %rax
	movq	-16(%rbp), %rdx
	salq	$3, %rdx
	addq	%rax, %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, 16(%rax)
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1142:
	.size	_ZNSt12_Vector_baseIPfSaIS0_EE17_M_create_storageEm, .-_ZNSt12_Vector_baseIPfSaIS0_EE17_M_create_storageEm
	.section	.text._ZSt18uninitialized_copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_ET0_T_SB_SA_,"axG",@progbits,_ZSt18uninitialized_copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_ET0_T_SB_SA_,comdat
	.weak	_ZSt18uninitialized_copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_ET0_T_SB_SA_
	.type	_ZSt18uninitialized_copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_ET0_T_SB_SA_, @function
_ZSt18uninitialized_copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_ET0_T_SB_SA_:
.LFB1143:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movb	$1, -1(%rbp)
	movq	-40(%rbp), %rdx
	movq	-32(%rbp), %rcx
	movq	-24(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZNSt20__uninitialized_copyILb1EE13__uninit_copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS4_SaIS4_EEEEPS4_EET0_T_SD_SC_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1143:
	.size	_ZSt18uninitialized_copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_ET0_T_SB_SA_, .-_ZSt18uninitialized_copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_ET0_T_SB_SA_
	.section	.text._ZNSt16allocator_traitsISaIPfEE8allocateERS1_m,"axG",@progbits,_ZNSt16allocator_traitsISaIPfEE8allocateERS1_m,comdat
	.weak	_ZNSt16allocator_traitsISaIPfEE8allocateERS1_m
	.type	_ZNSt16allocator_traitsISaIPfEE8allocateERS1_m, @function
_ZNSt16allocator_traitsISaIPfEE8allocateERS1_m:
.LFB1144:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rcx
	movq	-8(%rbp), %rax
	movl	$0, %edx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZN9__gnu_cxx13new_allocatorIPfE8allocateEmPKv
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1144:
	.size	_ZNSt16allocator_traitsISaIPfEE8allocateERS1_m, .-_ZNSt16allocator_traitsISaIPfEE8allocateERS1_m
	.section	.text._ZSt12__niter_baseIPKPfSt6vectorIS0_SaIS0_EEET_N9__gnu_cxx17__normal_iteratorIS6_T0_EE,"axG",@progbits,_ZSt12__niter_baseIPKPfSt6vectorIS0_SaIS0_EEET_N9__gnu_cxx17__normal_iteratorIS6_T0_EE,comdat
	.weak	_ZSt12__niter_baseIPKPfSt6vectorIS0_SaIS0_EEET_N9__gnu_cxx17__normal_iteratorIS6_T0_EE
	.type	_ZSt12__niter_baseIPKPfSt6vectorIS0_SaIS0_EEET_N9__gnu_cxx17__normal_iteratorIS6_T0_EE, @function
_ZSt12__niter_baseIPKPfSt6vectorIS0_SaIS0_EEET_N9__gnu_cxx17__normal_iteratorIS6_T0_EE:
.LFB1145:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	leaq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNK9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS1_SaIS1_EEE4baseEv
	movq	(%rax), %rax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1145:
	.size	_ZSt12__niter_baseIPKPfSt6vectorIS0_SaIS0_EEET_N9__gnu_cxx17__normal_iteratorIS6_T0_EE, .-_ZSt12__niter_baseIPKPfSt6vectorIS0_SaIS0_EEET_N9__gnu_cxx17__normal_iteratorIS6_T0_EE
	.section	.text._ZSt12__niter_baseIPPfSt6vectorIS0_SaIS0_EEET_N9__gnu_cxx17__normal_iteratorIS5_T0_EE,"axG",@progbits,_ZSt12__niter_baseIPPfSt6vectorIS0_SaIS0_EEET_N9__gnu_cxx17__normal_iteratorIS5_T0_EE,comdat
	.weak	_ZSt12__niter_baseIPPfSt6vectorIS0_SaIS0_EEET_N9__gnu_cxx17__normal_iteratorIS5_T0_EE
	.type	_ZSt12__niter_baseIPPfSt6vectorIS0_SaIS0_EEET_N9__gnu_cxx17__normal_iteratorIS5_T0_EE, @function
_ZSt12__niter_baseIPPfSt6vectorIS0_SaIS0_EEET_N9__gnu_cxx17__normal_iteratorIS5_T0_EE:
.LFB1146:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	leaq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNK9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEE4baseEv
	movq	(%rax), %rax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1146:
	.size	_ZSt12__niter_baseIPPfSt6vectorIS0_SaIS0_EEET_N9__gnu_cxx17__normal_iteratorIS5_T0_EE, .-_ZSt12__niter_baseIPPfSt6vectorIS0_SaIS0_EEET_N9__gnu_cxx17__normal_iteratorIS5_T0_EE
	.section	.text._ZSt13__copy_move_aILb0EPKPfPS0_ET1_T0_S5_S4_,"axG",@progbits,_ZSt13__copy_move_aILb0EPKPfPS0_ET1_T0_S5_S4_,comdat
	.weak	_ZSt13__copy_move_aILb0EPKPfPS0_ET1_T0_S5_S4_
	.type	_ZSt13__copy_move_aILb0EPKPfPS0_ET1_T0_S5_S4_, @function
_ZSt13__copy_move_aILb0EPKPfPS0_ET1_T0_S5_S4_:
.LFB1147:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movb	$1, -1(%rbp)
	movq	-40(%rbp), %rdx
	movq	-32(%rbp), %rcx
	movq	-24(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZNSt11__copy_moveILb0ELb1ESt26random_access_iterator_tagE8__copy_mIPfEEPT_PKS4_S7_S5_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1147:
	.size	_ZSt13__copy_move_aILb0EPKPfPS0_ET1_T0_S5_S4_, .-_ZSt13__copy_move_aILb0EPKPfPS0_ET1_T0_S5_S4_
	.section	.text._ZSt12__niter_wrapIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS2_SaIS2_EEEES3_ET_S8_T0_,"axG",@progbits,_ZSt12__niter_wrapIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS2_SaIS2_EEEES3_ET_S8_T0_,comdat
	.weak	_ZSt12__niter_wrapIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS2_SaIS2_EEEES3_ET_S8_T0_
	.type	_ZSt12__niter_wrapIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS2_SaIS2_EEEES3_ET_S8_T0_, @function
_ZSt12__niter_wrapIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS2_SaIS2_EEEES3_ET_S8_T0_:
.LFB1148:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__niter_baseIPPfSt6vectorIS0_SaIS0_EEET_N9__gnu_cxx17__normal_iteratorIS5_T0_EE
	movq	-16(%rbp), %rdx
	subq	%rax, %rdx
	movq	%rdx, %rax
	sarq	$3, %rax
	movq	%rax, %rdx
	leaq	-8(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZNK9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEEplEl
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1148:
	.size	_ZSt12__niter_wrapIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS2_SaIS2_EEEES3_ET_S8_T0_, .-_ZSt12__niter_wrapIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS2_SaIS2_EEEES3_ET_S8_T0_
	.section	.text._ZNSt12_Destroy_auxILb1EE9__destroyIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS4_SaIS4_EEEEEEvT_SA_,"axG",@progbits,_ZNSt12_Destroy_auxILb1EE9__destroyIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS4_SaIS4_EEEEEEvT_SA_,comdat
	.weak	_ZNSt12_Destroy_auxILb1EE9__destroyIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS4_SaIS4_EEEEEEvT_SA_
	.type	_ZNSt12_Destroy_auxILb1EE9__destroyIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS4_SaIS4_EEEEEEvT_SA_, @function
_ZNSt12_Destroy_auxILb1EE9__destroyIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS4_SaIS4_EEEEEEvT_SA_:
.LFB1149:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1149:
	.size	_ZNSt12_Destroy_auxILb1EE9__destroyIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS4_SaIS4_EEEEEEvT_SA_, .-_ZNSt12_Destroy_auxILb1EE9__destroyIN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS4_SaIS4_EEEEEEvT_SA_
	.section	.text._ZSt12__niter_baseIPPfET_S2_,"axG",@progbits,_ZSt12__niter_baseIPPfET_S2_,comdat
	.weak	_ZSt12__niter_baseIPPfET_S2_
	.type	_ZSt12__niter_baseIPPfET_S2_, @function
_ZSt12__niter_baseIPPfET_S2_:
.LFB1150:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1150:
	.size	_ZSt12__niter_baseIPPfET_S2_, .-_ZSt12__niter_baseIPPfET_S2_
	.section	.text._ZSt13__copy_move_aILb0EPPfS1_ET1_T0_S3_S2_,"axG",@progbits,_ZSt13__copy_move_aILb0EPPfS1_ET1_T0_S3_S2_,comdat
	.weak	_ZSt13__copy_move_aILb0EPPfS1_ET1_T0_S3_S2_
	.type	_ZSt13__copy_move_aILb0EPPfS1_ET1_T0_S3_S2_, @function
_ZSt13__copy_move_aILb0EPPfS1_ET1_T0_S3_S2_:
.LFB1151:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movb	$1, -1(%rbp)
	movq	-40(%rbp), %rdx
	movq	-32(%rbp), %rcx
	movq	-24(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZNSt11__copy_moveILb0ELb1ESt26random_access_iterator_tagE8__copy_mIPfEEPT_PKS4_S7_S5_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1151:
	.size	_ZSt13__copy_move_aILb0EPPfS1_ET1_T0_S3_S2_, .-_ZSt13__copy_move_aILb0EPPfS1_ET1_T0_S3_S2_
	.section	.text._ZSt12__niter_wrapIPPfET_RKS2_S2_,"axG",@progbits,_ZSt12__niter_wrapIPPfET_RKS2_S2_,comdat
	.weak	_ZSt12__niter_wrapIPPfET_RKS2_S2_
	.type	_ZSt12__niter_wrapIPPfET_RKS2_S2_, @function
_ZSt12__niter_wrapIPPfET_RKS2_S2_:
.LFB1152:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	-16(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1152:
	.size	_ZSt12__niter_wrapIPPfET_RKS2_S2_, .-_ZSt12__niter_wrapIPPfET_RKS2_S2_
	.section	.text._ZNSt20__uninitialized_copyILb1EE13__uninit_copyIPPfS3_EET0_T_S5_S4_,"axG",@progbits,_ZNSt20__uninitialized_copyILb1EE13__uninit_copyIPPfS3_EET0_T_S5_S4_,comdat
	.weak	_ZNSt20__uninitialized_copyILb1EE13__uninit_copyIPPfS3_EET0_T_S5_S4_
	.type	_ZNSt20__uninitialized_copyILb1EE13__uninit_copyIPPfS3_EET0_T_S5_S4_, @function
_ZNSt20__uninitialized_copyILb1EE13__uninit_copyIPPfS3_EET0_T_S5_S4_:
.LFB1153:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-24(%rbp), %rdx
	movq	-16(%rbp), %rcx
	movq	-8(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZSt4copyIPPfS1_ET0_T_S3_S2_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1153:
	.size	_ZNSt20__uninitialized_copyILb1EE13__uninit_copyIPPfS3_EET0_T_S5_S4_, .-_ZNSt20__uninitialized_copyILb1EE13__uninit_copyIPPfS3_EET0_T_S5_S4_
	.section	.text._ZN9__gnu_cxx13new_allocatorIPfEC2ERKS2_,"axG",@progbits,_ZN9__gnu_cxx13new_allocatorIPfEC5ERKS2_,comdat
	.align 2
	.weak	_ZN9__gnu_cxx13new_allocatorIPfEC2ERKS2_
	.type	_ZN9__gnu_cxx13new_allocatorIPfEC2ERKS2_, @function
_ZN9__gnu_cxx13new_allocatorIPfEC2ERKS2_:
.LFB1155:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	nop
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1155:
	.size	_ZN9__gnu_cxx13new_allocatorIPfEC2ERKS2_, .-_ZN9__gnu_cxx13new_allocatorIPfEC2ERKS2_
	.weak	_ZN9__gnu_cxx13new_allocatorIPfEC1ERKS2_
	.set	_ZN9__gnu_cxx13new_allocatorIPfEC1ERKS2_,_ZN9__gnu_cxx13new_allocatorIPfEC2ERKS2_
	.section	.text._ZNSt20__uninitialized_copyILb1EE13__uninit_copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS4_SaIS4_EEEEPS4_EET0_T_SD_SC_,"axG",@progbits,_ZNSt20__uninitialized_copyILb1EE13__uninit_copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS4_SaIS4_EEEEPS4_EET0_T_SD_SC_,comdat
	.weak	_ZNSt20__uninitialized_copyILb1EE13__uninit_copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS4_SaIS4_EEEEPS4_EET0_T_SD_SC_
	.type	_ZNSt20__uninitialized_copyILb1EE13__uninit_copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS4_SaIS4_EEEEPS4_EET0_T_SD_SC_, @function
_ZNSt20__uninitialized_copyILb1EE13__uninit_copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS4_SaIS4_EEEEPS4_EET0_T_SD_SC_:
.LFB1157:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-24(%rbp), %rdx
	movq	-16(%rbp), %rcx
	movq	-8(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	_ZSt4copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_ET0_T_SB_SA_
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1157:
	.size	_ZNSt20__uninitialized_copyILb1EE13__uninit_copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS4_SaIS4_EEEEPS4_EET0_T_SD_SC_, .-_ZNSt20__uninitialized_copyILb1EE13__uninit_copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS4_SaIS4_EEEEPS4_EET0_T_SD_SC_
	.section	.text._ZN9__gnu_cxx13new_allocatorIPfE8allocateEmPKv,"axG",@progbits,_ZN9__gnu_cxx13new_allocatorIPfE8allocateEmPKv,comdat
	.align 2
	.weak	_ZN9__gnu_cxx13new_allocatorIPfE8allocateEmPKv
	.type	_ZN9__gnu_cxx13new_allocatorIPfE8allocateEmPKv, @function
_ZN9__gnu_cxx13new_allocatorIPfE8allocateEmPKv:
.LFB1158:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -8(%rbp)
	movq	%rsi, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, %rdi
	call	_ZNK9__gnu_cxx13new_allocatorIPfE8max_sizeEv
	cmpq	%rax, -16(%rbp)
	seta	%al
	testb	%al, %al
	je	.L320
	call	_ZSt17__throw_bad_allocv@PLT
.L320:
	movq	-16(%rbp), %rax
	salq	$3, %rax
	movq	%rax, %rdi
	call	_Znwm@PLT
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1158:
	.size	_ZN9__gnu_cxx13new_allocatorIPfE8allocateEmPKv, .-_ZN9__gnu_cxx13new_allocatorIPfE8allocateEmPKv
	.section	.text._ZNK9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS1_SaIS1_EEE4baseEv,"axG",@progbits,_ZNK9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS1_SaIS1_EEE4baseEv,comdat
	.align 2
	.weak	_ZNK9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS1_SaIS1_EEE4baseEv
	.type	_ZNK9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS1_SaIS1_EEE4baseEv, @function
_ZNK9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS1_SaIS1_EEE4baseEv:
.LFB1159:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1159:
	.size	_ZNK9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS1_SaIS1_EEE4baseEv, .-_ZNK9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS1_SaIS1_EEE4baseEv
	.section	.text._ZNK9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEE4baseEv,"axG",@progbits,_ZNK9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEE4baseEv,comdat
	.align 2
	.weak	_ZNK9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEE4baseEv
	.type	_ZNK9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEE4baseEv, @function
_ZNK9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEE4baseEv:
.LFB1160:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movq	-8(%rbp), %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1160:
	.size	_ZNK9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEE4baseEv, .-_ZNK9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEE4baseEv
	.section	.text._ZNSt11__copy_moveILb0ELb1ESt26random_access_iterator_tagE8__copy_mIPfEEPT_PKS4_S7_S5_,"axG",@progbits,_ZNSt11__copy_moveILb0ELb1ESt26random_access_iterator_tagE8__copy_mIPfEEPT_PKS4_S7_S5_,comdat
	.weak	_ZNSt11__copy_moveILb0ELb1ESt26random_access_iterator_tagE8__copy_mIPfEEPT_PKS4_S7_S5_
	.type	_ZNSt11__copy_moveILb0ELb1ESt26random_access_iterator_tagE8__copy_mIPfEEPT_PKS4_S7_S5_, @function
_ZNSt11__copy_moveILb0ELb1ESt26random_access_iterator_tagE8__copy_mIPfEEPT_PKS4_S7_S5_:
.LFB1161:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	-32(%rbp), %rax
	subq	-24(%rbp), %rax
	sarq	$3, %rax
	movq	%rax, -8(%rbp)
	cmpq	$0, -8(%rbp)
	je	.L327
	movq	-8(%rbp), %rax
	leaq	0(,%rax,8), %rdx
	movq	-24(%rbp), %rcx
	movq	-40(%rbp), %rax
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	memmove@PLT
.L327:
	movq	-8(%rbp), %rax
	leaq	0(,%rax,8), %rdx
	movq	-40(%rbp), %rax
	addq	%rdx, %rax
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1161:
	.size	_ZNSt11__copy_moveILb0ELb1ESt26random_access_iterator_tagE8__copy_mIPfEEPT_PKS4_S7_S5_, .-_ZNSt11__copy_moveILb0ELb1ESt26random_access_iterator_tagE8__copy_mIPfEEPT_PKS4_S7_S5_
	.section	.text._ZNK9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEEplEl,"axG",@progbits,_ZNK9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEEplEl,comdat
	.align 2
	.weak	_ZNK9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEEplEl
	.type	_ZNK9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEEplEl, @function
_ZNK9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEEplEl:
.LFB1162:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	-40(%rbp), %rax
	movq	(%rax), %rax
	movq	-48(%rbp), %rdx
	salq	$3, %rdx
	addq	%rdx, %rax
	movq	%rax, -24(%rbp)
	leaq	-24(%rbp), %rdx
	leaq	-16(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZN9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEEC1ERKS2_
	movq	-16(%rbp), %rax
	movq	-8(%rbp), %rcx
	xorq	%fs:40, %rcx
	je	.L331
	call	__stack_chk_fail@PLT
.L331:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1162:
	.size	_ZNK9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEEplEl, .-_ZNK9__gnu_cxx17__normal_iteratorIPPfSt6vectorIS1_SaIS1_EEEplEl
	.section	.text._ZSt4copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_ET0_T_SB_SA_,"axG",@progbits,_ZSt4copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_ET0_T_SB_SA_,comdat
	.weak	_ZSt4copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_ET0_T_SB_SA_
	.type	_ZSt4copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_ET0_T_SB_SA_, @function
_ZSt4copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_ET0_T_SB_SA_:
.LFB1163:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$40, %rsp
	.cfi_offset 3, -24
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__miter_baseIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEET_S9_
	movq	%rax, %rbx
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__miter_baseIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEET_S9_
	movq	%rax, %rcx
	movq	-40(%rbp), %rax
	movq	%rax, %rdx
	movq	%rbx, %rsi
	movq	%rcx, %rdi
	call	_ZSt14__copy_move_a2ILb0EN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_ET1_T0_SB_SA_
	addq	$40, %rsp
	popq	%rbx
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1163:
	.size	_ZSt4copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_ET0_T_SB_SA_, .-_ZSt4copyIN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_ET0_T_SB_SA_
	.section	.text._ZNK9__gnu_cxx13new_allocatorIPfE8max_sizeEv,"axG",@progbits,_ZNK9__gnu_cxx13new_allocatorIPfE8max_sizeEv,comdat
	.align 2
	.weak	_ZNK9__gnu_cxx13new_allocatorIPfE8max_sizeEv
	.type	_ZNK9__gnu_cxx13new_allocatorIPfE8max_sizeEv, @function
_ZNK9__gnu_cxx13new_allocatorIPfE8max_sizeEv:
.LFB1164:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	movq	%rdi, -8(%rbp)
	movabsq	$1152921504606846975, %rax
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1164:
	.size	_ZNK9__gnu_cxx13new_allocatorIPfE8max_sizeEv, .-_ZNK9__gnu_cxx13new_allocatorIPfE8max_sizeEv
	.section	.text._ZSt14__copy_move_a2ILb0EN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_ET1_T0_SB_SA_,"axG",@progbits,_ZSt14__copy_move_a2ILb0EN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_ET1_T0_SB_SA_,comdat
	.weak	_ZSt14__copy_move_a2ILb0EN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_ET1_T0_SB_SA_
	.type	_ZSt14__copy_move_a2ILb0EN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_ET1_T0_SB_SA_, @function
_ZSt14__copy_move_a2ILb0EN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_ET1_T0_SB_SA_:
.LFB1165:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r12
	pushq	%rbx
	subq	$32, %rsp
	.cfi_offset 12, -24
	.cfi_offset 3, -32
	movq	%rdi, -24(%rbp)
	movq	%rsi, -32(%rbp)
	movq	%rdx, -40(%rbp)
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__niter_baseIPPfET_S2_
	movq	%rax, %r12
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__niter_baseIPKPfSt6vectorIS0_SaIS0_EEET_N9__gnu_cxx17__normal_iteratorIS6_T0_EE
	movq	%rax, %rbx
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	_ZSt12__niter_baseIPKPfSt6vectorIS0_SaIS0_EEET_N9__gnu_cxx17__normal_iteratorIS6_T0_EE
	movq	%r12, %rdx
	movq	%rbx, %rsi
	movq	%rax, %rdi
	call	_ZSt13__copy_move_aILb0EPKPfPS0_ET1_T0_S5_S4_
	movq	%rax, %rdx
	leaq	-40(%rbp), %rax
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	_ZSt12__niter_wrapIPPfET_RKS2_S2_
	addq	$32, %rsp
	popq	%rbx
	popq	%r12
	popq	%rbp
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1165:
	.size	_ZSt14__copy_move_a2ILb0EN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_ET1_T0_SB_SA_, .-_ZSt14__copy_move_a2ILb0EN9__gnu_cxx17__normal_iteratorIPKPfSt6vectorIS2_SaIS2_EEEEPS2_ET1_T0_SB_SA_
	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.rel.local.DW.ref.__gxx_personality_v0,"awG",@progbits,DW.ref.__gxx_personality_v0,comdat
	.align 8
	.type	DW.ref.__gxx_personality_v0, @object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0
	.ident	"GCC: (Ubuntu 9.4.0-1ubuntu1~20.04.1) 9.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	 1f - 0f
	.long	 4f - 1f
	.long	 5
0:
	.string	 "GNU"
1:
	.align 8
	.long	 0xc0000002
	.long	 3f - 2f
2:
	.long	 0x3
3:
	.align 8
4:
