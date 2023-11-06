	.file	"scorch-spmm-csr-raw.cpp"
	.text
	.p2align 4
	.globl	_Z8evaluateSt6vectorIiSaIiEES1_PKiS3_PKfS1_S5_Pf
	.type	_Z8evaluateSt6vectorIiSaIiEES1_PKiS3_PKfS1_S5_Pf, @function
_Z8evaluateSt6vectorIiSaIiEES1_PKiS3_PKfS1_S5_Pf:
.LFB878:
	.cfi_startproc
	endbr64
	pushq	%r15
	.cfi_def_cfa_offset 16
	.cfi_offset 15, -16
	movq	%rdx, %rax
	pushq	%r14
	.cfi_def_cfa_offset 24
	.cfi_offset 14, -24
	pushq	%r13
	.cfi_def_cfa_offset 32
	.cfi_offset 13, -32
	pushq	%r12
	.cfi_def_cfa_offset 40
	.cfi_offset 12, -40
	pushq	%rbp
	.cfi_def_cfa_offset 48
	.cfi_offset 6, -48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	.cfi_offset 3, -56
	movq	%rcx, %rbx
	subq	$216, %rsp
	.cfi_def_cfa_offset 272
	movq	(%rdi), %rdx
	movq	280(%rsp), %rcx
	movq	272(%rsp), %rbp
	movl	4(%rdx), %edi
	movq	(%rsi), %rdx
	movq	%rcx, 56(%rsp)
	movl	(%rdx), %edx
	movq	%fs:40, %rcx
	movq	%rcx, 200(%rsp)
	xorl	%ecx, %ecx
	movq	(%r9), %rcx
	movl	%edi, 44(%rsp)
	movl	4(%rcx), %r12d
	testl	%edx, %edx
	jle	.L1
	testl	%r12d, %r12d
	jle	.L1
	movq	%rax, %rdi
	movq	%rax, 16(%rsp)
	leal	-1(%rdx), %eax
	movq	%r8, %r11
	leaq	4(%rdi,%rax,4), %rax
	movl	$0, 40(%rsp)
	leaq	64(%rsp), %r13
	movq	%rax, 48(%rsp)
	leaq	192(%rsp), %rsi
.L13:
	movq	16(%rsp), %rax
	movq	56(%rsp), %rcx
	movl	(%rax), %edi
	movl	4(%rax), %r9d
	movslq	40(%rsp), %rax
	movl	%edi, 12(%rsp)
	leaq	(%rcx,%rax,4), %r10
	leal	-1(%r9), %eax
	leal	1(%rdi), %ecx
	movl	%eax, 28(%rsp)
	cmpl	%ecx, %eax
	movl	%ecx, 32(%rsp)
	setg	%dl
	cmpl	$-2147483648, %r9d
	setne	%al
	xorl	%r8d, %r8d
	andl	%eax, %edx
	leal	2(%rdi), %eax
	movb	%dl, 27(%rsp)
	movl	%eax, 36(%rsp)
.L4:
	movl	$16, %ecx
	movq	%r13, %rdi
	xorl	%eax, %eax
	rep stosq
	cmpl	%r9d, 12(%rsp)
	jge	.L12
	cmpb	$0, 27(%rsp)
	movl	32(%rsp), %r15d
	je	.L15
	movslq	36(%rsp), %rdi
.L10:
	movl	-8(%rbx,%rdi,4), %r14d
	movl	-4(%rbx,%rdi,4), %ecx
	movq	%r13, %rax
	movss	-8(%r11,%rdi,4), %xmm3
	movss	-4(%r11,%rdi,4), %xmm2
	imull	%r12d, %r14d
	imull	%r12d, %ecx
	movslq	%r14d, %r14
	leaq	(%r14,%r8), %rdx
	movslq	%ecx, %rcx
	leaq	0(%rbp,%rdx,4), %rdx
	subq	%r14, %rcx
	.p2align 4,,10
	.p2align 3
.L9:
	movss	(%rdx,%rcx,4), %xmm1
	movss	(%rdx), %xmm0
	addq	$4, %rax
	addq	$4, %rdx
	mulss	%xmm2, %xmm1
	mulss	%xmm3, %xmm0
	addss	-4(%rax), %xmm0
	addss	%xmm1, %xmm0
	movss	%xmm0, -4(%rax)
	cmpq	%rax, %rsi
	jne	.L9
	movslq	%edi, %rdx
	addl	$2, %r15d
	addq	$2, %rdi
	cmpl	28(%rsp), %r15d
	jl	.L10
.L8:
	movaps	64(%rsp), %xmm9
	movaps	80(%rsp), %xmm8
	movaps	96(%rsp), %xmm7
	movaps	112(%rsp), %xmm6
	movaps	128(%rsp), %xmm5
	movaps	144(%rsp), %xmm4
	movaps	160(%rsp), %xmm3
	movaps	176(%rsp), %xmm2
	.p2align 4,,10
	.p2align 3
.L11:
	movl	(%rbx,%rdx,4), %eax
	movss	(%r11,%rdx,4), %xmm0
	addq	$1, %rdx
	imull	%r12d, %eax
	shufps	$0, %xmm0, %xmm0
	cltq
	addq	%r8, %rax
	leaq	0(%rbp,%rax,4), %rax
	movups	(%rax), %xmm1
	mulps	%xmm0, %xmm1
	addps	%xmm1, %xmm9
	movups	16(%rax), %xmm1
	mulps	%xmm0, %xmm1
	addps	%xmm1, %xmm8
	movups	32(%rax), %xmm1
	mulps	%xmm0, %xmm1
	addps	%xmm1, %xmm7
	movups	48(%rax), %xmm1
	mulps	%xmm0, %xmm1
	addps	%xmm1, %xmm6
	movups	64(%rax), %xmm1
	mulps	%xmm0, %xmm1
	addps	%xmm1, %xmm5
	movups	80(%rax), %xmm1
	mulps	%xmm0, %xmm1
	addps	%xmm1, %xmm4
	movups	96(%rax), %xmm1
	mulps	%xmm0, %xmm1
	addps	%xmm1, %xmm3
	movups	112(%rax), %xmm1
	mulps	%xmm1, %xmm0
	addps	%xmm0, %xmm2
	cmpl	%edx, %r9d
	jg	.L11
	movaps	%xmm9, 64(%rsp)
	movaps	%xmm8, 80(%rsp)
	movaps	%xmm7, 96(%rsp)
	movaps	%xmm6, 112(%rsp)
	movaps	%xmm5, 128(%rsp)
	movaps	%xmm4, 144(%rsp)
	movaps	%xmm3, 160(%rsp)
	movaps	%xmm2, 176(%rsp)
.L12:
	movdqa	0(%r13), %xmm4
	addq	$32, %r8
	subq	$-128, %r10
	movups	%xmm4, -128(%r10)
	movdqa	16(%r13), %xmm5
	movups	%xmm5, -112(%r10)
	movdqa	32(%r13), %xmm6
	movups	%xmm6, -96(%r10)
	movdqa	48(%r13), %xmm7
	movups	%xmm7, -80(%r10)
	movdqa	64(%r13), %xmm4
	movups	%xmm4, -64(%r10)
	movdqa	80(%r13), %xmm5
	movups	%xmm5, -48(%r10)
	movdqa	96(%r13), %xmm6
	movups	%xmm6, -32(%r10)
	movdqa	112(%r13), %xmm7
	movups	%xmm7, -16(%r10)
	cmpl	%r8d, %r12d
	jg	.L4
	addq	$4, 16(%rsp)
	movl	44(%rsp), %ecx
	movq	16(%rsp), %rax
	addl	%ecx, 40(%rsp)
	cmpq	48(%rsp), %rax
	jne	.L13
.L1:
	movq	200(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L21
	addq	$216, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%rbp
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	ret
.L15:
	.cfi_restore_state
	movslq	12(%rsp), %rdx
	jmp	.L8
.L21:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE878:
	.size	_Z8evaluateSt6vectorIiSaIiEES1_PKiS3_PKfS1_S5_Pf, .-_Z8evaluateSt6vectorIiSaIiEES1_PKiS3_PKfS1_S5_Pf
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
