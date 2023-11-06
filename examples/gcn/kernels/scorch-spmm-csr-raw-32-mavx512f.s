	.file	"scorch-spmm-csr-raw.cpp"
	.text
	.p2align 4
	.globl	_Z8evaluateSt6vectorIiSaIiEES1_PKiS3_PKfS1_S5_Pf
	.type	_Z8evaluateSt6vectorIiSaIiEES1_PKiS3_PKfS1_S5_Pf, @function
_Z8evaluateSt6vectorIiSaIiEES1_PKiS3_PKfS1_S5_Pf:
.LFB878:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rdx, %rax
	movq	%rcx, %r10
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	andq	$-64, %rsp
	subq	$256, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movq	24(%rbp), %rbx
	movq	(%rdi), %rdx
	movq	(%r9), %rcx
	movq	16(%rbp), %r15
	movq	%rbx, 8(%rsp)
	movl	4(%rcx), %r11d
	movq	%fs:40, %rbx
	movq	%rbx, 248(%rsp)
	xorl	%ebx, %ebx
	movl	4(%rdx), %ebx
	movq	(%rsi), %rdx
	movl	(%rdx), %edx
	movl	%ebx, 24(%rsp)
	testl	%edx, %edx
	jle	.L1
	testl	%r11d, %r11d
	jle	.L1
	movq	%rax, %rdi
	movq	%rax, 48(%rsp)
	leal	-1(%rdx), %eax
	movq	%r8, %r12
	leaq	4(%rdi,%rax,4), %rax
	movl	$0, 28(%rsp)
	leaq	64(%rsp), %rbx
	movq	%rax, 16(%rsp)
	leaq	192(%rsp), %r13
.L13:
	movq	48(%rsp), %rax
	movq	8(%rsp), %rsi
	movl	(%rax), %edi
	movl	4(%rax), %r8d
	movslq	28(%rsp), %rax
	movl	%edi, 56(%rsp)
	leaq	(%rsi,%rax,4), %r14
	leal	-1(%r8), %eax
	leal	1(%rdi), %esi
	movl	%eax, 40(%rsp)
	cmpl	%esi, %eax
	movl	%esi, 36(%rsp)
	setg	%dl
	cmpl	$-2147483648, %r8d
	setne	%al
	xorl	%esi, %esi
	andl	%eax, %edx
	leal	2(%rdi), %eax
	movb	%dl, 47(%rsp)
	movl	%eax, 32(%rsp)
.L4:
	movl	$16, %ecx
	movq	%rbx, %rdi
	xorl	%eax, %eax
	rep stosq
	cmpl	%r8d, 56(%rsp)
	jge	.L12
	movl	36(%rsp), %eax
	cmpb	$0, 47(%rsp)
	movl	%eax, 60(%rsp)
	je	.L15
	movslq	32(%rsp), %rdi
.L10:
	movl	-8(%r10,%rdi,4), %r9d
	movl	-4(%r10,%rdi,4), %ecx
	movq	%rbx, %rax
	vmovss	-8(%r12,%rdi,4), %xmm2
	vmovss	-4(%r12,%rdi,4), %xmm1
	imull	%r11d, %r9d
	imull	%r11d, %ecx
	movslq	%r9d, %r9
	leaq	(%r9,%rsi), %rdx
	movslq	%ecx, %rcx
	leaq	(%r15,%rdx,4), %rdx
	subq	%r9, %rcx
	.p2align 4,,10
	.p2align 3
.L9:
	vmovss	(%rdx), %xmm0
	vfmadd213ss	(%rax), %xmm2, %xmm0
	addq	$4, %rax
	vfmadd231ss	(%rdx,%rcx,4), %xmm1, %xmm0
	addq	$4, %rdx
	vmovss	%xmm0, -4(%rax)
	cmpq	%r13, %rax
	jne	.L9
	addl	$2, 60(%rsp)
	movslq	%edi, %rdx
	movl	60(%rsp), %eax
	addq	$2, %rdi
	cmpl	40(%rsp), %eax
	jl	.L10
.L8:
	vmovaps	64(%rsp), %zmm2
	vmovaps	128(%rsp), %zmm1
	.p2align 4,,10
	.p2align 3
.L11:
	movl	(%r10,%rdx,4), %eax
	vbroadcastss	(%r12,%rdx,4), %zmm0
	addq	$1, %rdx
	imull	%r11d, %eax
	cltq
	addq	%rsi, %rax
	leaq	(%r15,%rax,4), %rax
	vfmadd231ps	(%rax), %zmm0, %zmm2
	vfmadd231ps	64(%rax), %zmm0, %zmm1
	cmpl	%edx, %r8d
	jg	.L11
	vmovaps	%zmm2, 64(%rsp)
	vmovaps	%zmm1, 128(%rsp)
.L12:
	vmovdqa	(%rbx), %xmm3
	addq	$32, %rsi
	subq	$-128, %r14
	vmovups	%xmm3, -128(%r14)
	vmovdqa	16(%rbx), %xmm4
	vmovups	%xmm4, -112(%r14)
	vmovdqa	32(%rbx), %xmm5
	vmovups	%xmm5, -96(%r14)
	vmovdqa	48(%rbx), %xmm6
	vmovups	%xmm6, -80(%r14)
	vmovdqa	64(%rbx), %xmm7
	vmovups	%xmm7, -64(%r14)
	vmovdqa	80(%rbx), %xmm3
	vmovups	%xmm3, -48(%r14)
	vmovdqa	96(%rbx), %xmm4
	vmovups	%xmm4, -32(%r14)
	vmovdqa	112(%rbx), %xmm5
	vmovups	%xmm5, -16(%r14)
	cmpl	%esi, %r11d
	jg	.L4
	addq	$4, 48(%rsp)
	movl	24(%rsp), %esi
	movq	48(%rsp), %rax
	addl	%esi, 28(%rsp)
	cmpq	16(%rsp), %rax
	jne	.L13
	vzeroupper
.L1:
	movq	248(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L22
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_remember_state
	.cfi_def_cfa 7, 8
	ret
.L15:
	.cfi_restore_state
	movslq	56(%rsp), %rdx
	jmp	.L8
.L22:
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
