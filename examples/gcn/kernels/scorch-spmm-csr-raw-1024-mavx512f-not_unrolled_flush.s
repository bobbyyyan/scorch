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
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	andq	$-64, %rsp
	subq	$4096, %rsp
	orq	$0, (%rsp)
	addq	$-128, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movq	16(%rbp), %r12
	movq	%rcx, %rbx
	movq	24(%rbp), %rcx
	movq	%rdx, %rax
	movq	(%rdi), %rdx
	movq	%rcx, 16(%rsp)
	movl	4(%rdx), %edi
	movq	(%rsi), %rdx
	movq	%fs:40, %rcx
	movq	%rcx, 4216(%rsp)
	xorl	%ecx, %ecx
	movl	(%rdx), %edx
	movq	(%r9), %rcx
	movl	%edi, 40(%rsp)
	movl	4(%rcx), %r15d
	testl	%edx, %edx
	jle	.L1
	testl	%r15d, %r15d
	jle	.L1
	movq	%rax, %rsi
	movq	%rax, 56(%rsp)
	leal	-1(%rdx), %eax
	movq	%r8, %r11
	leaq	4(%rsi,%rax,4), %rax
	movl	$0, 44(%rsp)
	leaq	64(%rsp), %r13
	movq	%rax, 24(%rsp)
	leaq	4168(%rsp), %rax
	leaq	4160(%rsp), %r14
	movq	%rax, (%rsp)
.L14:
	movq	56(%rsp), %rax
	movq	16(%rsp), %rsi
	movl	(%rax), %edi
	movl	4(%rax), %r9d
	movslq	44(%rsp), %rax
	movl	%edi, 48(%rsp)
	leaq	(%rsi,%rax,4), %r10
	leal	-1(%r9), %eax
	leal	1(%rdi), %esi
	movl	%eax, 32(%rsp)
	cmpl	%esi, %eax
	movl	%esi, 12(%rsp)
	setg	%dl
	cmpl	$-2147483648, %r9d
	setne	%al
	xorl	%r8d, %r8d
	andl	%eax, %edx
	leal	2(%rdi), %eax
	movb	%dl, 39(%rsp)
	movl	%eax, 8(%rsp)
.L4:
	movl	$512, %ecx
	movq	%r13, %rdi
	xorl	%eax, %eax
	rep stosq
	cmpl	%r9d, 48(%rsp)
	jge	.L12
	movl	12(%rsp), %eax
	cmpb	$0, 39(%rsp)
	movl	%eax, 52(%rsp)
	je	.L16
	movslq	8(%rsp), %rsi
.L10:
	movl	-8(%rbx,%rsi,4), %edi
	movl	-4(%rbx,%rsi,4), %ecx
	movq	%r13, %rax
	vmovss	-8(%r11,%rsi,4), %xmm2
	vmovss	-4(%r11,%rsi,4), %xmm1
	imull	%r15d, %edi
	imull	%r15d, %ecx
	movslq	%edi, %rdi
	leaq	(%rdi,%r8), %rdx
	movslq	%ecx, %rcx
	leaq	(%r12,%rdx,4), %rdx
	subq	%rdi, %rcx
	.p2align 4,,10
	.p2align 3
.L9:
	vmovss	(%rdx), %xmm0
	vfmadd213ss	(%rax), %xmm2, %xmm0
	addq	$4, %rax
	vfmadd231ss	(%rdx,%rcx,4), %xmm1, %xmm0
	addq	$4, %rdx
	vmovss	%xmm0, -4(%rax)
	cmpq	%r14, %rax
	jne	.L9
	addl	$2, 52(%rsp)
	movslq	%esi, %rcx
	movl	52(%rsp), %eax
	addq	$2, %rsi
	cmpl	32(%rsp), %eax
	jl	.L10
.L13:
	movl	(%rbx,%rcx,4), %eax
	vbroadcastss	(%r11,%rcx,4), %zmm1
	imull	%r15d, %eax
	cltq
	addq	%r8, %rax
	leaq	(%r12,%rax,4), %rdx
	movq	%r13, %rax
	.p2align 4,,10
	.p2align 3
.L11:
	vmovups	(%rdx), %zmm0
	vfmadd213ps	(%rax), %zmm1, %zmm0
	addq	$64, %rax
	addq	$64, %rdx
	vmovaps	%zmm0, -64(%rax)
	cmpq	%r14, %rax
	jne	.L11
	addq	$1, %rcx
	cmpl	%ecx, %r9d
	jg	.L13
.L12:
	movq	0(%r13), %rax
	leaq	8(%r10), %rdi
	movq	%r13, %rsi
	addq	$1024, %r8
	andq	$-8, %rdi
	movq	%rax, (%r10)
	movq	(%rsp), %rax
	movq	-16(%rax), %rax
	movq	%rax, 4088(%r10)
	movq	%r10, %rax
	addq	$4096, %r10
	subq	%rdi, %rax
	leal	4096(%rax), %ecx
	subq	%rax, %rsi
	shrl	$3, %ecx
	rep movsq
	cmpl	%r8d, %r15d
	jg	.L4
	addq	$4, 56(%rsp)
	movl	40(%rsp), %esi
	movq	56(%rsp), %rax
	addl	%esi, 44(%rsp)
	cmpq	24(%rsp), %rax
	jne	.L14
	vzeroupper
.L1:
	movq	4216(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L23
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
.L16:
	.cfi_restore_state
	movslq	48(%rsp), %rcx
	jmp	.L13
.L23:
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
