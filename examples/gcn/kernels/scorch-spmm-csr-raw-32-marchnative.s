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
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%r15
	pushq	%r14
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	movq	%rcx, %r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	andq	$-32, %rsp
	subq	$256, %rsp
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movq	24(%rbp), %rdx
	movq	(%r9), %rcx
	movq	%rdx, (%rsp)
	movq	16(%rbp), %rbx
	movl	4(%rcx), %r12d
	movq	%fs:40, %rdx
	movq	%rdx, 248(%rsp)
	xorl	%edx, %edx
	movq	(%rdi), %rdx
	movl	4(%rdx), %edi
	movq	(%rsi), %rdx
	movl	%edi, 20(%rsp)
	movl	(%rdx), %edx
	testl	%edx, %edx
	jle	.L1
	testl	%r12d, %r12d
	jle	.L1
	movq	%rax, %rdi
	movq	%rax, 32(%rsp)
	leal	-1(%rdx), %eax
	leaq	4(%rdi,%rax,4), %rax
	movq	%rax, 8(%rsp)
	movl	$0, 24(%rsp)
	vmovdqa64	.LC0(%rip), %ymm6
	vmovdqa64	.LC1(%rip), %ymm5
	vmovdqa64	.LC2(%rip), %ymm4
	vmovdqa64	.LC3(%rip), %ymm3
	movq	%r8, %r13
	leaq	96(%rsp), %r15
.L22:
	movq	32(%rsp), %rax
	movq	(%rsp), %rcx
	movl	(%rax), %edx
	movl	4(%rax), %edi
	movslq	24(%rsp), %rax
	movl	%edx, 60(%rsp)
	leaq	(%rcx,%rax,4), %rax
	movq	%rax, 48(%rsp)
	leal	-1(%rdi), %ecx
	leal	1(%rdx), %eax
	cmpl	%eax, %ecx
	setg	%dl
	cmpl	$-2147483648, %edi
	setne	%al
	andl	%eax, %edx
	movl	%edi, 84(%rsp)
	movl	%ecx, 28(%rsp)
	movb	%dl, 59(%rsp)
	movl	$4, 68(%rsp)
	movl	$3, 72(%rsp)
	movl	$2, 76(%rsp)
	movl	$1, 80(%rsp)
	movq	$0, 88(%rsp)
.L4:
	movq	88(%rsp), %rax
	xorl	%ecx, %ecx
	movl	%eax, 64(%rsp)
	movl	%eax, %edi
	xorl	%edx, %edx
.L6:
	movl	%edx, %eax
	addl	$32, %edx
	movq	%rcx, 96(%rsp,%rax)
	movq	%rcx, 104(%rsp,%rax)
	movq	%rcx, 112(%rsp,%rax)
	movq	%rcx, 120(%rsp,%rax)
	cmpl	$128, %edx
	jb	.L6
	movl	84(%rsp), %edx
	cmpl	%edx, 60(%rsp)
	jge	.L20
	cmpb	$0, 59(%rsp)
	je	.L24
	movl	60(%rsp), %eax
	movq	%r15, 40(%rsp)
	addl	$2, %eax
	cltq
	vpbroadcastd	%edi, %ymm8
	jmp	.L15
.L81:
	vmovups	96(%r8), %ymm0
	vpaddd	%ymm11, %ymm8, %ymm11
	vfmadd132ps	96(%r11), %ymm0, %ymm7
	vpaddd	%ymm9, %ymm11, %ymm9
	vmovups	%ymm7, 96(%r8)
	vgatherdps	(%rbx,%ymm9,4), %ymm0{%k1}
	vfmadd132ps	-4(%r13,%rax,4){1to8}, %ymm7, %ymm0
	vmovups	%ymm0, 96(%r8)
.L14:
	movslq	%eax, %r10
	addq	$2, %rax
	leal	-1(%rax), %edx
	cmpl	%edx, 28(%rsp)
	jle	.L80
.L15:
	movl	-8(%r14,%rax,4), %esi
	vmovss	-8(%r13,%rax,4), %xmm1
	imull	%r12d, %esi
	movslq	%esi, %r9
	addq	88(%rsp), %r9
	leaq	(%rbx,%r9,4), %rdx
	shrq	$2, %rdx
	negq	%rdx
	andl	$7, %edx
	je	.L25
	leal	(%rsi,%rdi), %ecx
	movslq	%ecx, %rcx
	vmovss	(%rbx,%rcx,4), %xmm2
	movl	-4(%r14,%rax,4), %ecx
	vfmadd213ss	96(%rsp), %xmm1, %xmm2
	imull	%r12d, %ecx
	vmovss	-4(%r13,%rax,4), %xmm0
	leal	(%rdi,%rcx), %r8d
	movslq	%r8d, %r8
	vfmadd231ss	(%rbx,%r8,4), %xmm0, %xmm2
	vmovss	%xmm2, 96(%rsp)
	cmpl	$1, %edx
	je	.L26
	movl	80(%rsp), %r15d
	leal	(%rsi,%r15), %r8d
	movslq	%r8d, %r8
	vmovss	(%rbx,%r8,4), %xmm2
	leal	(%rcx,%r15), %r8d
	vfmadd213ss	100(%rsp), %xmm1, %xmm2
	movslq	%r8d, %r8
	vfmadd231ss	(%rbx,%r8,4), %xmm0, %xmm2
	vmovss	%xmm2, 100(%rsp)
	cmpl	$2, %edx
	je	.L27
	movl	76(%rsp), %r15d
	leal	(%rsi,%r15), %r8d
	movslq	%r8d, %r8
	vmovss	(%rbx,%r8,4), %xmm2
	leal	(%rcx,%r15), %r8d
	vfmadd213ss	104(%rsp), %xmm1, %xmm2
	movslq	%r8d, %r8
	vfmadd231ss	(%rbx,%r8,4), %xmm0, %xmm2
	vmovss	%xmm2, 104(%rsp)
	cmpl	$3, %edx
	je	.L28
	movl	72(%rsp), %r15d
	leal	(%rsi,%r15), %r8d
	movslq	%r8d, %r8
	vmovss	(%rbx,%r8,4), %xmm2
	leal	(%rcx,%r15), %r8d
	vfmadd213ss	108(%rsp), %xmm1, %xmm2
	movslq	%r8d, %r8
	vfmadd231ss	(%rbx,%r8,4), %xmm0, %xmm2
	vmovss	%xmm2, 108(%rsp)
	cmpl	$4, %edx
	je	.L29
	movl	68(%rsp), %r15d
	leal	(%rsi,%r15), %r8d
	movslq	%r8d, %r8
	vmovss	(%rbx,%r8,4), %xmm2
	leal	(%rcx,%r15), %r8d
	vfmadd213ss	112(%rsp), %xmm1, %xmm2
	movslq	%r8d, %r8
	vfmadd231ss	(%rbx,%r8,4), %xmm0, %xmm2
	vmovss	%xmm2, 112(%rsp)
	cmpl	$5, %edx
	je	.L30
	movl	64(%rsp), %r15d
	leal	5(%r15), %r8d
	leal	(%rsi,%r8), %r10d
	movslq	%r10d, %r10
	vmovss	(%rbx,%r10,4), %xmm2
	addl	%ecx, %r8d
	vfmadd213ss	116(%rsp), %xmm1, %xmm2
	movslq	%r8d, %r8
	vfmadd231ss	(%rbx,%r8,4), %xmm0, %xmm2
	vmovss	%xmm2, 116(%rsp)
	cmpl	$6, %edx
	je	.L31
	leal	6(%r15), %r8d
	leal	(%rsi,%r8), %r10d
	movslq	%r10d, %r10
	vmovss	(%rbx,%r10,4), %xmm2
	addl	%r8d, %ecx
	vfmadd213ss	120(%rsp), %xmm1, %xmm2
	movslq	%ecx, %rcx
	movl	$25, %r10d
	vfmadd132ss	(%rbx,%rcx,4), %xmm2, %xmm0
	movl	$7, %ecx
	vmovss	%xmm0, 120(%rsp)
.L11:
	movl	%edx, %r11d
	movq	40(%rsp), %r15
	addq	%r11, %r9
	leaq	(%r15,%r11,4), %r8
	leaq	(%rbx,%r9,4), %r11
	vmovaps	(%r11), %ymm11
	movl	-4(%r14,%rax,4), %r9d
	vbroadcastss	%xmm1, %ymm7
	vfmadd213ps	(%r8), %ymm7, %ymm11
	imull	%r12d, %r9d
	vpbroadcastd	%ecx, %ymm0
	vpaddd	%ymm6, %ymm0, %ymm2
	vpbroadcastd	%r9d, %ymm9
	vpaddd	%ymm8, %ymm2, %ymm2
	vmovups	%ymm11, (%r8)
	vpaddd	%ymm9, %ymm2, %ymm2
	kxnorb	%k1, %k1, %k1
	kmovb	%k1, %k2
	vgatherdps	(%rbx,%ymm2,4), %ymm10{%k2}
	vmovaps	%ymm10, %ymm2
	vfmadd132ps	-4(%r13,%rax,4){1to8}, %ymm11, %ymm2
	kmovb	%k1, %k3
	kmovb	%k1, %k4
	movl	$32, %r15d
	subl	%edx, %r15d
	vmovups	%ymm2, (%r8)
	vmovaps	32(%r11), %ymm11
	vpaddd	%ymm5, %ymm0, %ymm2
	vfmadd213ps	32(%r8), %ymm7, %ymm11
	vpaddd	%ymm8, %ymm2, %ymm2
	vpaddd	%ymm9, %ymm2, %ymm2
	movl	%r15d, %edx
	shrl	$3, %edx
	vmovups	%ymm11, 32(%r8)
	vgatherdps	(%rbx,%ymm2,4), %ymm10{%k3}
	vmovaps	%ymm10, %ymm2
	vfmadd132ps	-4(%r13,%rax,4){1to8}, %ymm11, %ymm2
	vpaddd	%ymm4, %ymm0, %ymm11
	vpaddd	%ymm3, %ymm0, %ymm0
	vpaddd	%ymm8, %ymm0, %ymm0
	vpaddd	%ymm9, %ymm0, %ymm0
	vmovups	%ymm2, 32(%r8)
	vmovaps	64(%r11), %ymm10
	vfmadd213ps	64(%r8), %ymm7, %ymm10
	vmovups	%ymm10, 64(%r8)
	vgatherdps	(%rbx,%ymm0,4), %ymm2{%k4}
	vmovaps	%ymm2, %ymm0
	vfmadd132ps	-4(%r13,%rax,4){1to8}, %ymm10, %ymm0
	vmovups	%ymm0, 64(%r8)
	cmpl	$3, %edx
	jne	.L81
	leal	24(%rcx), %r8d
	leal	(%rdi,%r8), %edx
	leal	(%rsi,%rdx), %r11d
	movslq	%r11d, %r11
	vmovss	(%rbx,%r11,4), %xmm2
	movslq	%r8d, %r8
	vfmadd213ss	96(%rsp,%r8,4), %xmm1, %xmm2
	vmovss	-4(%r13,%rax,4), %xmm0
	addl	%r9d, %edx
	movslq	%edx, %rdx
	vfmadd231ss	(%rbx,%rdx,4), %xmm0, %xmm2
	vmovss	%xmm2, 96(%rsp,%r8,4)
	leal	25(%rcx), %r8d
	cmpl	$25, %r10d
	je	.L14
	leal	(%rdi,%r8), %edx
	leal	(%rsi,%rdx), %r11d
	movslq	%r11d, %r11
	vmovss	(%rbx,%r11,4), %xmm2
	movslq	%r8d, %r8
	vfmadd213ss	96(%rsp,%r8,4), %xmm1, %xmm2
	addl	%r9d, %edx
	movslq	%edx, %rdx
	vfmadd231ss	(%rbx,%rdx,4), %xmm0, %xmm2
	vmovss	%xmm2, 96(%rsp,%r8,4)
	leal	26(%rcx), %r8d
	cmpl	$26, %r10d
	je	.L14
	leal	(%rdi,%r8), %edx
	leal	(%rsi,%rdx), %r11d
	movslq	%r11d, %r11
	vmovss	(%rbx,%r11,4), %xmm2
	movslq	%r8d, %r8
	vfmadd213ss	96(%rsp,%r8,4), %xmm1, %xmm2
	addl	%r9d, %edx
	movslq	%edx, %rdx
	vfmadd231ss	(%rbx,%rdx,4), %xmm0, %xmm2
	vmovss	%xmm2, 96(%rsp,%r8,4)
	leal	27(%rcx), %r8d
	cmpl	$27, %r10d
	je	.L14
	leal	(%rdi,%r8), %edx
	leal	(%rsi,%rdx), %r11d
	movslq	%r11d, %r11
	vmovss	(%rbx,%r11,4), %xmm2
	movslq	%r8d, %r8
	vfmadd213ss	96(%rsp,%r8,4), %xmm1, %xmm2
	addl	%r9d, %edx
	movslq	%edx, %rdx
	vfmadd231ss	(%rbx,%rdx,4), %xmm0, %xmm2
	vmovss	%xmm2, 96(%rsp,%r8,4)
	leal	28(%rcx), %r8d
	cmpl	$28, %r10d
	je	.L14
	leal	(%rdi,%r8), %edx
	leal	(%rsi,%rdx), %r11d
	movslq	%r11d, %r11
	vmovss	(%rbx,%r11,4), %xmm2
	movslq	%r8d, %r8
	vfmadd213ss	96(%rsp,%r8,4), %xmm1, %xmm2
	addl	%r9d, %edx
	movslq	%edx, %rdx
	vfmadd231ss	(%rbx,%rdx,4), %xmm0, %xmm2
	vmovss	%xmm2, 96(%rsp,%r8,4)
	leal	29(%rcx), %r8d
	cmpl	$29, %r10d
	je	.L14
	leal	(%rdi,%r8), %edx
	leal	(%rsi,%rdx), %r11d
	movslq	%r11d, %r11
	vmovss	(%rbx,%r11,4), %xmm2
	movslq	%r8d, %r8
	vfmadd213ss	96(%rsp,%r8,4), %xmm1, %xmm2
	addl	%r9d, %edx
	movslq	%edx, %rdx
	addl	$30, %ecx
	vfmadd231ss	(%rbx,%rdx,4), %xmm0, %xmm2
	vmovss	%xmm2, 96(%rsp,%r8,4)
	cmpl	$30, %r10d
	je	.L14
	leal	(%rdi,%rcx), %r8d
	movslq	%ecx, %rcx
	vmovss	96(%rsp,%rcx,4), %xmm7
	leal	(%rsi,%r8), %edx
	movslq	%edx, %rdx
	vfmadd132ss	(%rbx,%rdx,4), %xmm7, %xmm1
	leal	(%r9,%r8), %edx
	movslq	%edx, %rdx
	movslq	%eax, %r10
	addq	$2, %rax
	vfmadd231ss	(%rbx,%rdx,4), %xmm0, %xmm1
	leal	-1(%rax), %edx
	vmovss	%xmm1, 96(%rsp,%rcx,4)
	cmpl	%edx, 28(%rsp)
	jg	.L15
.L80:
	movq	40(%rsp), %r15
	jmp	.L21
	.p2align 4,,10
	.p2align 3
.L82:
	vmovups	96(%rsi), %ymm7
	vfmadd132ps	96(%r8), %ymm7, %ymm1
	vmovups	%ymm1, 96(%rsi)
.L19:
	incq	%r10
	cmpl	%r10d, 84(%rsp)
	jle	.L20
.L21:
	movl	(%r14,%r10,4), %edx
	vmovss	0(%r13,%r10,4), %xmm0
	imull	%r12d, %edx
	movslq	%edx, %r8
	addq	88(%rsp), %r8
	leaq	(%rbx,%r8,4), %rax
	shrq	$2, %rax
	negq	%rax
	andl	$7, %eax
	je	.L32
	leal	(%rdi,%rdx), %ecx
	movslq	%ecx, %rcx
	vmovss	(%rbx,%rcx,4), %xmm1
	vfmadd213ss	96(%rsp), %xmm0, %xmm1
	vmovss	%xmm1, 96(%rsp)
	cmpl	$1, %eax
	je	.L33
	movl	80(%rsp), %ecx
	addl	%edx, %ecx
	movslq	%ecx, %rcx
	vmovss	(%rbx,%rcx,4), %xmm1
	vfmadd213ss	100(%rsp), %xmm0, %xmm1
	vmovss	%xmm1, 100(%rsp)
	cmpl	$2, %eax
	je	.L34
	movl	76(%rsp), %ecx
	addl	%edx, %ecx
	movslq	%ecx, %rcx
	vmovss	(%rbx,%rcx,4), %xmm1
	vfmadd213ss	104(%rsp), %xmm0, %xmm1
	vmovss	%xmm1, 104(%rsp)
	cmpl	$3, %eax
	je	.L35
	movl	72(%rsp), %ecx
	addl	%edx, %ecx
	movslq	%ecx, %rcx
	vmovss	(%rbx,%rcx,4), %xmm1
	vfmadd213ss	108(%rsp), %xmm0, %xmm1
	vmovss	%xmm1, 108(%rsp)
	cmpl	$4, %eax
	je	.L36
	movl	68(%rsp), %ecx
	addl	%edx, %ecx
	movslq	%ecx, %rcx
	vmovss	(%rbx,%rcx,4), %xmm1
	vfmadd213ss	112(%rsp), %xmm0, %xmm1
	vmovss	%xmm1, 112(%rsp)
	cmpl	$5, %eax
	je	.L37
	movl	64(%rsp), %esi
	leal	5(%rdx,%rsi), %ecx
	movslq	%ecx, %rcx
	vmovss	(%rbx,%rcx,4), %xmm1
	vfmadd213ss	116(%rsp), %xmm0, %xmm1
	vmovss	%xmm1, 116(%rsp)
	cmpl	$6, %eax
	je	.L38
	leal	6(%rdx,%rsi), %ecx
	movslq	%ecx, %rcx
	vmovss	(%rbx,%rcx,4), %xmm1
	movl	$25, %r9d
	vfmadd213ss	120(%rsp), %xmm0, %xmm1
	movl	$7, %ecx
	vmovss	%xmm1, 120(%rsp)
.L16:
	movl	%eax, %r11d
	addq	%r11, %r8
	leaq	(%rbx,%r8,4), %r8
	vmovaps	(%r8), %ymm2
	leaq	(%r15,%r11,4), %rsi
	vbroadcastss	%xmm0, %ymm1
	vfmadd213ps	(%rsi), %ymm1, %ymm2
	movl	$32, %r11d
	subl	%eax, %r11d
	movl	%r11d, %eax
	shrl	$3, %eax
	vmovups	%ymm2, (%rsi)
	vmovaps	32(%r8), %ymm2
	vfmadd213ps	32(%rsi), %ymm1, %ymm2
	vmovups	%ymm2, 32(%rsi)
	vmovaps	64(%r8), %ymm2
	vfmadd213ps	64(%rsi), %ymm1, %ymm2
	vmovups	%ymm2, 64(%rsi)
	cmpl	$3, %eax
	jne	.L82
	leal	24(%rcx), %eax
	leal	(%rdi,%rax), %esi
	addl	%edx, %esi
	movslq	%esi, %rsi
	vmovss	(%rbx,%rsi,4), %xmm1
	cltq
	vfmadd213ss	96(%rsp,%rax,4), %xmm0, %xmm1
	vmovss	%xmm1, 96(%rsp,%rax,4)
	leal	25(%rcx), %eax
	cmpl	$25, %r9d
	je	.L19
	leal	(%rdi,%rax), %esi
	addl	%edx, %esi
	movslq	%esi, %rsi
	vmovss	(%rbx,%rsi,4), %xmm1
	cltq
	vfmadd213ss	96(%rsp,%rax,4), %xmm0, %xmm1
	vmovss	%xmm1, 96(%rsp,%rax,4)
	leal	26(%rcx), %eax
	cmpl	$26, %r9d
	je	.L19
	leal	(%rdi,%rax), %esi
	addl	%edx, %esi
	movslq	%esi, %rsi
	vmovss	(%rbx,%rsi,4), %xmm1
	cltq
	vfmadd213ss	96(%rsp,%rax,4), %xmm0, %xmm1
	vmovss	%xmm1, 96(%rsp,%rax,4)
	leal	27(%rcx), %eax
	cmpl	$27, %r9d
	je	.L19
	leal	(%rdi,%rax), %esi
	addl	%edx, %esi
	movslq	%esi, %rsi
	vmovss	(%rbx,%rsi,4), %xmm1
	cltq
	vfmadd213ss	96(%rsp,%rax,4), %xmm0, %xmm1
	vmovss	%xmm1, 96(%rsp,%rax,4)
	leal	28(%rcx), %eax
	cmpl	$28, %r9d
	je	.L19
	leal	(%rdi,%rax), %esi
	addl	%edx, %esi
	movslq	%esi, %rsi
	vmovss	(%rbx,%rsi,4), %xmm1
	cltq
	vfmadd213ss	96(%rsp,%rax,4), %xmm0, %xmm1
	vmovss	%xmm1, 96(%rsp,%rax,4)
	leal	29(%rcx), %eax
	cmpl	$29, %r9d
	je	.L19
	leal	(%rdi,%rax), %esi
	addl	%edx, %esi
	movslq	%esi, %rsi
	vmovss	(%rbx,%rsi,4), %xmm1
	cltq
	vfmadd213ss	96(%rsp,%rax,4), %xmm0, %xmm1
	addl	$30, %ecx
	vmovss	%xmm1, 96(%rsp,%rax,4)
	cmpl	$30, %r9d
	je	.L19
	leal	(%rdi,%rcx), %eax
	movslq	%ecx, %rcx
	vmovss	96(%rsp,%rcx,4), %xmm7
	addl	%edx, %eax
	cltq
	vfmadd132ss	(%rbx,%rax,4), %xmm7, %xmm0
	incq	%r10
	vmovss	%xmm0, 96(%rsp,%rcx,4)
	cmpl	%r10d, 84(%rsp)
	jg	.L21
.L20:
	movq	48(%rsp), %rax
	vmovdqa64	(%r15), %xmm7
	addq	$32, 88(%rsp)
	vmovups	%xmm7, (%rax)
	vmovdqa64	16(%r15), %xmm7
	subq	$-128, %rax
	vmovups	%xmm7, -112(%rax)
	vmovdqa64	32(%r15), %xmm7
	addl	$32, 80(%rsp)
	vmovups	%xmm7, -96(%rax)
	vmovdqa64	48(%r15), %xmm7
	addl	$32, 76(%rsp)
	vmovups	%xmm7, -80(%rax)
	vmovdqa64	64(%r15), %xmm7
	addl	$32, 72(%rsp)
	vmovups	%xmm7, -64(%rax)
	vmovdqa64	80(%r15), %xmm7
	addl	$32, 68(%rsp)
	vmovups	%xmm7, -48(%rax)
	vmovdqa64	96(%r15), %xmm7
	vmovups	%xmm7, -32(%rax)
	vmovdqa64	112(%r15), %xmm7
	vmovups	%xmm7, -16(%rax)
	movq	%rax, 48(%rsp)
	movq	88(%rsp), %rax
	cmpl	%eax, %r12d
	jg	.L4
	addq	$4, 32(%rsp)
	movl	20(%rsp), %edx
	addl	%edx, 24(%rsp)
	movq	32(%rsp), %rax
	cmpq	8(%rsp), %rax
	jne	.L22
	vzeroupper
.L1:
	movq	248(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L83
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
.L33:
	.cfi_restore_state
	movl	$31, %r9d
	movl	$1, %ecx
	jmp	.L16
.L32:
	movl	$32, %r9d
	xorl	%ecx, %ecx
	jmp	.L16
.L34:
	movl	$30, %r9d
	movl	$2, %ecx
	jmp	.L16
.L35:
	movl	$29, %r9d
	movl	$3, %ecx
	jmp	.L16
.L36:
	movl	$28, %r9d
	movl	$4, %ecx
	jmp	.L16
.L37:
	movl	$27, %r9d
	movl	$5, %ecx
	jmp	.L16
.L38:
	movl	$26, %r9d
	movl	$6, %ecx
	jmp	.L16
.L26:
	movl	$31, %r10d
	movl	$1, %ecx
	jmp	.L11
.L25:
	movl	$32, %r10d
	xorl	%ecx, %ecx
	jmp	.L11
.L28:
	movl	$29, %r10d
	movl	$3, %ecx
	jmp	.L11
.L27:
	movl	$30, %r10d
	movl	$2, %ecx
	jmp	.L11
.L24:
	movslq	60(%rsp), %r10
	jmp	.L21
.L31:
	movl	$26, %r10d
	movl	$6, %ecx
	jmp	.L11
.L30:
	movl	$27, %r10d
	movl	$5, %ecx
	jmp	.L11
.L29:
	movl	$28, %r10d
	movl	$4, %ecx
	jmp	.L11
.L83:
	call	__stack_chk_fail@PLT
	.cfi_endproc
.LFE878:
	.size	_Z8evaluateSt6vectorIiSaIiEES1_PKiS3_PKfS1_S5_Pf, .-_Z8evaluateSt6vectorIiSaIiEES1_PKiS3_PKfS1_S5_Pf
	.section	.rodata.cst32,"aM",@progbits,32
	.align 32
.LC0:
	.long	0
	.long	1
	.long	2
	.long	3
	.long	4
	.long	5
	.long	6
	.long	7
	.align 32
.LC1:
	.long	8
	.long	9
	.long	10
	.long	11
	.long	12
	.long	13
	.long	14
	.long	15
	.align 32
.LC2:
	.long	24
	.long	25
	.long	26
	.long	27
	.long	28
	.long	29
	.long	30
	.long	31
	.align 32
.LC3:
	.long	16
	.long	17
	.long	18
	.long	19
	.long	20
	.long	21
	.long	22
	.long	23
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
