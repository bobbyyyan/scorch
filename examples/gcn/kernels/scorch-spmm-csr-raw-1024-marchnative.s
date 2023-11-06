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
	andq	$-32, %rsp
	subq	$4096, %rsp
	orq	$0, (%rsp)
	subq	$192, %rsp
	.cfi_offset 15, -24
	.cfi_offset 14, -32
	.cfi_offset 13, -40
	.cfi_offset 12, -48
	.cfi_offset 3, -56
	movq	24(%rbp), %rbx
	movq	%rcx, 144(%rsp)
	movq	%r8, 136(%rsp)
	movq	%rbx, 32(%rsp)
	movq	%rdx, %rax
	movq	(%rdi), %rdx
	movq	%fs:40, %rbx
	movq	%rbx, 4280(%rsp)
	xorl	%ebx, %ebx
	movl	4(%rdx), %ebx
	movq	(%rsi), %rdx
	movq	(%r9), %rcx
	movl	(%rdx), %edx
	movl	%ebx, 88(%rsp)
	movq	16(%rbp), %r14
	movl	4(%rcx), %r15d
	testl	%edx, %edx
	jle	.L1
	testl	%r15d, %r15d
	jle	.L1
	movq	%rax, %rbx
	movq	%rax, 112(%rsp)
	leal	-1(%rdx), %eax
	leaq	4(%rbx,%rax,4), %rax
	leaq	160(%rsp), %r13
	movl	$0, 92(%rsp)
	movq	%rax, 72(%rsp)
	movq	%r13, %rax
	movl	%r15d, %r13d
	movq	%rax, %r15
.L20:
	movq	112(%rsp), %rax
	movq	32(%rsp), %rsi
	movl	(%rax), %ebx
	movl	4(%rax), %eax
	movl	%ebx, 100(%rsp)
	movl	%eax, %edi
	movl	%eax, 156(%rsp)
	movslq	92(%rsp), %rax
	movl	$4, 120(%rsp)
	leaq	(%rsi,%rax,4), %rax
	movq	%rax, 80(%rsp)
	leal	-1(%rdi), %esi
	movl	%edi, %eax
	leal	1(%rbx), %edi
	cmpl	%edi, %esi
	setg	%dl
	cmpl	$-2147483648, %eax
	setne	%al
	andl	%eax, %edx
	movslq	%ebx, %rax
	movq	144(%rsp), %rbx
	salq	$2, %rax
	addq	%rax, %rbx
	addq	136(%rsp), %rax
	movl	%esi, 64(%rsp)
	movl	%edi, 28(%rsp)
	movb	%dl, 71(%rsp)
	movq	%rbx, 48(%rsp)
	movq	%rax, 56(%rsp)
	movl	$3, 128(%rsp)
	movl	$2, 132(%rsp)
	movl	$1, 152(%rsp)
	xorl	%r12d, %r12d
.L4:
	xorl	%esi, %esi
	movq	%r15, %rdi
	movl	$4096, %edx
	movl	%r12d, 124(%rsp)
	call	memset@PLT
	movl	156(%rsp), %edi
	vmovdqa64	.LC1(%rip), %ymm6
	cmpl	%edi, 100(%rsp)
	movl	%r12d, %ebx
	jge	.L18
	movl	28(%rsp), %eax
	cmpb	$0, 71(%rsp)
	movl	%eax, 96(%rsp)
	je	.L22
	movq	48(%rsp), %rax
	movq	56(%rsp), %rcx
	movq	%rax, 104(%rsp)
	vpbroadcastd	124(%rsp), %ymm7
.L13:
	movq	104(%rsp), %rdi
	vmovss	(%rcx), %xmm5
	movl	(%rdi), %r8d
	imull	%r13d, %r8d
	movslq	%r8d, %rdx
	addq	%r12, %rdx
	leaq	(%r14,%rdx,4), %rax
	shrq	$2, %rax
	negq	%rax
	andl	$7, %eax
	je	.L23
	leal	(%r8,%rbx), %esi
	movslq	%esi, %rsi
	vmovss	(%r14,%rsi,4), %xmm1
	movl	4(%rdi), %esi
	vfmadd213ss	160(%rsp), %xmm5, %xmm1
	imull	%r13d, %esi
	vmovss	4(%rcx), %xmm0
	leal	(%rbx,%rsi), %edi
	movslq	%edi, %rdi
	vfmadd231ss	(%r14,%rdi,4), %xmm0, %xmm1
	vmovss	%xmm1, 160(%rsp)
	cmpl	$1, %eax
	je	.L24
	movl	152(%rsp), %r11d
	leal	(%r8,%r11), %edi
	movslq	%edi, %rdi
	vmovss	(%r14,%rdi,4), %xmm1
	leal	(%rsi,%r11), %edi
	vfmadd213ss	164(%rsp), %xmm5, %xmm1
	movslq	%edi, %rdi
	vfmadd231ss	(%r14,%rdi,4), %xmm0, %xmm1
	vmovss	%xmm1, 164(%rsp)
	cmpl	$2, %eax
	je	.L25
	movl	132(%rsp), %r11d
	leal	(%r8,%r11), %edi
	movslq	%edi, %rdi
	vmovss	(%r14,%rdi,4), %xmm1
	leal	(%rsi,%r11), %edi
	vfmadd213ss	168(%rsp), %xmm5, %xmm1
	movslq	%edi, %rdi
	vfmadd231ss	(%r14,%rdi,4), %xmm0, %xmm1
	vmovss	%xmm1, 168(%rsp)
	cmpl	$3, %eax
	je	.L26
	movl	128(%rsp), %r11d
	leal	(%r8,%r11), %edi
	movslq	%edi, %rdi
	vmovss	(%r14,%rdi,4), %xmm1
	leal	(%rsi,%r11), %edi
	vfmadd213ss	172(%rsp), %xmm5, %xmm1
	movslq	%edi, %rdi
	vfmadd231ss	(%r14,%rdi,4), %xmm0, %xmm1
	vmovss	%xmm1, 172(%rsp)
	cmpl	$4, %eax
	je	.L27
	movl	120(%rsp), %r11d
	leal	(%r8,%r11), %edi
	movslq	%edi, %rdi
	vmovss	(%r14,%rdi,4), %xmm1
	leal	(%rsi,%r11), %edi
	vfmadd213ss	176(%rsp), %xmm5, %xmm1
	movslq	%edi, %rdi
	vfmadd231ss	(%r14,%rdi,4), %xmm0, %xmm1
	vmovss	%xmm1, 176(%rsp)
	cmpl	$5, %eax
	je	.L28
	movl	124(%rsp), %r11d
	leal	5(%r11), %edi
	leal	(%r8,%rdi), %r9d
	movslq	%r9d, %r9
	vmovss	(%r14,%r9,4), %xmm1
	addl	%esi, %edi
	vfmadd213ss	180(%rsp), %xmm5, %xmm1
	movslq	%edi, %rdi
	vfmadd231ss	(%r14,%rdi,4), %xmm0, %xmm1
	vmovss	%xmm1, 180(%rsp)
	cmpl	$6, %eax
	je	.L29
	leal	6(%r11), %edi
	leal	(%r8,%rdi), %r9d
	movslq	%r9d, %r9
	vmovss	(%r14,%r9,4), %xmm1
	addl	%edi, %esi
	vfmadd213ss	184(%rsp), %xmm5, %xmm1
	movslq	%esi, %rsi
	movl	$1017, %r10d
	movl	$7, %edi
	vfmadd132ss	(%r14,%rsi,4), %xmm1, %xmm0
	vmovss	%xmm0, 184(%rsp)
.L9:
	movl	%eax, %esi
	addq	%rsi, %rdx
	movl	$1024, %r11d
	subl	%eax, %r11d
	leaq	(%r15,%rsi,4), %rax
	leaq	(%r14,%rdx,4), %rsi
	movq	104(%rsp), %rdx
	movq	%rsi, 40(%rsp)
	movl	4(%rdx), %r9d
	movl	%r11d, %esi
	imull	%r13d, %r9d
	vpbroadcastd	%edi, %ymm2
	shrl	$3, %esi
	salq	$5, %rsi
	vpaddd	.LC0(%rip), %ymm2, %ymm2
	movq	40(%rsp), %rdx
	vbroadcastss	%xmm5, %ymm9
	addq	%rax, %rsi
	vpbroadcastd	%r9d, %ymm8
	kxnorb	%k1, %k1, %k1
	.p2align 4,,10
	.p2align 3
.L11:
	vmovaps	(%rdx), %ymm3
	vmovdqa64	%ymm2, %ymm0
	vfmadd213ps	(%rax), %ymm9, %ymm3
	vpaddd	%ymm7, %ymm0, %ymm0
	vpaddd	%ymm8, %ymm0, %ymm0
	kmovb	%k1, %k2
	addq	$32, %rax
	vmovups	%ymm3, -32(%rax)
	vgatherdps	(%r14,%ymm0,4), %ymm1{%k2}
	vfmadd132ps	4(%rcx){1to8}, %ymm3, %ymm1
	vmovss	4(%rcx), %xmm4
	vpaddd	%ymm6, %ymm2, %ymm2
	addq	$32, %rdx
	vmovups	%ymm1, -32(%rax)
	cmpq	%rax, %rsi
	jne	.L11
	movl	%r11d, %edx
	andl	$-8, %edx
	movl	%edx, %eax
	addl	%edx, %edi
	notl	%eax
	cmpl	%edx, %r11d
	je	.L12
	leal	(%rbx,%rdi), %edx
	leal	(%r8,%rdx), %esi
	movslq	%esi, %rsi
	vmovss	(%r14,%rsi,4), %xmm0
	movslq	%edi, %r11
	vfmadd213ss	160(%rsp,%r11,4), %xmm5, %xmm0
	addl	%r9d, %edx
	movslq	%edx, %rdx
	vfmadd231ss	(%r14,%rdx,4), %xmm4, %xmm0
	leal	1(%rdi), %edx
	vmovss	%xmm0, 160(%rsp,%r11,4)
	addl	%r10d, %eax
	je	.L12
	leal	(%rbx,%rdx), %esi
	leal	(%r8,%rsi), %r10d
	movslq	%r10d, %r10
	vmovss	(%r14,%r10,4), %xmm0
	movslq	%edx, %rdx
	vfmadd213ss	160(%rsp,%rdx,4), %xmm5, %xmm0
	addl	%r9d, %esi
	movslq	%esi, %rsi
	vfmadd231ss	(%r14,%rsi,4), %xmm4, %xmm0
	leal	2(%rdi), %esi
	vmovss	%xmm0, 160(%rsp,%rdx,4)
	cmpl	$1, %eax
	je	.L12
	leal	(%rbx,%rsi), %edx
	leal	(%r8,%rdx), %r10d
	movslq	%r10d, %r10
	vmovss	(%r14,%r10,4), %xmm0
	movslq	%esi, %rsi
	vfmadd213ss	160(%rsp,%rsi,4), %xmm5, %xmm0
	addl	%r9d, %edx
	movslq	%edx, %rdx
	vfmadd231ss	(%r14,%rdx,4), %xmm4, %xmm0
	vmovss	%xmm0, 160(%rsp,%rsi,4)
	leal	3(%rdi), %esi
	cmpl	$2, %eax
	je	.L12
	leal	(%rbx,%rsi), %edx
	leal	(%r8,%rdx), %r10d
	movslq	%r10d, %r10
	vmovss	(%r14,%r10,4), %xmm0
	movslq	%esi, %rsi
	vfmadd213ss	160(%rsp,%rsi,4), %xmm5, %xmm0
	addl	%r9d, %edx
	movslq	%edx, %rdx
	vfmadd231ss	(%r14,%rdx,4), %xmm4, %xmm0
	vmovss	%xmm0, 160(%rsp,%rsi,4)
	leal	4(%rdi), %esi
	cmpl	$3, %eax
	je	.L12
	leal	(%rbx,%rsi), %edx
	leal	(%r8,%rdx), %r10d
	movslq	%r10d, %r10
	vmovss	(%r14,%r10,4), %xmm0
	movslq	%esi, %rsi
	vfmadd213ss	160(%rsp,%rsi,4), %xmm5, %xmm0
	addl	%r9d, %edx
	movslq	%edx, %rdx
	vfmadd231ss	(%r14,%rdx,4), %xmm4, %xmm0
	vmovss	%xmm0, 160(%rsp,%rsi,4)
	leal	5(%rdi), %esi
	cmpl	$4, %eax
	je	.L12
	leal	(%rbx,%rsi), %edx
	leal	(%r8,%rdx), %r10d
	movslq	%r10d, %r10
	vmovss	(%r14,%r10,4), %xmm0
	movslq	%esi, %rsi
	vfmadd213ss	160(%rsp,%rsi,4), %xmm5, %xmm0
	addl	%r9d, %edx
	movslq	%edx, %rdx
	addl	$6, %edi
	vfmadd231ss	(%r14,%rdx,4), %xmm4, %xmm0
	vmovss	%xmm0, 160(%rsp,%rsi,4)
	cmpl	$5, %eax
	je	.L12
	leal	(%rbx,%rdi), %eax
	movslq	%edi, %rdi
	vmovss	160(%rsp,%rdi,4), %xmm3
	leal	(%r8,%rax), %edx
	movslq	%edx, %rdx
	vfmadd132ss	(%r14,%rdx,4), %xmm3, %xmm5
	addl	%r9d, %eax
	cltq
	vfmadd132ss	(%r14,%rax,4), %xmm5, %xmm4
	vmovss	%xmm4, 160(%rsp,%rdi,4)
.L12:
	movl	96(%rsp), %eax
	addq	$8, 104(%rsp)
	leal	1(%rax), %r9d
	addl	$2, %eax
	movl	%eax, 96(%rsp)
	addq	$8, %rcx
	cmpl	64(%rsp), %eax
	jl	.L13
.L8:
	movslq	%r9d, %r9
.L19:
	movq	144(%rsp), %rax
	movl	(%rax,%r9,4), %r8d
	movq	136(%rsp), %rax
	imull	%r13d, %r8d
	vmovss	(%rax,%r9,4), %xmm2
	movslq	%r8d, %rax
	addq	%r12, %rax
	leaq	(%r14,%rax,4), %rdx
	shrq	$2, %rdx
	negq	%rdx
	andl	$7, %edx
	je	.L30
	leal	(%rbx,%r8), %ecx
	movslq	%ecx, %rcx
	vmovss	(%r14,%rcx,4), %xmm0
	vfmadd213ss	160(%rsp), %xmm2, %xmm0
	vmovss	%xmm0, 160(%rsp)
	cmpl	$1, %edx
	je	.L31
	movl	152(%rsp), %edi
	leal	(%rdi,%r8), %ecx
	movslq	%ecx, %rcx
	vmovss	(%r14,%rcx,4), %xmm0
	vfmadd213ss	164(%rsp), %xmm2, %xmm0
	vmovss	%xmm0, 164(%rsp)
	cmpl	$2, %edx
	je	.L32
	movl	132(%rsp), %edi
	leal	(%rdi,%r8), %ecx
	movslq	%ecx, %rcx
	vmovss	(%r14,%rcx,4), %xmm0
	vfmadd213ss	168(%rsp), %xmm2, %xmm0
	vmovss	%xmm0, 168(%rsp)
	cmpl	$3, %edx
	je	.L33
	movl	128(%rsp), %edi
	leal	(%rdi,%r8), %ecx
	movslq	%ecx, %rcx
	vmovss	(%r14,%rcx,4), %xmm0
	vfmadd213ss	172(%rsp), %xmm2, %xmm0
	vmovss	%xmm0, 172(%rsp)
	cmpl	$4, %edx
	je	.L34
	movl	120(%rsp), %edi
	leal	(%rdi,%r8), %ecx
	movslq	%ecx, %rcx
	vmovss	(%r14,%rcx,4), %xmm0
	vfmadd213ss	176(%rsp), %xmm2, %xmm0
	vmovss	%xmm0, 176(%rsp)
	cmpl	$5, %edx
	je	.L35
	movl	124(%rsp), %esi
	leal	5(%r8,%rsi), %ecx
	movslq	%ecx, %rcx
	vmovss	(%r14,%rcx,4), %xmm0
	vfmadd213ss	180(%rsp), %xmm2, %xmm0
	vmovss	%xmm0, 180(%rsp)
	cmpl	$6, %edx
	je	.L36
	leal	6(%r8,%rsi), %ecx
	movslq	%ecx, %rcx
	vmovss	(%r14,%rcx,4), %xmm0
	movl	$1017, %r11d
	vfmadd213ss	184(%rsp), %xmm2, %xmm0
	movl	$7, %esi
	vmovss	%xmm0, 184(%rsp)
.L14:
	movl	$1024, %r10d
	movl	%edx, %ecx
	subl	%edx, %r10d
	addq	%rcx, %rax
	leaq	(%r15,%rcx,4), %rdx
	movl	%r10d, %ecx
	shrl	$3, %ecx
	leaq	(%r14,%rax,4), %rdi
	vbroadcastss	%xmm2, %ymm1
	salq	$5, %rcx
	xorl	%eax, %eax
	.p2align 4,,10
	.p2align 3
.L16:
	vmovaps	(%rdi,%rax), %ymm0
	vfmadd213ps	(%rdx,%rax), %ymm1, %ymm0
	vmovups	%ymm0, (%rdx,%rax)
	addq	$32, %rax
	cmpq	%rax, %rcx
	jne	.L16
	movl	%r10d, %eax
	andl	$-8, %eax
	movl	%eax, %edx
	addl	%eax, %esi
	notl	%edx
	cmpl	%eax, %r10d
	je	.L17
	leal	(%rbx,%rsi), %eax
	addl	%r8d, %eax
	cltq
	vmovss	(%r14,%rax,4), %xmm0
	movslq	%esi, %rcx
	vfmadd213ss	160(%rsp,%rcx,4), %xmm2, %xmm0
	leal	1(%rsi), %eax
	vmovss	%xmm0, 160(%rsp,%rcx,4)
	addl	%r11d, %edx
	je	.L17
	leal	(%rbx,%rax), %ecx
	addl	%r8d, %ecx
	movslq	%ecx, %rcx
	vmovss	(%r14,%rcx,4), %xmm0
	cltq
	vfmadd213ss	160(%rsp,%rax,4), %xmm2, %xmm0
	vmovss	%xmm0, 160(%rsp,%rax,4)
	leal	2(%rsi), %eax
	cmpl	$1, %edx
	je	.L17
	leal	(%rbx,%rax), %ecx
	addl	%r8d, %ecx
	movslq	%ecx, %rcx
	vmovss	(%r14,%rcx,4), %xmm0
	cltq
	vfmadd213ss	160(%rsp,%rax,4), %xmm2, %xmm0
	vmovss	%xmm0, 160(%rsp,%rax,4)
	leal	3(%rsi), %eax
	cmpl	$2, %edx
	je	.L17
	leal	(%rbx,%rax), %ecx
	addl	%r8d, %ecx
	movslq	%ecx, %rcx
	vmovss	(%r14,%rcx,4), %xmm0
	cltq
	vfmadd213ss	160(%rsp,%rax,4), %xmm2, %xmm0
	vmovss	%xmm0, 160(%rsp,%rax,4)
	leal	4(%rsi), %eax
	cmpl	$3, %edx
	je	.L17
	leal	(%rbx,%rax), %ecx
	addl	%r8d, %ecx
	movslq	%ecx, %rcx
	vmovss	(%r14,%rcx,4), %xmm0
	cltq
	vfmadd213ss	160(%rsp,%rax,4), %xmm2, %xmm0
	vmovss	%xmm0, 160(%rsp,%rax,4)
	leal	5(%rsi), %eax
	cmpl	$4, %edx
	je	.L17
	leal	(%rbx,%rax), %ecx
	addl	%r8d, %ecx
	movslq	%ecx, %rcx
	vmovss	(%r14,%rcx,4), %xmm0
	cltq
	vfmadd213ss	160(%rsp,%rax,4), %xmm2, %xmm0
	addl	$6, %esi
	vmovss	%xmm0, 160(%rsp,%rax,4)
	cmpl	$5, %edx
	je	.L17
	leal	(%rbx,%rsi), %eax
	movslq	%esi, %rsi
	vmovss	160(%rsp,%rsi,4), %xmm7
	addl	%r8d, %eax
	cltq
	vfmadd132ss	(%r14,%rax,4), %xmm7, %xmm2
	vmovss	%xmm2, 160(%rsp,%rsi,4)
.L17:
	incq	%r9
	cmpl	%r9d, 156(%rsp)
	jg	.L19
.L18:
	movq	80(%rsp), %rbx
	movl	$4096, %edx
	movq	%r15, %rsi
	movq	%rbx, %rdi
	vzeroupper
	call	memcpy@PLT
	addq	$4096, %rax
	addq	$1024, %r12
	addl	$1024, 152(%rsp)
	addl	$1024, 132(%rsp)
	addl	$1024, 128(%rsp)
	addl	$1024, 120(%rsp)
	movq	%rax, 80(%rsp)
	cmpl	%r12d, %r13d
	jg	.L4
	addq	$4, 112(%rsp)
	movl	88(%rsp), %esi
	addl	%esi, 92(%rsp)
	movq	112(%rsp), %rax
	cmpq	%rax, 72(%rsp)
	jne	.L20
.L1:
	movq	4280(%rsp), %rax
	xorq	%fs:40, %rax
	jne	.L79
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
.L31:
	.cfi_restore_state
	movl	$1023, %r11d
	movl	$1, %esi
	jmp	.L14
.L30:
	movl	$1024, %r11d
	xorl	%esi, %esi
	jmp	.L14
.L32:
	movl	$1022, %r11d
	movl	$2, %esi
	jmp	.L14
.L33:
	movl	$1021, %r11d
	movl	$3, %esi
	jmp	.L14
.L34:
	movl	$1020, %r11d
	movl	$4, %esi
	jmp	.L14
.L35:
	movl	$1019, %r11d
	movl	$5, %esi
	jmp	.L14
.L36:
	movl	$1018, %r11d
	movl	$6, %esi
	jmp	.L14
.L29:
	movl	$1018, %r10d
	movl	$6, %edi
	jmp	.L9
.L28:
	movl	$1019, %r10d
	movl	$5, %edi
	jmp	.L9
.L23:
	movl	$1024, %r10d
	xorl	%edi, %edi
	jmp	.L9
.L22:
	movl	100(%rsp), %r9d
	jmp	.L8
.L27:
	movl	$1020, %r10d
	movl	$4, %edi
	jmp	.L9
.L26:
	movl	$1021, %r10d
	movl	$3, %edi
	jmp	.L9
.L25:
	movl	$1022, %r10d
	movl	$2, %edi
	jmp	.L9
.L24:
	movl	$1023, %r10d
	movl	$1, %edi
	jmp	.L9
.L79:
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
	.long	8
	.long	8
	.long	8
	.long	8
	.long	8
	.long	8
	.long	8
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
