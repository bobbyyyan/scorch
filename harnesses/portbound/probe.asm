
probe:     file format elf64-x86-64


Disassembly of section .init:

0000000000001000 <_init>:
    1000:	endbr64 
    1004:	sub    $0x8,%rsp
    1008:	mov    0x5fe1(%rip),%rax        # 6ff0 <__gmon_start__@Base>
    100f:	test   %rax,%rax
    1012:	je     1016 <_init+0x16>
    1014:	call   *%rax
    1016:	add    $0x8,%rsp
    101a:	ret    

Disassembly of section .plt:

0000000000001020 <.plt>:
    1020:	push   0x5f42(%rip)        # 6f68 <_GLOBAL_OFFSET_TABLE_+0x8>
    1026:	bnd jmp *0x5f43(%rip)        # 6f70 <_GLOBAL_OFFSET_TABLE_+0x10>
    102d:	nopl   (%rax)
    1030:	endbr64 
    1034:	push   $0x0
    1039:	bnd jmp 1020 <_init+0x20>
    103f:	nop
    1040:	endbr64 
    1044:	push   $0x1
    1049:	bnd jmp 1020 <_init+0x20>
    104f:	nop
    1050:	endbr64 
    1054:	push   $0x2
    1059:	bnd jmp 1020 <_init+0x20>
    105f:	nop
    1060:	endbr64 
    1064:	push   $0x3
    1069:	bnd jmp 1020 <_init+0x20>
    106f:	nop
    1070:	endbr64 
    1074:	push   $0x4
    1079:	bnd jmp 1020 <_init+0x20>
    107f:	nop
    1080:	endbr64 
    1084:	push   $0x5
    1089:	bnd jmp 1020 <_init+0x20>
    108f:	nop
    1090:	endbr64 
    1094:	push   $0x6
    1099:	bnd jmp 1020 <_init+0x20>
    109f:	nop
    10a0:	endbr64 
    10a4:	push   $0x7
    10a9:	bnd jmp 1020 <_init+0x20>
    10af:	nop
    10b0:	endbr64 
    10b4:	push   $0x8
    10b9:	bnd jmp 1020 <_init+0x20>
    10bf:	nop
    10c0:	endbr64 
    10c4:	push   $0x9
    10c9:	bnd jmp 1020 <_init+0x20>
    10cf:	nop
    10d0:	endbr64 
    10d4:	push   $0xa
    10d9:	bnd jmp 1020 <_init+0x20>
    10df:	nop
    10e0:	endbr64 
    10e4:	push   $0xb
    10e9:	bnd jmp 1020 <_init+0x20>
    10ef:	nop

Disassembly of section .plt.got:

00000000000010f0 <__cxa_finalize@plt>:
    10f0:	endbr64 
    10f4:	bnd jmp *0x5edd(%rip)        # 6fd8 <__cxa_finalize@GLIBC_2.2.5>
    10fb:	nopl   0x0(%rax,%rax,1)

Disassembly of section .plt.sec:

0000000000001100 <__printf_chk@plt>:
    1100:	endbr64 
    1104:	bnd jmp *0x5e6d(%rip)        # 6f78 <__printf_chk@GLIBC_2.3.4>
    110b:	nopl   0x0(%rax,%rax,1)

0000000000001110 <_ZSt20__throw_length_errorPKc@plt>:
    1110:	endbr64 
    1114:	bnd jmp *0x5e65(%rip)        # 6f80 <_ZSt20__throw_length_errorPKc@GLIBCXX_3.4>
    111b:	nopl   0x0(%rax,%rax,1)

0000000000001120 <memset@plt>:
    1120:	endbr64 
    1124:	bnd jmp *0x5e5d(%rip)        # 6f88 <memset@GLIBC_2.2.5>
    112b:	nopl   0x0(%rax,%rax,1)

0000000000001130 <memcpy@plt>:
    1130:	endbr64 
    1134:	bnd jmp *0x5e55(%rip)        # 6f90 <memcpy@GLIBC_2.14>
    113b:	nopl   0x0(%rax,%rax,1)

0000000000001140 <clock_gettime@plt>:
    1140:	endbr64 
    1144:	bnd jmp *0x5e4d(%rip)        # 6f98 <clock_gettime@GLIBC_2.17>
    114b:	nopl   0x0(%rax,%rax,1)

0000000000001150 <_Znwm@plt>:
    1150:	endbr64 
    1154:	bnd jmp *0x5e45(%rip)        # 6fa0 <_Znwm@GLIBCXX_3.4>
    115b:	nopl   0x0(%rax,%rax,1)

0000000000001160 <_ZdlPvm@plt>:
    1160:	endbr64 
    1164:	bnd jmp *0x5e3d(%rip)        # 6fa8 <_ZdlPvm@CXXABI_1.3.9>
    116b:	nopl   0x0(%rax,%rax,1)

0000000000001170 <__stack_chk_fail@plt>:
    1170:	endbr64 
    1174:	bnd jmp *0x5e35(%rip)        # 6fb0 <__stack_chk_fail@GLIBC_2.4>
    117b:	nopl   0x0(%rax,%rax,1)

0000000000001180 <strtol@plt>:
    1180:	endbr64 
    1184:	bnd jmp *0x5e2d(%rip)        # 6fb8 <strtol@GLIBC_2.2.5>
    118b:	nopl   0x0(%rax,%rax,1)

0000000000001190 <memmove@plt>:
    1190:	endbr64 
    1194:	bnd jmp *0x5e25(%rip)        # 6fc0 <memmove@GLIBC_2.2.5>
    119b:	nopl   0x0(%rax,%rax,1)

00000000000011a0 <__fprintf_chk@plt>:
    11a0:	endbr64 
    11a4:	bnd jmp *0x5e1d(%rip)        # 6fc8 <__fprintf_chk@GLIBC_2.3.4>
    11ab:	nopl   0x0(%rax,%rax,1)

00000000000011b0 <_Unwind_Resume@plt>:
    11b0:	endbr64 
    11b4:	bnd jmp *0x5e15(%rip)        # 6fd0 <_Unwind_Resume@GCC_3.0>
    11bb:	nopl   0x0(%rax,%rax,1)

Disassembly of section .text:

00000000000011c0 <main.cold>:
    11c0:	mov    -0x80(%rbp),%rsi
    11c4:	mov    -0x190(%rbp),%rdi
    11cb:	sub    %rdi,%rsi
    11ce:	vzeroupper 
    11d1:	call   1160 <_ZdlPvm@plt>
    11d6:	lea    -0xb0(%rbp),%rdi
    11dd:	call   3fd0 <_ZNSt6vectorIS_IdSaIdEESaIS1_EED1Ev>
    11e2:	mov    -0x170(%rbp),%rdi
    11e9:	mov    $0x58,%esi
    11ee:	call   1160 <_ZdlPvm@plt>
    11f3:	mov    -0xd0(%rbp),%rdi
    11fa:	mov    -0xc0(%rbp),%rsi
    1201:	sub    %rdi,%rsi
    1204:	test   %rdi,%rdi
    1207:	je     120e <main.cold+0x4e>
    1209:	call   1160 <_ZdlPvm@plt>
    120e:	mov    -0xf0(%rbp),%rdi
    1215:	mov    -0xe0(%rbp),%rsi
    121c:	sub    %rdi,%rsi
    121f:	test   %rdi,%rdi
    1222:	je     1229 <main.cold+0x69>
    1224:	call   1160 <_ZdlPvm@plt>
    1229:	mov    -0x110(%rbp),%rdi
    1230:	mov    -0x100(%rbp),%rsi
    1237:	sub    %rdi,%rsi
    123a:	test   %rdi,%rdi
    123d:	je     1244 <main.cold+0x84>
    123f:	call   1160 <_ZdlPvm@plt>
    1244:	mov    -0x130(%rbp),%rdi
    124b:	mov    -0x120(%rbp),%rsi
    1252:	sub    %rdi,%rsi
    1255:	test   %rdi,%rdi
    1258:	je     125f <main.cold+0x9f>
    125a:	call   1160 <_ZdlPvm@plt>
    125f:	mov    %r14,%rdi
    1262:	call   11b0 <_Unwind_Resume@plt>
    1267:	nopw   0x0(%rax,%rax,1)

0000000000001270 <main>:
    1270:	endbr64 
    1274:	lea    0x8(%rsp),%r10
    1279:	and    $0xffffffffffffffe0,%rsp
    127d:	push   -0x8(%r10)
    1281:	push   %rbp
    1282:	mov    %rsp,%rbp
    1285:	push   %r15
    1287:	push   %r14
    1289:	push   %r13
    128b:	push   %r12
    128d:	push   %r10
    128f:	push   %rbx
    1290:	sub    $0x180,%rsp
    1297:	mov    %fs:0x28,%rax
    12a0:	mov    %rax,-0x38(%rbp)
    12a4:	xor    %eax,%eax
    12a6:	cmp    $0x1,%edi
    12a9:	jg     2865 <main+0x15f5>
    12af:	movl   $0x1000,-0x184(%rbp)
    12b9:	movl   $0x1f,-0x180(%rbp)
    12c3:	movl   $0x1000,-0x164(%rbp)
    12cd:	movl   $0x3039,-0x188(%rbp)
    12d7:	mov    -0x164(%rbp),%r15d
    12de:	lea    -0x70(%rbp),%r13
    12e2:	lea    -0x130(%rbp),%rdi
    12e9:	mov    %r13,%rdx
    12ec:	movslq %r15d,%r14
    12ef:	mov    %r14,%rsi
    12f2:	mov    %r14,%r12
    12f5:	call   3e10 <_ZNSt6vectorIiSaIiEEC1EmRKS0_>
    12fa:	lea    -0x110(%rbp),%rdi
    1301:	mov    %r13,%rdx
    1304:	mov    %r14,%rsi
    1307:	call   3ef0 <_ZNSt6vectorIfSaIfEEC1EmRKS0_>
    130c:	mov    -0x184(%rbp),%ebx
    1312:	lea    -0xf0(%rbp),%rdi
    1319:	mov    %r13,%rdx
    131c:	lea    0x2(%rbx),%esi
    131f:	movslq %esi,%rsi
    1322:	shl    $0x2,%rsi
    1326:	call   3ef0 <_ZNSt6vectorIfSaIfEEC1EmRKS0_>
    132b:	add    $0x2,%r12d
    132f:	lea    -0xd0(%rbp),%rdi
    1336:	mov    %r13,%rdx
    1339:	movslq %r12d,%rsi
    133c:	shl    $0x2,%rsi
    1340:	call   3ef0 <_ZNSt6vectorIfSaIfEEC1EmRKS0_>
    1345:	mov    -0x110(%rbp),%rax
    134c:	mov    -0x130(%rbp),%rdx
    1353:	lea    0x1(%r15),%ecx
    1357:	mov    %rax,-0x1a8(%rbp)
    135e:	mov    %rdx,-0x1b0(%rbp)
    1365:	cmp    %ebx,%ecx
    1367:	jge    28ee <main+0x167e>
    136d:	mov    -0x188(%rbp),%ecx
    1373:	test   %r15d,%r15d
    1376:	jle    1898 <main+0x628>
    137c:	mov    -0x164(%rbp),%edi
    1382:	mov    -0x1b0(%rbp),%r11
    1389:	mov    -0x188(%rbp),%ecx
    138f:	lea    -0x1(%rdi),%r10d
    1393:	lea    0x4(%r11),%r9
    1397:	mov    %r11,%r12
    139a:	shl    $0x2,%r10
    139e:	lea    (%r9,%r10,1),%r8
    13a2:	mov    %r8,%rax
    13a5:	sub    %r11,%rax
    13a8:	mov    %rax,%r15
    13ab:	sub    $0x4,%r15
    13af:	shr    $0x2,%r15
    13b3:	add    $0x1,%r15
    13b7:	and    $0x7,%r15d
    13bb:	je     14db <main+0x26b>
    13c1:	cmp    $0x1,%r15
    13c5:	je     14b3 <main+0x243>
    13cb:	cmp    $0x2,%r15
    13cf:	je     1494 <main+0x224>
    13d5:	cmp    $0x3,%r15
    13d9:	je     1475 <main+0x205>
    13df:	cmp    $0x4,%r15
    13e3:	je     1456 <main+0x1e6>
    13e5:	cmp    $0x5,%r15
    13e9:	je     1437 <main+0x1c7>
    13eb:	cmp    $0x6,%r15
    13ef:	je     1418 <main+0x1a8>
    13f1:	imul   $0x19660d,-0x188(%rbp),%ecx
    13fb:	xor    %edx,%edx
    13fd:	mov    -0x1b0(%rbp),%r14
    1404:	mov    %r9,%r12
    1407:	add    $0x3c6ef35f,%ecx
    140d:	mov    %ecx,%eax
    140f:	divl   -0x184(%rbp)
    1415:	mov    %edx,(%r14)
    1418:	imul   $0x19660d,%ecx,%ecx
    141e:	xor    %edx,%edx
    1420:	add    $0x4,%r12
    1424:	add    $0x3c6ef35f,%ecx
    142a:	mov    %ecx,%eax
    142c:	divl   -0x184(%rbp)
    1432:	mov    %edx,-0x4(%r12)
    1437:	imul   $0x19660d,%ecx,%ecx
    143d:	xor    %edx,%edx
    143f:	add    $0x4,%r12
    1443:	add    $0x3c6ef35f,%ecx
    1449:	mov    %ecx,%eax
    144b:	divl   -0x184(%rbp)
    1451:	mov    %edx,-0x4(%r12)
    1456:	imul   $0x19660d,%ecx,%ecx
    145c:	xor    %edx,%edx
    145e:	add    $0x4,%r12
    1462:	add    $0x3c6ef35f,%ecx
    1468:	mov    %ecx,%eax
    146a:	divl   -0x184(%rbp)
    1470:	mov    %edx,-0x4(%r12)
    1475:	imul   $0x19660d,%ecx,%ecx
    147b:	xor    %edx,%edx
    147d:	add    $0x4,%r12
    1481:	add    $0x3c6ef35f,%ecx
    1487:	mov    %ecx,%eax
    1489:	divl   -0x184(%rbp)
    148f:	mov    %edx,-0x4(%r12)
    1494:	imul   $0x19660d,%ecx,%ecx
    149a:	xor    %edx,%edx
    149c:	add    $0x4,%r12
    14a0:	add    $0x3c6ef35f,%ecx
    14a6:	mov    %ecx,%eax
    14a8:	divl   -0x184(%rbp)
    14ae:	mov    %edx,-0x4(%r12)
    14b3:	imul   $0x19660d,%ecx,%ecx
    14b9:	xor    %edx,%edx
    14bb:	add    $0x4,%r12
    14bf:	add    $0x3c6ef35f,%ecx
    14c5:	mov    %ecx,%eax
    14c7:	divl   -0x184(%rbp)
    14cd:	mov    %edx,-0x4(%r12)
    14d2:	cmp    %r8,%r12
    14d5:	je     15b3 <main+0x343>
    14db:	mov    -0x184(%rbp),%ebx
    14e1:	imul   $0x19660d,%ecx,%esi
    14e7:	xor    %edx,%edx
    14e9:	add    $0x3c6ef35f,%esi
    14ef:	mov    %esi,%eax
    14f1:	imul   $0x19660d,%esi,%r11d
    14f8:	div    %ebx
    14fa:	add    $0x3c6ef35f,%r11d
    1501:	mov    %r11d,%eax
    1504:	imul   $0x19660d,%r11d,%edi
    150b:	add    $0x3c6ef35f,%edi
    1511:	imul   $0x19660d,%edi,%r9d
    1518:	add    $0x3c6ef35f,%r9d
    151f:	imul   $0x19660d,%r9d,%r15d
    1526:	add    $0x3c6ef35f,%r15d
    152d:	imul   $0x19660d,%r15d,%r14d
    1534:	mov    %edx,(%r12)
    1538:	xor    %edx,%edx
    153a:	div    %ebx
    153c:	mov    %edi,%eax
    153e:	add    $0x3c6ef35f,%r14d
    1545:	imul   $0x19660d,%r14d,%ecx
    154c:	add    $0x3c6ef35f,%ecx
    1552:	mov    %edx,0x4(%r12)
    1557:	xor    %edx,%edx
    1559:	div    %ebx
    155b:	mov    %r9d,%eax
    155e:	mov    %edx,0x8(%r12)
    1563:	xor    %edx,%edx
    1565:	div    %ebx
    1567:	mov    %r15d,%eax
    156a:	mov    %edx,0xc(%r12)
    156f:	xor    %edx,%edx
    1571:	div    %ebx
    1573:	mov    %r14d,%eax
    1576:	mov    %edx,0x10(%r12)
    157b:	xor    %edx,%edx
    157d:	div    %ebx
    157f:	mov    %ecx,%eax
    1581:	imul   $0x19660d,%ecx,%ecx
    1587:	add    $0x3c6ef35f,%ecx
    158d:	mov    %edx,0x14(%r12)
    1592:	xor    %edx,%edx
    1594:	div    %ebx
    1596:	mov    %ecx,%eax
    1598:	mov    %edx,0x18(%r12)
    159d:	xor    %edx,%edx
    159f:	div    %ebx
    15a1:	add    $0x20,%r12
    15a5:	mov    %edx,-0x4(%r12)
    15aa:	cmp    %r8,%r12
    15ad:	jne    14e1 <main+0x271>
    15b3:	mov    -0x1a8(%rbp),%r8
    15ba:	vmovss 0x3b42(%rip),%xmm0        # 5104 <_IO_stdin_used+0x104>
    15c2:	lea    0x4(%r8),%r12
    15c6:	mov    %r8,%rsi
    15c9:	add    %r12,%r10
    15cc:	mov    %r10,%rbx
    15cf:	sub    %r8,%rbx
    15d2:	sub    $0x4,%rbx
    15d6:	shr    $0x2,%rbx
    15da:	add    $0x1,%rbx
    15de:	and    $0x7,%ebx
    15e1:	je     175f <main+0x4ef>
    15e7:	cmp    $0x1,%rbx
    15eb:	je     1729 <main+0x4b9>
    15f1:	cmp    $0x2,%rbx
    15f5:	je     16fd <main+0x48d>
    15fb:	cmp    $0x3,%rbx
    15ff:	je     16d0 <main+0x460>
    1605:	cmp    $0x4,%rbx
    1609:	je     16a4 <main+0x434>
    160f:	cmp    $0x5,%rbx
    1613:	je     1678 <main+0x408>
    1615:	cmp    $0x6,%rbx
    1619:	je     164b <main+0x3db>
    161b:	imul   $0x19660d,%ecx,%ecx
    1621:	vxorps %xmm4,%xmm4,%xmm4
    1625:	mov    -0x1a8(%rbp),%rdi
    162c:	add    $0x3c6ef35f,%ecx
    1632:	mov    %ecx,%esi
    1634:	shr    $0x8,%esi
    1637:	movzwl %si,%r11d
    163b:	mov    %r12,%rsi
    163e:	vcvtsi2ss %r11d,%xmm4,%xmm1
    1643:	vmulss %xmm0,%xmm1,%xmm2
    1647:	vmovss %xmm2,(%rdi)
    164b:	imul   $0x19660d,%ecx,%ecx
    1651:	vxorps %xmm5,%xmm5,%xmm5
    1655:	add    $0x4,%rsi
    1659:	add    $0x3c6ef35f,%ecx
    165f:	mov    %ecx,%r9d
    1662:	shr    $0x8,%r9d
    1666:	movzwl %r9w,%r15d
    166a:	vcvtsi2ss %r15d,%xmm5,%xmm3
    166f:	vmulss %xmm0,%xmm3,%xmm6
    1673:	vmovss %xmm6,-0x4(%rsi)
    1678:	imul   $0x19660d,%ecx,%ecx
    167e:	vxorps %xmm7,%xmm7,%xmm7
    1682:	add    $0x4,%rsi
    1686:	add    $0x3c6ef35f,%ecx
    168c:	mov    %ecx,%r14d
    168f:	shr    $0x8,%r14d
    1693:	movzwl %r14w,%eax
    1697:	vcvtsi2ss %eax,%xmm7,%xmm8
    169b:	vmulss %xmm0,%xmm8,%xmm9
    169f:	vmovss %xmm9,-0x4(%rsi)
    16a4:	imul   $0x19660d,%ecx,%ecx
    16aa:	vxorps %xmm10,%xmm10,%xmm10
    16af:	add    $0x4,%rsi
    16b3:	add    $0x3c6ef35f,%ecx
    16b9:	mov    %ecx,%edx
    16bb:	shr    $0x8,%edx
    16be:	movzwl %dx,%r8d
    16c2:	vcvtsi2ss %r8d,%xmm10,%xmm11
    16c7:	vmulss %xmm0,%xmm11,%xmm12
    16cb:	vmovss %xmm12,-0x4(%rsi)
    16d0:	imul   $0x19660d,%ecx,%ecx
    16d6:	vxorps %xmm13,%xmm13,%xmm13
    16db:	add    $0x4,%rsi
    16df:	add    $0x3c6ef35f,%ecx
    16e5:	mov    %ecx,%r12d
    16e8:	shr    $0x8,%r12d
    16ec:	movzwl %r12w,%ebx
    16f0:	vcvtsi2ss %ebx,%xmm13,%xmm14
    16f4:	vmulss %xmm0,%xmm14,%xmm15
    16f8:	vmovss %xmm15,-0x4(%rsi)
    16fd:	imul   $0x19660d,%ecx,%ecx
    1703:	vxorps %xmm4,%xmm4,%xmm4
    1707:	add    $0x4,%rsi
    170b:	add    $0x3c6ef35f,%ecx
    1711:	mov    %ecx,%r11d
    1714:	shr    $0x8,%r11d
    1718:	movzwl %r11w,%edi
    171c:	vcvtsi2ss %edi,%xmm4,%xmm1
    1720:	vmulss %xmm0,%xmm1,%xmm2
    1724:	vmovss %xmm2,-0x4(%rsi)
    1729:	imul   $0x19660d,%ecx,%ecx
    172f:	vxorps %xmm5,%xmm5,%xmm5
    1733:	add    $0x4,%rsi
    1737:	add    $0x3c6ef35f,%ecx
    173d:	mov    %ecx,%r9d
    1740:	shr    $0x8,%r9d
    1744:	movzwl %r9w,%r15d
    1748:	vcvtsi2ss %r15d,%xmm5,%xmm3
    174d:	vmulss %xmm0,%xmm3,%xmm6
    1751:	vmovss %xmm6,-0x4(%rsi)
    1756:	cmp    %rsi,%r10
    1759:	je     1898 <main+0x628>
    175f:	imul   $0x19660d,%ecx,%r14d
    1766:	vxorps %xmm7,%xmm7,%xmm7
    176a:	add    $0x20,%rsi
    176e:	add    $0x3c6ef35f,%r14d
    1775:	imul   $0x19660d,%r14d,%edx
    177c:	mov    %r14d,%ecx
    177f:	shr    $0x8,%ecx
    1782:	movzwl %cx,%eax
    1785:	add    $0x3c6ef35f,%edx
    178b:	vcvtsi2ss %eax,%xmm7,%xmm8
    178f:	imul   $0x19660d,%edx,%ebx
    1795:	mov    %edx,%r8d
    1798:	shr    $0x8,%r8d
    179c:	movzwl %r8w,%r12d
    17a0:	vcvtsi2ss %r12d,%xmm7,%xmm10
    17a5:	vmulss %xmm0,%xmm8,%xmm9
    17a9:	add    $0x3c6ef35f,%ebx
    17af:	imul   $0x19660d,%ebx,%r9d
    17b6:	mov    %ebx,%r11d
    17b9:	shr    $0x8,%r11d
    17bd:	movzwl %r11w,%edi
    17c1:	vcvtsi2ss %edi,%xmm7,%xmm12
    17c5:	vmulss %xmm0,%xmm10,%xmm11
    17c9:	add    $0x3c6ef35f,%r9d
    17d0:	imul   $0x19660d,%r9d,%eax
    17d7:	mov    %r9d,%r15d
    17da:	vmovss %xmm9,-0x20(%rsi)
    17df:	shr    $0x8,%r15d
    17e3:	movzwl %r15w,%r14d
    17e7:	vcvtsi2ss %r14d,%xmm7,%xmm14
    17ec:	vmulss %xmm0,%xmm12,%xmm13
    17f0:	add    $0x3c6ef35f,%eax
    17f5:	imul   $0x19660d,%eax,%r8d
    17fc:	mov    %eax,%ecx
    17fe:	vmovss %xmm11,-0x1c(%rsi)
    1803:	shr    $0x8,%ecx
    1806:	movzwl %cx,%edx
    1809:	vcvtsi2ss %edx,%xmm7,%xmm4
    180d:	vmulss %xmm0,%xmm14,%xmm15
    1811:	add    $0x3c6ef35f,%r8d
    1818:	imul   $0x19660d,%r8d,%r11d
    181f:	mov    %r8d,%r12d
    1822:	vmovss %xmm13,-0x18(%rsi)
    1827:	shr    $0x8,%r12d
    182b:	movzwl %r12w,%ebx
    182f:	vcvtsi2ss %ebx,%xmm7,%xmm2
    1833:	vmulss %xmm0,%xmm4,%xmm1
    1837:	add    $0x3c6ef35f,%r11d
    183e:	imul   $0x19660d,%r11d,%r15d
    1845:	mov    %r11d,%edi
    1848:	vmovss %xmm15,-0x14(%rsi)
    184d:	shr    $0x8,%edi
    1850:	movzwl %di,%r9d
    1854:	vcvtsi2ss %r9d,%xmm7,%xmm3
    1859:	lea    0x3c6ef35f(%r15),%ecx
    1860:	vmulss %xmm0,%xmm2,%xmm5
    1864:	mov    %ecx,%r14d
    1867:	vmovss %xmm1,-0x10(%rsi)
    186c:	shr    $0x8,%r14d
    1870:	movzwl %r14w,%eax
    1874:	vcvtsi2ss %eax,%xmm7,%xmm7
    1878:	vmulss %xmm0,%xmm3,%xmm6
    187c:	vmovss %xmm5,-0xc(%rsi)
    1881:	vmulss %xmm0,%xmm7,%xmm8
    1885:	vmovss %xmm6,-0x8(%rsi)
    188a:	vmovss %xmm8,-0x4(%rsi)
    188f:	cmp    %rsi,%r10
    1892:	jne    175f <main+0x4ef>
    1898:	mov    -0xf0(%rbp),%rdi
    189f:	mov    -0xe8(%rbp),%r10
    18a6:	sub    %rdi,%r10
    18a9:	mov    %rdi,-0x1a0(%rbp)
    18b0:	mov    %r10,%rsi
    18b3:	shr    $0x2,%rsi
    18b7:	je     1b93 <main+0x923>
    18bd:	lea    (%r10,%rdi,1),%r12
    18c1:	sub    $0x4,%r10
    18c5:	vmovss 0x3837(%rip),%xmm0        # 5104 <_IO_stdin_used+0x104>
    18cd:	shr    $0x2,%r10
    18d1:	add    $0x1,%r10
    18d5:	and    $0x7,%r10d
    18d9:	je     1a5a <main+0x7ea>
    18df:	cmp    $0x1,%r10
    18e3:	je     1a23 <main+0x7b3>
    18e9:	cmp    $0x2,%r10
    18ed:	je     19f7 <main+0x787>
    18f3:	cmp    $0x3,%r10
    18f7:	je     19cb <main+0x75b>
    18fd:	cmp    $0x4,%r10
    1901:	je     199f <main+0x72f>
    1907:	cmp    $0x5,%r10
    190b:	je     1972 <main+0x702>
    190d:	cmp    $0x6,%r10
    1911:	je     1944 <main+0x6d4>
    1913:	imul   $0x19660d,%ecx,%ecx
    1919:	vxorps %xmm9,%xmm9,%xmm9
    191e:	mov    -0x1a0(%rbp),%r11
    1925:	lea    0x4(%r11),%rdi
    1929:	add    $0x3c6ef35f,%ecx
    192f:	mov    %ecx,%edx
    1931:	shr    $0x8,%edx
    1934:	movzwl %dx,%ebx
    1937:	vcvtsi2ss %ebx,%xmm9,%xmm10
    193b:	vmulss %xmm0,%xmm10,%xmm11
    193f:	vmovss %xmm11,(%r11)
    1944:	imul   $0x19660d,%ecx,%ecx
    194a:	vxorps %xmm12,%xmm12,%xmm12
    194f:	add    $0x4,%rdi
    1953:	add    $0x3c6ef35f,%ecx
    1959:	mov    %ecx,%r9d
    195c:	shr    $0x8,%r9d
    1960:	movzwl %r9w,%r15d
    1964:	vcvtsi2ss %r15d,%xmm12,%xmm13
    1969:	vmulss %xmm0,%xmm13,%xmm14
    196d:	vmovss %xmm14,-0x4(%rdi)
    1972:	imul   $0x19660d,%ecx,%ecx
    1978:	vxorps %xmm15,%xmm15,%xmm15
    197d:	add    $0x4,%rdi
    1981:	add    $0x3c6ef35f,%ecx
    1987:	mov    %ecx,%r14d
    198a:	shr    $0x8,%r14d
    198e:	movzwl %r14w,%eax
    1992:	vcvtsi2ss %eax,%xmm15,%xmm4
    1996:	vmulss %xmm0,%xmm4,%xmm1
    199a:	vmovss %xmm1,-0x4(%rdi)
    199f:	imul   $0x19660d,%ecx,%ecx
    19a5:	vxorps %xmm2,%xmm2,%xmm2
    19a9:	add    $0x4,%rdi
    19ad:	add    $0x3c6ef35f,%ecx
    19b3:	mov    %ecx,%r10d
    19b6:	shr    $0x8,%r10d
    19ba:	movzwl %r10w,%esi
    19be:	vcvtsi2ss %esi,%xmm2,%xmm5
    19c2:	vmulss %xmm0,%xmm5,%xmm3
    19c6:	vmovss %xmm3,-0x4(%rdi)
    19cb:	imul   $0x19660d,%ecx,%ecx
    19d1:	vxorps %xmm6,%xmm6,%xmm6
    19d5:	add    $0x4,%rdi
    19d9:	add    $0x3c6ef35f,%ecx
    19df:	mov    %ecx,%r8d
    19e2:	shr    $0x8,%r8d
    19e6:	movzwl %r8w,%edx
    19ea:	vcvtsi2ss %edx,%xmm6,%xmm7
    19ee:	vmulss %xmm0,%xmm7,%xmm8
    19f2:	vmovss %xmm8,-0x4(%rdi)
    19f7:	imul   $0x19660d,%ecx,%ecx
    19fd:	vxorps %xmm9,%xmm9,%xmm9
    1a02:	add    $0x4,%rdi
    1a06:	add    $0x3c6ef35f,%ecx
    1a0c:	mov    %ecx,%ebx
    1a0e:	shr    $0x8,%ebx
    1a11:	movzwl %bx,%r11d
    1a15:	vcvtsi2ss %r11d,%xmm9,%xmm10
    1a1a:	vmulss %xmm0,%xmm10,%xmm11
    1a1e:	vmovss %xmm11,-0x4(%rdi)
    1a23:	imul   $0x19660d,%ecx,%ecx
    1a29:	vxorps %xmm12,%xmm12,%xmm12
    1a2e:	add    $0x4,%rdi
    1a32:	add    $0x3c6ef35f,%ecx
    1a38:	mov    %ecx,%r9d
    1a3b:	shr    $0x8,%r9d
    1a3f:	movzwl %r9w,%r15d
    1a43:	vcvtsi2ss %r15d,%xmm12,%xmm13
    1a48:	vmulss %xmm0,%xmm13,%xmm14
    1a4c:	vmovss %xmm14,-0x4(%rdi)
    1a51:	cmp    %r12,%rdi
    1a54:	je     1b93 <main+0x923>
    1a5a:	imul   $0x19660d,%ecx,%r14d
    1a61:	vxorps %xmm15,%xmm15,%xmm15
    1a66:	add    $0x20,%rdi
    1a6a:	add    $0x3c6ef35f,%r14d
    1a71:	imul   $0x19660d,%r14d,%r10d
    1a78:	mov    %r14d,%ecx
    1a7b:	shr    $0x8,%ecx
    1a7e:	movzwl %cx,%eax
    1a81:	add    $0x3c6ef35f,%r10d
    1a88:	vcvtsi2ss %eax,%xmm15,%xmm4
    1a8c:	imul   $0x19660d,%r10d,%edx
    1a93:	mov    %r10d,%esi
    1a96:	shr    $0x8,%esi
    1a99:	movzwl %si,%r8d
    1a9d:	vcvtsi2ss %r8d,%xmm15,%xmm2
    1aa2:	vmulss %xmm0,%xmm4,%xmm1
    1aa6:	add    $0x3c6ef35f,%edx
    1aac:	imul   $0x19660d,%edx,%r9d
    1ab3:	mov    %edx,%ebx
    1ab5:	shr    $0x8,%ebx
    1ab8:	movzwl %bx,%r11d
    1abc:	vcvtsi2ss %r11d,%xmm15,%xmm3
    1ac1:	vmulss %xmm0,%xmm2,%xmm5
    1ac5:	add    $0x3c6ef35f,%r9d
    1acc:	imul   $0x19660d,%r9d,%eax
    1ad3:	mov    %r9d,%r15d
    1ad6:	vmovss %xmm1,-0x20(%rdi)
    1adb:	shr    $0x8,%r15d
    1adf:	movzwl %r15w,%r14d
    1ae3:	vcvtsi2ss %r14d,%xmm15,%xmm7
    1ae8:	vmulss %xmm0,%xmm3,%xmm6
    1aec:	add    $0x3c6ef35f,%eax
    1af1:	imul   $0x19660d,%eax,%esi
    1af7:	mov    %eax,%ecx
    1af9:	vmovss %xmm5,-0x1c(%rdi)
    1afe:	shr    $0x8,%ecx
    1b01:	movzwl %cx,%r10d
    1b05:	vcvtsi2ss %r10d,%xmm15,%xmm9
    1b0a:	vmulss %xmm0,%xmm7,%xmm8
    1b0e:	add    $0x3c6ef35f,%esi
    1b14:	imul   $0x19660d,%esi,%ebx
    1b1a:	mov    %esi,%r8d
    1b1d:	vmovss %xmm6,-0x18(%rdi)
    1b22:	shr    $0x8,%r8d
    1b26:	movzwl %r8w,%edx
    1b2a:	vcvtsi2ss %edx,%xmm15,%xmm11
    1b2e:	vmulss %xmm0,%xmm9,%xmm10
    1b32:	add    $0x3c6ef35f,%ebx
    1b38:	imul   $0x19660d,%ebx,%r15d
    1b3f:	mov    %ebx,%r11d
    1b42:	vmovss %xmm8,-0x14(%rdi)
    1b47:	shr    $0x8,%r11d
    1b4b:	movzwl %r11w,%r9d
    1b4f:	vcvtsi2ss %r9d,%xmm15,%xmm13
    1b54:	lea    0x3c6ef35f(%r15),%ecx
    1b5b:	vmulss %xmm0,%xmm11,%xmm12
    1b5f:	mov    %ecx,%r14d
    1b62:	vmovss %xmm10,-0x10(%rdi)
    1b67:	shr    $0x8,%r14d
    1b6b:	movzwl %r14w,%eax
    1b6f:	vcvtsi2ss %eax,%xmm15,%xmm15
    1b73:	vmulss %xmm0,%xmm13,%xmm14
    1b77:	vmovss %xmm12,-0xc(%rdi)
    1b7c:	vmulss %xmm0,%xmm15,%xmm4
    1b80:	vmovss %xmm14,-0x8(%rdi)
    1b85:	vmovss %xmm4,-0x4(%rdi)
    1b8a:	cmp    %r12,%rdi
    1b8d:	jne    1a5a <main+0x7ea>
    1b93:	mov    -0xd0(%rbp),%r12
    1b9a:	mov    -0xc8(%rbp),%r10
    1ba1:	vmovq  -0x1a0(%rbp),%xmm0
    1ba9:	vmovq  -0x1b0(%rbp),%xmm2
    1bb1:	sub    %r12,%r10
    1bb4:	mov    %r12,-0x198(%rbp)
    1bbb:	vpinsrq $0x1,-0x1a8(%rbp),%xmm2,%xmm5
    1bc5:	mov    %r10,%rdi
    1bc8:	vpinsrq $0x1,%r12,%xmm0,%xmm1
    1bce:	shr    $0x2,%rdi
    1bd2:	vinserti128 $0x1,%xmm1,%ymm5,%ymm3
    1bd8:	je     1ed0 <main+0xc60>
    1bde:	mov    -0x198(%rbp),%rsi
    1be5:	vmovss 0x3517(%rip),%xmm8        # 5104 <_IO_stdin_used+0x104>
    1bed:	lea    (%r10,%rsi,1),%r8
    1bf1:	sub    $0x4,%r10
    1bf5:	mov    %rsi,%r9
    1bf8:	shr    $0x2,%r10
    1bfc:	add    $0x1,%r10
    1c00:	and    $0x7,%r10d
    1c04:	je     1d8c <main+0xb1c>
    1c0a:	cmp    $0x1,%r10
    1c0e:	je     1d53 <main+0xae3>
    1c14:	cmp    $0x2,%r10
    1c18:	je     1d26 <main+0xab6>
    1c1e:	cmp    $0x3,%r10
    1c22:	je     1cfb <main+0xa8b>
    1c28:	cmp    $0x4,%r10
    1c2c:	je     1ccd <main+0xa5d>
    1c32:	cmp    $0x5,%r10
    1c36:	je     1c9f <main+0xa2f>
    1c38:	cmp    $0x6,%r10
    1c3c:	je     1c6f <main+0x9ff>
    1c3e:	imul   $0x19660d,%ecx,%ecx
    1c44:	vxorps %xmm6,%xmm6,%xmm6
    1c48:	mov    -0x198(%rbp),%r11
    1c4f:	lea    0x4(%r11),%r9
    1c53:	add    $0x3c6ef35f,%ecx
    1c59:	mov    %ecx,%edx
    1c5b:	shr    $0x8,%edx
    1c5e:	movzwl %dx,%ebx
    1c61:	vcvtsi2ss %ebx,%xmm6,%xmm7
    1c65:	vmulss %xmm8,%xmm7,%xmm9
    1c6a:	vmovss %xmm9,(%r11)
    1c6f:	imul   $0x19660d,%ecx,%ecx
    1c75:	vxorps %xmm10,%xmm10,%xmm10
    1c7a:	add    $0x4,%r9
    1c7e:	add    $0x3c6ef35f,%ecx
    1c84:	mov    %ecx,%r15d
    1c87:	shr    $0x8,%r15d
    1c8b:	movzwl %r15w,%r14d
    1c8f:	vcvtsi2ss %r14d,%xmm10,%xmm11
    1c94:	vmulss %xmm8,%xmm11,%xmm12
    1c99:	vmovss %xmm12,-0x4(%r9)
    1c9f:	imul   $0x19660d,%ecx,%ecx
    1ca5:	vxorps %xmm13,%xmm13,%xmm13
    1caa:	add    $0x4,%r9
    1cae:	add    $0x3c6ef35f,%ecx
    1cb4:	mov    %ecx,%eax
    1cb6:	shr    $0x8,%eax
    1cb9:	movzwl %ax,%r12d
    1cbd:	vcvtsi2ss %r12d,%xmm13,%xmm14
    1cc2:	vmulss %xmm8,%xmm14,%xmm15
    1cc7:	vmovss %xmm15,-0x4(%r9)
    1ccd:	imul   $0x19660d,%ecx,%ecx
    1cd3:	vxorps %xmm4,%xmm4,%xmm4
    1cd7:	add    $0x4,%r9
    1cdb:	add    $0x3c6ef35f,%ecx
    1ce1:	mov    %ecx,%r10d
    1ce4:	shr    $0x8,%r10d
    1ce8:	movzwl %r10w,%edi
    1cec:	vcvtsi2ss %edi,%xmm4,%xmm0
    1cf0:	vmulss %xmm8,%xmm0,%xmm1
    1cf5:	vmovss %xmm1,-0x4(%r9)
    1cfb:	imul   $0x19660d,%ecx,%ecx
    1d01:	vxorps %xmm2,%xmm2,%xmm2
    1d05:	add    $0x4,%r9
    1d09:	add    $0x3c6ef35f,%ecx
    1d0f:	mov    %ecx,%esi
    1d11:	shr    $0x8,%esi
    1d14:	movzwl %si,%edx
    1d17:	vcvtsi2ss %edx,%xmm2,%xmm5
    1d1b:	vmulss %xmm8,%xmm5,%xmm6
    1d20:	vmovss %xmm6,-0x4(%r9)
    1d26:	imul   $0x19660d,%ecx,%ecx
    1d2c:	vxorps %xmm7,%xmm7,%xmm7
    1d30:	add    $0x4,%r9
    1d34:	add    $0x3c6ef35f,%ecx
    1d3a:	mov    %ecx,%ebx
    1d3c:	shr    $0x8,%ebx
    1d3f:	movzwl %bx,%r11d
    1d43:	vcvtsi2ss %r11d,%xmm7,%xmm9
    1d48:	vmulss %xmm8,%xmm9,%xmm10
    1d4d:	vmovss %xmm10,-0x4(%r9)
    1d53:	imul   $0x19660d,%ecx,%ecx
    1d59:	vxorps %xmm11,%xmm11,%xmm11
    1d5e:	add    $0x4,%r9
    1d62:	add    $0x3c6ef35f,%ecx
    1d68:	mov    %ecx,%r15d
    1d6b:	shr    $0x8,%r15d
    1d6f:	movzwl %r15w,%r14d
    1d73:	vcvtsi2ss %r14d,%xmm11,%xmm12
    1d78:	vmulss %xmm8,%xmm12,%xmm13
    1d7d:	vmovss %xmm13,-0x4(%r9)
    1d83:	cmp    %r9,%r8
    1d86:	je     1ed0 <main+0xc60>
    1d8c:	imul   $0x19660d,%ecx,%eax
    1d92:	vxorps %xmm14,%xmm14,%xmm14
    1d97:	add    $0x20,%r9
    1d9b:	add    $0x3c6ef35f,%eax
    1da0:	imul   $0x19660d,%eax,%r10d
    1da7:	mov    %eax,%ecx
    1da9:	shr    $0x8,%ecx
    1dac:	movzwl %cx,%r12d
    1db0:	add    $0x3c6ef35f,%r10d
    1db7:	vcvtsi2ss %r12d,%xmm14,%xmm15
    1dbc:	imul   $0x19660d,%r10d,%edx
    1dc3:	mov    %r10d,%edi
    1dc6:	shr    $0x8,%edi
    1dc9:	movzwl %di,%esi
    1dcc:	vcvtsi2ss %esi,%xmm14,%xmm0
    1dd0:	vmulss %xmm8,%xmm15,%xmm4
    1dd5:	add    $0x3c6ef35f,%edx
    1ddb:	imul   $0x19660d,%edx,%r15d
    1de2:	mov    %edx,%ebx
    1de4:	shr    $0x8,%ebx
    1de7:	movzwl %bx,%r11d
    1deb:	vcvtsi2ss %r11d,%xmm14,%xmm2
    1df0:	vmulss %xmm8,%xmm0,%xmm1
    1df5:	add    $0x3c6ef35f,%r15d
    1dfc:	imul   $0x19660d,%r15d,%r12d
    1e03:	mov    %r15d,%r14d
    1e06:	vmovss %xmm4,-0x20(%r9)
    1e0c:	shr    $0x8,%r14d
    1e10:	movzwl %r14w,%eax
    1e14:	vcvtsi2ss %eax,%xmm14,%xmm6
    1e18:	vmulss %xmm8,%xmm2,%xmm5
    1e1d:	add    $0x3c6ef35f,%r12d
    1e24:	imul   $0x19660d,%r12d,%edi
    1e2b:	mov    %r12d,%ecx
    1e2e:	vmovss %xmm1,-0x1c(%r9)
    1e34:	shr    $0x8,%ecx
    1e37:	movzwl %cx,%r10d
    1e3b:	vcvtsi2ss %r10d,%xmm14,%xmm9
    1e40:	vmulss %xmm8,%xmm6,%xmm7
    1e45:	add    $0x3c6ef35f,%edi
    1e4b:	imul   $0x19660d,%edi,%ebx
    1e51:	mov    %edi,%esi
    1e53:	vmovss %xmm5,-0x18(%r9)
    1e59:	shr    $0x8,%esi
    1e5c:	movzwl %si,%edx
    1e5f:	vcvtsi2ss %edx,%xmm14,%xmm11
    1e63:	vmulss %xmm8,%xmm9,%xmm10
    1e68:	add    $0x3c6ef35f,%ebx
    1e6e:	imul   $0x19660d,%ebx,%r14d
    1e75:	mov    %ebx,%r11d
    1e78:	vmovss %xmm7,-0x14(%r9)
    1e7e:	shr    $0x8,%r11d
    1e82:	movzwl %r11w,%r15d
    1e86:	vcvtsi2ss %r15d,%xmm14,%xmm13
    1e8b:	lea    0x3c6ef35f(%r14),%ecx
    1e92:	vmulss %xmm8,%xmm11,%xmm12
    1e97:	mov    %ecx,%eax
    1e99:	vmovss %xmm10,-0x10(%r9)
    1e9f:	shr    $0x8,%eax
    1ea2:	movzwl %ax,%r12d
    1ea6:	vcvtsi2ss %r12d,%xmm14,%xmm14
    1eab:	vmulss %xmm8,%xmm13,%xmm15
    1eb0:	vmovss %xmm12,-0xc(%r9)
    1eb6:	vmulss %xmm8,%xmm14,%xmm4
    1ebb:	vmovss %xmm15,-0x8(%r9)
    1ec1:	vmovss %xmm4,-0x4(%r9)
    1ec7:	cmp    %r9,%r8
    1eca:	jne    1d8c <main+0xb1c>
    1ed0:	mov    -0x164(%rbp),%r8d
    1ed7:	mov    $0x1,%edi
    1edc:	vmovdqa %ymm3,-0x70(%rbp)
    1ee1:	lea    -0x90(%rbp),%rbx
    1ee8:	mov    %rbx,%rsi
    1eeb:	mov    %r8d,-0x50(%rbp)
    1eef:	vzeroupper 
    1ef2:	call   1140 <clock_gettime@plt>
    1ef7:	vxorpd %xmm3,%xmm3,%xmm3
    1efb:	mov    %r13,%rdi
    1efe:	vcvtsi2sdq -0x90(%rbp),%xmm3,%xmm8
    1f07:	vcvtsi2sdq -0x88(%rbp),%xmm3,%xmm0
    1f10:	vfmadd132sd 0x31f7(%rip),%xmm0,%xmm8        # 5110 <_IO_stdin_used+0x110>
    1f19:	vmovsd %xmm8,-0x150(%rbp)
    1f21:	call   2ae0 <_Z2v0RK4Args>
    1f26:	vxorps %xmm2,%xmm2,%xmm2
    1f2a:	mov    $0x40,%r9d
    1f30:	vmovaps %xmm0,%xmm1
    1f34:	vmovss %xmm2,-0x144(%rbp)
    1f3c:	nopl   0x0(%rax)
    1f40:	vaddss -0x144(%rbp),%xmm1,%xmm5
    1f48:	vaddss %xmm1,%xmm5,%xmm6
    1f4c:	vaddss %xmm1,%xmm6,%xmm7
    1f50:	vaddss %xmm1,%xmm7,%xmm9
    1f54:	vaddss %xmm1,%xmm9,%xmm10
    1f58:	vaddss %xmm1,%xmm10,%xmm11
    1f5c:	vaddss %xmm1,%xmm11,%xmm12
    1f60:	vaddss %xmm1,%xmm12,%xmm13
    1f64:	vmovss %xmm13,-0x144(%rbp)
    1f6c:	sub    $0x8,%r9d
    1f70:	jne    1f40 <main+0xcd0>
    1f72:	mov    $0x1,%edi
    1f77:	mov    %rbx,%rsi
    1f7a:	call   1140 <clock_gettime@plt>
    1f7f:	vxorpd %xmm15,%xmm15,%xmm15
    1f84:	mov    $0x1,%r10d
    1f8a:	vmovsd 0x318e(%rip),%xmm1        # 5120 <_IO_stdin_used+0x120>
    1f92:	vcvtsi2sdq -0x90(%rbp),%xmm15,%xmm14
    1f9b:	mov    $0x58,%edi
    1fa0:	vcvtsi2sdq -0x88(%rbp),%xmm15,%xmm4
    1fa9:	vfmadd132sd 0x315e(%rip),%xmm4,%xmm14        # 5110 <_IO_stdin_used+0x110>
    1fb2:	vsubsd -0x150(%rbp),%xmm14,%xmm3
    1fba:	vmulsd 0x3156(%rip),%xmm3,%xmm8        # 5118 <_IO_stdin_used+0x118>
    1fc2:	vmaxsd 0x313e(%rip),%xmm8,%xmm0        # 5108 <_IO_stdin_used+0x108>
    1fca:	vdivsd %xmm0,%xmm1,%xmm2
    1fce:	vcvttsd2si %xmm2,%ecx
    1fd2:	test   %ecx,%ecx
    1fd4:	cmovg  %ecx,%r10d
    1fd8:	mov    %r10d,-0x148(%rbp)
    1fdf:	call   1150 <_Znwm@plt>
    1fe4:	mov    0x3155(%rip),%rdi        # 5140 <_IO_stdin_used+0x140>
    1feb:	vmovapd 0x314d(%rip),%ymm5        # 5140 <_IO_stdin_used+0x140>
    1ff3:	vpxor  %xmm7,%xmm7,%xmm7
    1ff7:	mov    %rax,-0x170(%rbp)
    1ffe:	vmovapd 0x313a(%rip),%xmm6        # 5140 <_IO_stdin_used+0x140>
    2006:	vmovdqu %xmm7,-0xa8(%rbp)
    200e:	mov    %rdi,0x50(%rax)
    2012:	mov    $0x108,%edi
    2017:	vmovupd %ymm5,(%rax)
    201b:	vmovupd %ymm5,0x20(%rax)
    2020:	vmovupd %xmm6,0x40(%rax)
    2025:	vzeroupper 
    2028:	call   1150 <_Znwm@plt>
    202d:	lea    0x108(%rax),%rdx
    2034:	lea    -0x140(%rbp),%r15
    203b:	vpxor  %xmm9,%xmm9,%xmm9
    2040:	mov    %rbx,%rdi
    2043:	mov    %rdx,-0xa0(%rbp)
    204a:	mov    $0xb,%esi
    204f:	mov    %rdx,-0xa8(%rbp)
    2056:	mov    %r15,%rdx
    2059:	mov    %rax,-0x160(%rbp)
    2060:	mov    %rax,-0xb0(%rbp)
    2067:	movq   $0x0,0xc0(%rax)
    2072:	movq   $0x0,0xc8(%rax)
    207d:	movq   $0x0,0xd0(%rax)
    2088:	movq   $0x0,0xd8(%rax)
    2093:	movq   $0x0,0xe0(%rax)
    209e:	movq   $0x0,0xe8(%rax)
    20a9:	movq   $0x0,0xf0(%rax)
    20b4:	movq   $0x0,0xf8(%rax)
    20bf:	movq   $0x0,0x100(%rax)
    20ca:	vmovdqu %ymm9,(%rax)
    20ce:	vmovdqu %ymm9,0x20(%rax)
    20d3:	vmovdqu %ymm9,0x40(%rax)
    20d8:	vmovdqu %ymm9,0x60(%rax)
    20dd:	vmovdqu %ymm9,0x80(%rax)
    20e5:	vmovdqu %ymm9,0xa0(%rax)
    20ed:	vzeroupper 
    20f0:	call   3e10 <_ZNSt6vectorIiSaIiEEC1EmRKS0_>
    20f5:	mov    -0x90(%rbp),%r11
    20fc:	mov    -0x188(%rbp),%r12d
    2103:	movl   $0x0,-0x168(%rbp)
    210d:	vmovdqa 0x304b(%rip),%ymm10        # 5160 <_IO_stdin_used+0x160>
    2115:	mov    0x300c(%rip),%r14        # 5128 <_IO_stdin_used+0x128>
    211c:	mov    %r11,%rax
    211f:	mov    -0x180(%rbp),%r8d
    2126:	xor    $0x5bf03635,%r12d
    212d:	mov    %r11,-0x190(%rbp)
    2134:	add    $0x2c,%rax
    2138:	mov    %r14,0x20(%r11)
    213c:	movl   $0xa,0x28(%r11)
    2144:	mov    %r12d,-0x17c(%rbp)
    214b:	mov    %rax,-0x178(%rbp)
    2152:	vmovdqu %ymm10,(%r11)
    2157:	test   %r8d,%r8d
    215a:	jle    28e6 <main+0x1676>
    2160:	vzeroupper 
    2163:	nopl   0x0(%rax,%rax,1)
    2168:	imul   $0x19660d,-0x17c(%rbp),%ebx
    2172:	mov    $0xba2e8ba3,%ecx
    2177:	mov    -0x190(%rbp),%r11
    217e:	mov    0x28(%r11),%eax
    2182:	lea    0x3c6ef35f(%rbx),%r9d
    2189:	mov    %r9,%r10
    218c:	imul   %rcx,%r9
    2190:	imul   $0x19660d,%r10d,%r8d
    2197:	mov    %r10d,%edx
    219a:	shr    $0x23,%r9
    219e:	lea    (%r9,%r9,4),%edi
    21a2:	lea    0x3c6ef35f(%r8),%ebx
    21a9:	lea    (%r9,%rdi,2),%esi
    21ad:	mov    %rbx,%r9
    21b0:	mov    $0xcccccccd,%edi
    21b5:	sub    %esi,%edx
    21b7:	imul   %rdi,%rbx
    21bb:	mov    %r9d,%ecx
    21be:	lea    (%r11,%rdx,4),%r14
    21c2:	mov    (%r14),%r12d
    21c5:	shr    $0x23,%rbx
    21c9:	mov    %r12d,0x28(%r11)
    21cd:	lea    (%rbx,%rbx,4),%r10d
    21d1:	mov    %eax,(%r14)
    21d4:	imul   $0x19660d,%r9d,%eax
    21db:	add    %r10d,%r10d
    21de:	mov    0x24(%r11),%r14d
    21e2:	sub    %r10d,%ecx
    21e5:	lea    (%r11,%rcx,4),%rdx
    21e9:	lea    0x3c6ef35f(%rax),%r12d
    21f0:	mov    (%rdx),%esi
    21f2:	imul   $0x38e38e39,%r12,%rbx
    21f9:	mov    %r12d,%r10d
    21fc:	imul   $0x19660d,%r12d,%eax
    2203:	mov    %esi,0x24(%r11)
    2207:	mov    %r14d,(%rdx)
    220a:	mov    0x20(%r11),%ecx
    220e:	shr    $0x21,%rbx
    2212:	add    $0x3c6ef35f,%eax
    2217:	lea    (%rbx,%rbx,8),%r9d
    221b:	sub    %r9d,%r10d
    221e:	imul   $0x19660d,%eax,%r9d
    2225:	mov    %eax,%esi
    2227:	lea    (%r11,%r10,4),%rdx
    222b:	and    $0x7,%esi
    222e:	mov    (%rdx),%r14d
    2231:	lea    (%r11,%rsi,4),%r12
    2235:	lea    0x3c6ef35f(%r9),%r10d
    223c:	mov    %r14d,0x20(%r11)
    2240:	imul   $0x19660d,%r10d,%r9d
    2247:	mov    %r10d,%esi
    224a:	mov    %ecx,(%rdx)
    224c:	imul   $0x24924925,%r10,%rcx
    2253:	mov    %r10d,%edx
    2256:	mov    (%r12),%ebx
    225a:	mov    0x1c(%r11),%r8d
    225e:	mov    $0xaaaaaaab,%r10d
    2264:	lea    0x3c6ef35f(%r9),%r14d
    226b:	mov    %ebx,0x1c(%r11)
    226f:	shr    $0x20,%rcx
    2273:	mov    %r8d,(%r12)
    2277:	mov    0x18(%r11),%r8d
    227b:	sub    %ecx,%edx
    227d:	shr    %edx
    227f:	add    %ecx,%edx
    2281:	shr    $0x2,%edx
    2284:	lea    0x0(,%rdx,8),%eax
    228b:	sub    %edx,%eax
    228d:	sub    %eax,%esi
    228f:	mov    %r14,%rax
    2292:	imul   %r10,%r14
    2296:	lea    (%r11,%rsi,4),%r12
    229a:	mov    %eax,%ecx
    229c:	mov    (%r12),%ebx
    22a0:	shr    $0x22,%r14
    22a4:	mov    %ebx,0x18(%r11)
    22a8:	imul   $0x19660d,%eax,%ebx
    22ae:	lea    (%r14,%r14,2),%edx
    22b2:	mov    %r8d,(%r12)
    22b6:	mov    0x14(%r11),%r8d
    22ba:	add    %edx,%edx
    22bc:	sub    %edx,%ecx
    22be:	lea    0x3c6ef35f(%rbx),%r9d
    22c5:	lea    (%r11,%rcx,4),%r12
    22c9:	mov    %r9,%r14
    22cc:	imul   %rdi,%r9
    22d0:	mov    (%r12),%esi
    22d4:	mov    %r14d,%eax
    22d7:	mov    %esi,0x14(%r11)
    22db:	shr    $0x22,%r9
    22df:	mov    %r8d,(%r12)
    22e3:	imul   $0x19660d,%r14d,%r8d
    22ea:	mov    0x10(%r11),%ecx
    22ee:	lea    (%r9,%r9,4),%edi
    22f2:	sub    %edi,%eax
    22f4:	add    $0x3c6ef35f,%r8d
    22fb:	lea    (%r11,%rax,4),%rdx
    22ff:	imul   $0x19660d,%r8d,%edi
    2306:	mov    (%rdx),%r12d
    2309:	mov    %r8d,%esi
    230c:	and    $0x3,%esi
    230f:	mov    %r12d,0x10(%r11)
    2313:	lea    (%r11,%rsi,4),%rbx
    2317:	mov    %ecx,(%rdx)
    2319:	lea    0x3c6ef35f(%rdi),%edx
    231f:	mov    0xc(%r11),%r9d
    2323:	mov    %rdx,%rax
    2326:	mov    (%rbx),%r14d
    2329:	imul   %r10,%rdx
    232d:	mov    %eax,%ecx
    232f:	mov    %r14d,0xc(%r11)
    2333:	mov    %r9d,(%rbx)
    2336:	shr    $0x21,%rdx
    233a:	mov    0x8(%r11),%r8d
    233e:	imul   $0x19660d,%eax,%ebx
    2344:	lea    (%rdx,%rdx,2),%r10d
    2348:	mov    %r11,-0x150(%rbp)
    234f:	sub    %r10d,%ecx
    2352:	lea    0x3c6ef35f(%rbx),%eax
    2358:	lea    (%r11,%rcx,4),%r12
    235c:	mov    %eax,%r9d
    235f:	mov    (%r12),%esi
    2363:	mov    %eax,-0x17c(%rbp)
    2369:	and    $0x1,%r9d
    236d:	lea    (%r11,%r9,4),%r14
    2371:	mov    %esi,0x8(%r11)
    2375:	mov    %r8d,(%r12)
    2379:	mov    0x4(%r11),%edi
    237d:	mov    (%r14),%edx
    2380:	mov    %edx,0x4(%r11)
    2384:	mov    %edi,(%r14)
    2387:	nopw   0x0(%rax,%rax,1)
    2390:	mov    -0x150(%rbp),%r11
    2397:	mov    %r15,%rsi
    239a:	mov    $0x1,%edi
    239f:	xor    %ebx,%ebx
    23a1:	movslq (%r11),%r14
    23a4:	call   1140 <clock_gettime@plt>
    23a9:	vxorpd %xmm11,%xmm11,%xmm11
    23ae:	mov    -0x148(%rbp),%r8d
    23b5:	lea    0x48e4(%rip),%rcx        # 6ca0 <_ZL4ARMS>
    23bc:	vcvtsi2sdq -0x140(%rbp),%xmm11,%xmm12
    23c5:	mov    %r14,%r10
    23c8:	vcvtsi2sdq -0x138(%rbp),%xmm11,%xmm13
    23d1:	shl    $0x4,%r10
    23d5:	vfmadd132sd 0x2d32(%rip),%xmm13,%xmm12        # 5110 <_IO_stdin_used+0x110>
    23de:	mov    0x8(%rcx,%r10,1),%r12
    23e3:	vmovsd %xmm12,-0x158(%rbp)
    23eb:	and    $0x7,%r8d
    23ef:	je     24de <main+0x126e>
    23f5:	cmp    $0x1,%r8d
    23f9:	je     24b9 <main+0x1249>
    23ff:	cmp    $0x2,%r8d
    2403:	je     24a0 <main+0x1230>
    2409:	cmp    $0x3,%r8d
    240d:	je     2487 <main+0x1217>
    240f:	cmp    $0x4,%r8d
    2413:	je     246e <main+0x11fe>
    2415:	cmp    $0x5,%r8d
    2419:	je     2455 <main+0x11e5>
    241b:	cmp    $0x6,%r8d
    241f:	je     243c <main+0x11cc>
    2421:	mov    %r13,%rdi
    2424:	call   *%r12
    2427:	vaddss -0x144(%rbp),%xmm0,%xmm15
    242f:	mov    $0x1,%ebx
    2434:	vmovss %xmm15,-0x144(%rbp)
    243c:	mov    %r13,%rdi
    243f:	call   *%r12
    2442:	vaddss -0x144(%rbp),%xmm0,%xmm14
    244a:	add    $0x1,%ebx
    244d:	vmovss %xmm14,-0x144(%rbp)
    2455:	mov    %r13,%rdi
    2458:	call   *%r12
    245b:	vaddss -0x144(%rbp),%xmm0,%xmm4
    2463:	add    $0x1,%ebx
    2466:	vmovss %xmm4,-0x144(%rbp)
    246e:	mov    %r13,%rdi
    2471:	call   *%r12
    2474:	vaddss -0x144(%rbp),%xmm0,%xmm3
    247c:	add    $0x1,%ebx
    247f:	vmovss %xmm3,-0x144(%rbp)
    2487:	mov    %r13,%rdi
    248a:	call   *%r12
    248d:	vaddss -0x144(%rbp),%xmm0,%xmm8
    2495:	add    $0x1,%ebx
    2498:	vmovss %xmm8,-0x144(%rbp)
    24a0:	mov    %r13,%rdi
    24a3:	call   *%r12
    24a6:	vaddss -0x144(%rbp),%xmm0,%xmm0
    24ae:	add    $0x1,%ebx
    24b1:	vmovss %xmm0,-0x144(%rbp)
    24b9:	mov    %r13,%rdi
    24bc:	call   *%r12
    24bf:	vaddss -0x144(%rbp),%xmm0,%xmm1
    24c7:	add    $0x1,%ebx
    24ca:	vmovss %xmm1,-0x144(%rbp)
    24d2:	cmp    %ebx,-0x148(%rbp)
    24d8:	je     25a0 <main+0x1330>
    24de:	mov    %r13,%rdi
    24e1:	call   *%r12
    24e4:	vaddss -0x144(%rbp),%xmm0,%xmm2
    24ec:	add    $0x1,%ebx
    24ef:	mov    %r13,%rdi
    24f2:	vmovss %xmm2,-0x144(%rbp)
    24fa:	call   *%r12
    24fd:	vaddss -0x144(%rbp),%xmm0,%xmm5
    2505:	mov    %r13,%rdi
    2508:	vmovss %xmm5,-0x144(%rbp)
    2510:	call   *%r12
    2513:	vaddss -0x144(%rbp),%xmm0,%xmm6
    251b:	mov    %r13,%rdi
    251e:	vmovss %xmm6,-0x144(%rbp)
    2526:	call   *%r12
    2529:	vaddss -0x144(%rbp),%xmm0,%xmm7
    2531:	mov    %r13,%rdi
    2534:	vmovss %xmm7,-0x144(%rbp)
    253c:	call   *%r12
    253f:	vaddss -0x144(%rbp),%xmm0,%xmm9
    2547:	mov    %r13,%rdi
    254a:	vmovss %xmm9,-0x144(%rbp)
    2552:	call   *%r12
    2555:	vaddss -0x144(%rbp),%xmm0,%xmm10
    255d:	mov    %r13,%rdi
    2560:	vmovss %xmm10,-0x144(%rbp)
    2568:	call   *%r12
    256b:	vaddss -0x144(%rbp),%xmm0,%xmm11
    2573:	mov    %r13,%rdi
    2576:	vmovss %xmm11,-0x144(%rbp)
    257e:	call   *%r12
    2581:	vaddss -0x144(%rbp),%xmm0,%xmm12
    2589:	add    $0x7,%ebx
    258c:	vmovss %xmm12,-0x144(%rbp)
    2594:	cmp    %ebx,-0x148(%rbp)
    259a:	jne    24de <main+0x126e>
    25a0:	mov    %r15,%rsi
    25a3:	mov    $0x1,%edi
    25a8:	call   1140 <clock_gettime@plt>
    25ad:	vxorpd %xmm13,%xmm13,%xmm13
    25b2:	mov    -0x160(%rbp),%rax
    25b9:	lea    (%r14,%r14,2),%rsi
    25bd:	vcvtsi2sdl -0x148(%rbp),%xmm13,%xmm3
    25c5:	vcvtsi2sdl -0x164(%rbp),%xmm13,%xmm8
    25cd:	lea    (%rax,%rsi,8),%rdi
    25d1:	vcvtsi2sdq -0x140(%rbp),%xmm13,%xmm15
    25da:	mov    0x8(%rdi),%rsi
    25de:	vcvtsi2sdq -0x138(%rbp),%xmm13,%xmm14
    25e7:	vfmadd132sd 0x2b20(%rip),%xmm14,%xmm15        # 5110 <_IO_stdin_used+0x110>
    25f0:	vsubsd -0x158(%rbp),%xmm15,%xmm4
    25f8:	vmulsd %xmm8,%xmm3,%xmm0
    25fd:	vdivsd %xmm0,%xmm4,%xmm2
    2601:	vmovsd %xmm2,-0x140(%rbp)
    2609:	cmp    0x10(%rdi),%rsi
    260d:	je     2850 <main+0x15e0>
    2613:	vmovsd %xmm2,(%rsi)
    2617:	add    $0x8,%rsi
    261b:	mov    %rsi,0x8(%rdi)
    261f:	mov    -0x170(%rbp),%r9
    2626:	lea    (%r9,%r14,8),%rdi
    262a:	vmovsd (%rdi),%xmm1
    262e:	vcomisd %xmm2,%xmm1
    2632:	jbe    2828 <main+0x15b8>
    2638:	addq   $0x4,-0x150(%rbp)
    2640:	mov    -0x150(%rbp),%r11
    2647:	vmovsd %xmm2,(%rdi)
    264b:	cmp    %r11,-0x178(%rbp)
    2652:	jne    2390 <main+0x1120>
    2658:	addl   $0x1,-0x168(%rbp)
    265f:	mov    -0x168(%rbp),%r14d
    2666:	cmp    -0x180(%rbp),%r14d
    266d:	jne    2168 <main+0xef8>
    2673:	mov    -0x188(%rbp),%r13d
    267a:	sub    $0x8,%rsp
    267e:	mov    -0x184(%rbp),%r9d
    2685:	vxorpd %xmm5,%xmm5,%xmm5
    2689:	mov    -0x180(%rbp),%ecx
    268f:	mov    -0x148(%rbp),%r8d
    2696:	lea    0x2a1b(%rip),%rsi        # 50b8 <_IO_stdin_used+0xb8>
    269d:	mov    $0x1,%edi
    26a2:	push   %r13
    26a4:	mov    -0x164(%rbp),%edx
    26aa:	mov    $0x2,%eax
    26af:	vcvtss2sd -0x144(%rbp),%xmm0,%xmm0
    26b7:	vcvtss2sd 0x4981(%rip),%xmm5,%xmm1        # 7040 <_ZL6g_sink>
    26bf:	call   1100 <__printf_chk@plt>
    26c4:	pop    %rax
    26c5:	lea    0x29a3(%rip),%rsi        # 506f <_IO_stdin_used+0x6f>
    26cc:	pop    %rdx
    26cd:	mov    $0x1,%edi
    26d2:	xor    %eax,%eax
    26d4:	call   1100 <__printf_chk@plt>
    26d9:	mov    -0x160(%rbp),%r12
    26e0:	lea    0x45b9(%rip),%r15        # 6ca0 <_ZL4ARMS>
    26e7:	lea    0x2995(%rip),%r14        # 5083 <_IO_stdin_used+0x83>
    26ee:	add    $0x8,%r12
    26f2:	nopw   0x0(%rax,%rax,1)
    26f8:	mov    -0x8(%r12),%r10
    26fd:	cmp    (%r12),%r10
    2701:	je     2747 <main+0x14d7>
    2703:	mov    (%r15),%r13
    2706:	xor    %ebx,%ebx
    2708:	nopl   0x0(%rax,%rax,1)
    2710:	vmovsd (%r10,%rbx,8),%xmm0
    2716:	mov    %rbx,%rcx
    2719:	mov    %r13,%rdx
    271c:	mov    %r14,%rsi
    271f:	mov    $0x1,%edi
    2724:	mov    $0x1,%eax
    2729:	call   1100 <__printf_chk@plt>
    272e:	mov    -0x8(%r12),%r10
    2733:	mov    (%r12),%rcx
    2737:	add    $0x1,%rbx
    273b:	sub    %r10,%rcx
    273e:	sar    $0x3,%rcx
    2742:	cmp    %rcx,%rbx
    2745:	jb     2710 <main+0x14a0>
    2747:	add    $0x10,%r15
    274b:	lea    0x45fe(%rip),%r8        # 6d50 <_DYNAMIC>
    2752:	add    $0x18,%r12
    2756:	cmp    %r8,%r15
    2759:	jne    26f8 <main+0x1488>
    275b:	mov    -0x190(%rbp),%rdi
    2762:	mov    -0x80(%rbp),%rsi
    2766:	xor    %r12d,%r12d
    2769:	sub    %rdi,%rsi
    276c:	call   1160 <_ZdlPvm@plt>
    2771:	lea    -0xb0(%rbp),%rdi
    2778:	call   3fd0 <_ZNSt6vectorIS_IdSaIdEESaIS1_EED1Ev>
    277d:	mov    -0x170(%rbp),%rdi
    2784:	mov    $0x58,%esi
    2789:	call   1160 <_ZdlPvm@plt>
    278e:	mov    -0x198(%rbp),%rdi
    2795:	test   %rdi,%rdi
    2798:	je     27a9 <main+0x1539>
    279a:	mov    -0xc0(%rbp),%rsi
    27a1:	sub    %rdi,%rsi
    27a4:	call   1160 <_ZdlPvm@plt>
    27a9:	mov    -0x1a0(%rbp),%rdi
    27b0:	test   %rdi,%rdi
    27b3:	je     27c4 <main+0x1554>
    27b5:	mov    -0xe0(%rbp),%rsi
    27bc:	sub    %rdi,%rsi
    27bf:	call   1160 <_ZdlPvm@plt>
    27c4:	mov    -0x1a8(%rbp),%rdi
    27cb:	test   %rdi,%rdi
    27ce:	je     27df <main+0x156f>
    27d0:	mov    -0x100(%rbp),%rsi
    27d7:	sub    %rdi,%rsi
    27da:	call   1160 <_ZdlPvm@plt>
    27df:	mov    -0x1b0(%rbp),%rdi
    27e6:	test   %rdi,%rdi
    27e9:	je     27fa <main+0x158a>
    27eb:	mov    -0x120(%rbp),%rsi
    27f2:	sub    %rdi,%rsi
    27f5:	call   1160 <_ZdlPvm@plt>
    27fa:	mov    -0x38(%rbp),%rax
    27fe:	sub    %fs:0x28,%rax
    2807:	jne    295b <main+0x16eb>
    280d:	lea    -0x30(%rbp),%rsp
    2811:	mov    %r12d,%eax
    2814:	pop    %rbx
    2815:	pop    %r9
    2817:	pop    %r12
    2819:	pop    %r13
    281b:	pop    %r14
    281d:	pop    %r15
    281f:	pop    %rbp
    2820:	lea    -0x8(%r9),%rsp
    2824:	ret    
    2825:	nopl   (%rax)
    2828:	addq   $0x4,-0x150(%rbp)
    2830:	mov    -0x150(%rbp),%rdx
    2837:	cmp    %rdx,-0x178(%rbp)
    283e:	jne    2390 <main+0x1120>
    2844:	jmp    2658 <main+0x13e8>
    2849:	nopl   0x0(%rax)
    2850:	mov    %r15,%rdx
    2853:	call   4050 <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_>
    2858:	vmovsd -0x140(%rbp),%xmm2
    2860:	jmp    261f <main+0x13af>
    2865:	mov    %edi,%r12d
    2868:	mov    0x8(%rsi),%rdi
    286c:	mov    %rsi,%rbx
    286f:	mov    $0xa,%edx
    2874:	xor    %esi,%esi
    2876:	call   1180 <strtol@plt>
    287b:	mov    %eax,-0x164(%rbp)
    2881:	cmp    $0x2,%r12d
    2885:	je     2938 <main+0x16c8>
    288b:	mov    0x10(%rbx),%rdi
    288f:	mov    $0xa,%edx
    2894:	xor    %esi,%esi
    2896:	call   1180 <strtol@plt>
    289b:	mov    %eax,-0x180(%rbp)
    28a1:	cmp    $0x3,%r12d
    28a5:	je     296f <main+0x16ff>
    28ab:	mov    0x18(%rbx),%rdi
    28af:	mov    $0xa,%edx
    28b4:	xor    %esi,%esi
    28b6:	call   1180 <strtol@plt>
    28bb:	mov    %eax,-0x184(%rbp)
    28c1:	cmp    $0x4,%r12d
    28c5:	je     2960 <main+0x16f0>
    28cb:	mov    0x20(%rbx),%rdi
    28cf:	mov    $0xa,%edx
    28d4:	xor    %esi,%esi
    28d6:	call   1180 <strtol@plt>
    28db:	mov    %eax,-0x188(%rbp)
    28e1:	jmp    12d7 <main+0x67>
    28e6:	vzeroupper 
    28e9:	jmp    2673 <main+0x1403>
    28ee:	mov    -0x184(%rbp),%ecx
    28f4:	mov    %r12d,%r8d
    28f7:	mov    $0x1,%esi
    28fc:	xor    %eax,%eax
    28fe:	mov    0x471b(%rip),%rdi        # 7020 <stderr@GLIBC_2.2.5>
    2905:	lea    0x2747(%rip),%rdx        # 5053 <_IO_stdin_used+0x53>
    290c:	call   11a0 <__fprintf_chk@plt>
    2911:	mov    -0xd0(%rbp),%rsi
    2918:	mov    -0xf0(%rbp),%rax
    291f:	mov    $0x2,%r12d
    2925:	mov    %rsi,-0x198(%rbp)
    292c:	mov    %rax,-0x1a0(%rbp)
    2933:	jmp    278e <main+0x151e>
    2938:	movl   $0x1000,-0x184(%rbp)
    2942:	movl   $0x1f,-0x180(%rbp)
    294c:	movl   $0x3039,-0x188(%rbp)
    2956:	jmp    12d7 <main+0x67>
    295b:	call   1170 <__stack_chk_fail@plt>
    2960:	movl   $0x3039,-0x188(%rbp)
    296a:	jmp    12d7 <main+0x67>
    296f:	movl   $0x1000,-0x184(%rbp)
    2979:	movl   $0x3039,-0x188(%rbp)
    2983:	jmp    12d7 <main+0x67>
    2988:	endbr64 
    298c:	mov    %rax,%r14
    298f:	vzeroupper 
    2992:	jmp    1244 <main.cold+0x84>
    2997:	endbr64 
    299b:	mov    %rax,%r14
    299e:	vzeroupper 
    29a1:	jmp    1229 <main.cold+0x69>
    29a6:	endbr64 
    29aa:	mov    %rax,%r14
    29ad:	vzeroupper 
    29b0:	jmp    11f3 <main.cold+0x33>
    29b5:	endbr64 
    29b9:	mov    %rax,%r14
    29bc:	vzeroupper 
    29bf:	jmp    11d6 <main.cold+0x16>
    29c4:	endbr64 
    29c8:	mov    %rax,%r14
    29cb:	vzeroupper 
    29ce:	jmp    120e <main.cold+0x4e>
    29d3:	endbr64 
    29d7:	mov    %rax,%r14
    29da:	vzeroupper 
    29dd:	jmp    11e2 <main.cold+0x22>
    29e2:	endbr64 
    29e6:	mov    %rax,%r14
    29e9:	jmp    11c0 <main.cold>
    29ee:	xchg   %ax,%ax

00000000000029f0 <_start>:
    29f0:	endbr64 
    29f4:	xor    %ebp,%ebp
    29f6:	mov    %rdx,%r9
    29f9:	pop    %rsi
    29fa:	mov    %rsp,%rdx
    29fd:	and    $0xfffffffffffffff0,%rsp
    2a01:	push   %rax
    2a02:	push   %rsp
    2a03:	xor    %r8d,%r8d
    2a06:	xor    %ecx,%ecx
    2a08:	lea    -0x179f(%rip),%rdi        # 1270 <main>
    2a0f:	call   *0x45cb(%rip)        # 6fe0 <__libc_start_main@GLIBC_2.34>
    2a15:	hlt    
    2a16:	cs nopw 0x0(%rax,%rax,1)

0000000000002a20 <deregister_tm_clones>:
    2a20:	lea    0x45f1(%rip),%rdi        # 7018 <__TMC_END__>
    2a27:	lea    0x45ea(%rip),%rax        # 7018 <__TMC_END__>
    2a2e:	cmp    %rdi,%rax
    2a31:	je     2a48 <deregister_tm_clones+0x28>
    2a33:	mov    0x45ae(%rip),%rax        # 6fe8 <_ITM_deregisterTMCloneTable@Base>
    2a3a:	test   %rax,%rax
    2a3d:	je     2a48 <deregister_tm_clones+0x28>
    2a3f:	jmp    *%rax
    2a41:	nopl   0x0(%rax)
    2a48:	ret    
    2a49:	nopl   0x0(%rax)

0000000000002a50 <register_tm_clones>:
    2a50:	lea    0x45c1(%rip),%rdi        # 7018 <__TMC_END__>
    2a57:	lea    0x45ba(%rip),%rsi        # 7018 <__TMC_END__>
    2a5e:	sub    %rdi,%rsi
    2a61:	mov    %rsi,%rax
    2a64:	shr    $0x3f,%rsi
    2a68:	sar    $0x3,%rax
    2a6c:	add    %rax,%rsi
    2a6f:	sar    %rsi
    2a72:	je     2a88 <register_tm_clones+0x38>
    2a74:	mov    0x457d(%rip),%rax        # 6ff8 <_ITM_registerTMCloneTable@Base>
    2a7b:	test   %rax,%rax
    2a7e:	je     2a88 <register_tm_clones+0x38>
    2a80:	jmp    *%rax
    2a82:	nopw   0x0(%rax,%rax,1)
    2a88:	ret    
    2a89:	nopl   0x0(%rax)

0000000000002a90 <__do_global_dtors_aux>:
    2a90:	endbr64 
    2a94:	cmpb   $0x0,0x458d(%rip)        # 7028 <completed.0>
    2a9b:	jne    2ac8 <__do_global_dtors_aux+0x38>
    2a9d:	push   %rbp
    2a9e:	cmpq   $0x0,0x4532(%rip)        # 6fd8 <__cxa_finalize@GLIBC_2.2.5>
    2aa6:	mov    %rsp,%rbp
    2aa9:	je     2ab7 <__do_global_dtors_aux+0x27>
    2aab:	mov    0x4556(%rip),%rdi        # 7008 <__dso_handle>
    2ab2:	call   10f0 <__cxa_finalize@plt>
    2ab7:	call   2a20 <deregister_tm_clones>
    2abc:	movb   $0x1,0x4565(%rip)        # 7028 <completed.0>
    2ac3:	pop    %rbp
    2ac4:	ret    
    2ac5:	nopl   (%rax)
    2ac8:	ret    
    2ac9:	nopl   0x0(%rax)

0000000000002ad0 <frame_dummy>:
    2ad0:	endbr64 
    2ad4:	jmp    2a50 <register_tm_clones>
    2ad9:	nopl   0x0(%rax)

0000000000002ae0 <_Z2v0RK4Args>:
    2ae0:	endbr64 
    2ae4:	mov    0x20(%rdi),%eax
    2ae7:	cmp    $0x1,%eax
    2aea:	jle    2ca8 <_Z2v0RK4Args+0x1c8>
    2af0:	sub    $0x2,%eax
    2af3:	mov    0x10(%rdi),%rdx
    2af7:	mov    (%rdi),%rcx
    2afa:	vxorps %xmm12,%xmm12,%xmm12
    2aff:	shr    %eax
    2b01:	mov    0x8(%rdi),%rsi
    2b05:	xor    %r11d,%r11d
    2b08:	vmovaps %xmm12,%xmm0
    2b0c:	lea    0x1(%rax),%r8d
    2b10:	shl    $0x3,%r8
    2b14:	lea    -0x8(%r8),%rdi
    2b18:	shr    $0x3,%rdi
    2b1c:	add    $0x1,%rdi
    2b20:	and    $0x3,%edi
    2b23:	je     2bc6 <_Z2v0RK4Args+0xe6>
    2b29:	cmp    $0x1,%rdi
    2b2d:	je     2b8f <_Z2v0RK4Args+0xaf>
    2b2f:	cmp    $0x2,%rdi
    2b33:	je     2b61 <_Z2v0RK4Args+0x81>
    2b35:	movslq (%rcx),%r9
    2b38:	movslq 0x4(%rcx),%r10
    2b3c:	mov    $0x8,%r11d
    2b42:	vbroadcastss (%rsi),%xmm0
    2b47:	vbroadcastss 0x4(%rsi),%xmm2
    2b4d:	shl    $0x4,%r9
    2b51:	shl    $0x4,%r10
    2b55:	vfmadd132ps (%rdx,%r9,1),%xmm12,%xmm0
    2b5b:	vfmadd231ps (%rdx,%r10,1),%xmm2,%xmm12
    2b61:	movslq (%rcx,%r11,1),%rax
    2b65:	movslq 0x4(%rcx,%r11,1),%rdi
    2b6a:	vbroadcastss (%rsi,%r11,1),%xmm3
    2b70:	vbroadcastss 0x4(%rsi,%r11,1),%xmm4
    2b77:	add    $0x8,%r11
    2b7b:	shl    $0x4,%rax
    2b7f:	shl    $0x4,%rdi
    2b83:	vfmadd231ps (%rdx,%rax,1),%xmm3,%xmm0
    2b89:	vfmadd231ps (%rdx,%rdi,1),%xmm4,%xmm12
    2b8f:	movslq (%rcx,%r11,1),%r9
    2b93:	movslq 0x4(%rcx,%r11,1),%r10
    2b98:	vbroadcastss (%rsi,%r11,1),%xmm5
    2b9e:	vbroadcastss 0x4(%rsi,%r11,1),%xmm6
    2ba5:	add    $0x8,%r11
    2ba9:	shl    $0x4,%r9
    2bad:	shl    $0x4,%r10
    2bb1:	vfmadd231ps (%rdx,%r9,1),%xmm5,%xmm0
    2bb7:	vfmadd231ps (%rdx,%r10,1),%xmm6,%xmm12
    2bbd:	cmp    %r11,%r8
    2bc0:	je     2c81 <_Z2v0RK4Args+0x1a1>
    2bc6:	movslq (%rcx,%r11,1),%rax
    2bca:	movslq 0x4(%rcx,%r11,1),%rdi
    2bcf:	movslq 0x8(%rcx,%r11,1),%r9
    2bd4:	movslq 0xc(%rcx,%r11,1),%r10
    2bd9:	shl    $0x4,%rax
    2bdd:	shl    $0x4,%rdi
    2be1:	vbroadcastss (%rsi,%r11,1),%xmm7
    2be7:	vbroadcastss 0x4(%rsi,%r11,1),%xmm8
    2bee:	shl    $0x4,%r9
    2bf2:	vfmadd132ps (%rdx,%rax,1),%xmm0,%xmm7
    2bf8:	shl    $0x4,%r10
    2bfc:	movslq 0x10(%rcx,%r11,1),%rax
    2c01:	vfmadd231ps (%rdx,%rdi,1),%xmm8,%xmm12
    2c07:	movslq 0x14(%rcx,%r11,1),%rdi
    2c0c:	vbroadcastss 0x8(%rsi,%r11,1),%xmm9
    2c13:	vbroadcastss 0xc(%rsi,%r11,1),%xmm10
    2c1a:	shl    $0x4,%rax
    2c1e:	vfmadd132ps (%rdx,%r10,1),%xmm12,%xmm10
    2c24:	movslq 0x1c(%rcx,%r11,1),%r10
    2c29:	shl    $0x4,%rdi
    2c2d:	vfmadd132ps (%rdx,%r9,1),%xmm7,%xmm9
    2c33:	movslq 0x18(%rcx,%r11,1),%r9
    2c38:	vbroadcastss 0x10(%rsi,%r11,1),%xmm1
    2c3f:	vbroadcastss 0x14(%rsi,%r11,1),%xmm11
    2c46:	shl    $0x4,%r10
    2c4a:	vbroadcastss 0x18(%rsi,%r11,1),%xmm0
    2c51:	vbroadcastss 0x1c(%rsi,%r11,1),%xmm12
    2c58:	shl    $0x4,%r9
    2c5c:	add    $0x20,%r11
    2c60:	vfmadd132ps (%rdx,%rax,1),%xmm9,%xmm1
    2c66:	vfmadd132ps (%rdx,%rdi,1),%xmm10,%xmm11
    2c6c:	vfmadd132ps (%rdx,%r9,1),%xmm1,%xmm0
    2c72:	vfmadd132ps (%rdx,%r10,1),%xmm11,%xmm12
    2c78:	cmp    %r11,%r8
    2c7b:	jne    2bc6 <_Z2v0RK4Args+0xe6>
    2c81:	vaddps %xmm12,%xmm0,%xmm13
    2c86:	vshufps $0x55,%xmm13,%xmm13,%xmm15
    2c8c:	vaddss %xmm15,%xmm13,%xmm0
    2c91:	vunpckhps %xmm13,%xmm13,%xmm2
    2c96:	vshufps $0xff,%xmm13,%xmm13,%xmm4
    2c9c:	vaddss %xmm2,%xmm0,%xmm3
    2ca0:	vaddss %xmm4,%xmm3,%xmm0
    2ca4:	ret    
    2ca5:	nopl   (%rax)
    2ca8:	vxorps %xmm0,%xmm0,%xmm0
    2cac:	ret    
    2cad:	nopl   (%rax)

0000000000002cb0 <_Z3v0bRK4Args>:
    2cb0:	endbr64 
    2cb4:	mov    0x20(%rdi),%eax
    2cb7:	cmp    $0x1,%eax
    2cba:	jle    2e78 <_Z3v0bRK4Args+0x1c8>
    2cc0:	sub    $0x2,%eax
    2cc3:	mov    0x10(%rdi),%rdx
    2cc7:	mov    (%rdi),%rcx
    2cca:	vxorps %xmm12,%xmm12,%xmm12
    2ccf:	shr    %eax
    2cd1:	mov    0x8(%rdi),%rsi
    2cd5:	xor    %r11d,%r11d
    2cd8:	vmovaps %xmm12,%xmm0
    2cdc:	lea    0x1(%rax),%r8d
    2ce0:	shl    $0x3,%r8
    2ce4:	lea    -0x8(%r8),%rdi
    2ce8:	shr    $0x3,%rdi
    2cec:	add    $0x1,%rdi
    2cf0:	and    $0x3,%edi
    2cf3:	je     2d96 <_Z3v0bRK4Args+0xe6>
    2cf9:	cmp    $0x1,%rdi
    2cfd:	je     2d5f <_Z3v0bRK4Args+0xaf>
    2cff:	cmp    $0x2,%rdi
    2d03:	je     2d31 <_Z3v0bRK4Args+0x81>
    2d05:	movslq (%rcx),%r9
    2d08:	movslq 0x4(%rcx),%r10
    2d0c:	mov    $0x8,%r11d
    2d12:	vbroadcastss (%rsi),%xmm0
    2d17:	vbroadcastss 0x4(%rsi),%xmm2
    2d1d:	shl    $0x4,%r9
    2d21:	shl    $0x4,%r10
    2d25:	vfmadd132ps (%rdx,%r9,1),%xmm12,%xmm0
    2d2b:	vfmadd231ps (%rdx,%r10,1),%xmm2,%xmm12
    2d31:	movslq (%rcx,%r11,1),%rax
    2d35:	movslq 0x4(%rcx,%r11,1),%rdi
    2d3a:	vbroadcastss (%rsi,%r11,1),%xmm3
    2d40:	vbroadcastss 0x4(%rsi,%r11,1),%xmm4
    2d47:	add    $0x8,%r11
    2d4b:	shl    $0x4,%rax
    2d4f:	shl    $0x4,%rdi
    2d53:	vfmadd231ps (%rdx,%rax,1),%xmm3,%xmm0
    2d59:	vfmadd231ps (%rdx,%rdi,1),%xmm4,%xmm12
    2d5f:	movslq (%rcx,%r11,1),%r9
    2d63:	movslq 0x4(%rcx,%r11,1),%r10
    2d68:	vbroadcastss (%rsi,%r11,1),%xmm5
    2d6e:	vbroadcastss 0x4(%rsi,%r11,1),%xmm6
    2d75:	add    $0x8,%r11
    2d79:	shl    $0x4,%r9
    2d7d:	shl    $0x4,%r10
    2d81:	vfmadd231ps (%rdx,%r9,1),%xmm5,%xmm0
    2d87:	vfmadd231ps (%rdx,%r10,1),%xmm6,%xmm12
    2d8d:	cmp    %r11,%r8
    2d90:	je     2e51 <_Z3v0bRK4Args+0x1a1>
    2d96:	movslq (%rcx,%r11,1),%rax
    2d9a:	movslq 0x4(%rcx,%r11,1),%rdi
    2d9f:	movslq 0x8(%rcx,%r11,1),%r9
    2da4:	movslq 0xc(%rcx,%r11,1),%r10
    2da9:	shl    $0x4,%rax
    2dad:	shl    $0x4,%rdi
    2db1:	vbroadcastss (%rsi,%r11,1),%xmm7
    2db7:	vbroadcastss 0x4(%rsi,%r11,1),%xmm8
    2dbe:	shl    $0x4,%r9
    2dc2:	vfmadd132ps (%rdx,%rax,1),%xmm0,%xmm7
    2dc8:	shl    $0x4,%r10
    2dcc:	movslq 0x10(%rcx,%r11,1),%rax
    2dd1:	vfmadd231ps (%rdx,%rdi,1),%xmm8,%xmm12
    2dd7:	movslq 0x14(%rcx,%r11,1),%rdi
    2ddc:	vbroadcastss 0x8(%rsi,%r11,1),%xmm9
    2de3:	vbroadcastss 0xc(%rsi,%r11,1),%xmm10
    2dea:	shl    $0x4,%rax
    2dee:	vfmadd132ps (%rdx,%r10,1),%xmm12,%xmm10
    2df4:	movslq 0x1c(%rcx,%r11,1),%r10
    2df9:	shl    $0x4,%rdi
    2dfd:	vfmadd132ps (%rdx,%r9,1),%xmm7,%xmm9
    2e03:	movslq 0x18(%rcx,%r11,1),%r9
    2e08:	vbroadcastss 0x10(%rsi,%r11,1),%xmm1
    2e0f:	vbroadcastss 0x14(%rsi,%r11,1),%xmm11
    2e16:	shl    $0x4,%r10
    2e1a:	vbroadcastss 0x18(%rsi,%r11,1),%xmm0
    2e21:	vbroadcastss 0x1c(%rsi,%r11,1),%xmm12
    2e28:	shl    $0x4,%r9
    2e2c:	add    $0x20,%r11
    2e30:	vfmadd132ps (%rdx,%rax,1),%xmm9,%xmm1
    2e36:	vfmadd132ps (%rdx,%rdi,1),%xmm10,%xmm11
    2e3c:	vfmadd132ps (%rdx,%r9,1),%xmm1,%xmm0
    2e42:	vfmadd132ps (%rdx,%r10,1),%xmm11,%xmm12
    2e48:	cmp    %r11,%r8
    2e4b:	jne    2d96 <_Z3v0bRK4Args+0xe6>
    2e51:	vaddps %xmm12,%xmm0,%xmm13
    2e56:	vshufps $0x55,%xmm13,%xmm13,%xmm15
    2e5c:	vaddss %xmm15,%xmm13,%xmm0
    2e61:	vunpckhps %xmm13,%xmm13,%xmm2
    2e66:	vshufps $0xff,%xmm13,%xmm13,%xmm4
    2e6c:	vaddss %xmm2,%xmm0,%xmm3
    2e70:	vaddss %xmm4,%xmm3,%xmm0
    2e74:	ret    
    2e75:	nopl   (%rax)
    2e78:	vxorps %xmm0,%xmm0,%xmm0
    2e7c:	ret    
    2e7d:	nopl   (%rax)

0000000000002e80 <_Z2v1RK4Args>:
    2e80:	endbr64 
    2e84:	mov    0x20(%rdi),%eax
    2e87:	cmp    $0x1,%eax
    2e8a:	jle    2fc8 <_Z2v1RK4Args+0x148>
    2e90:	sub    $0x2,%eax
    2e93:	mov    0x10(%rdi),%rdx
    2e97:	mov    0x8(%rdi),%rcx
    2e9b:	vxorps %xmm12,%xmm12,%xmm12
    2ea0:	shr    %eax
    2ea2:	xor    %r8d,%r8d
    2ea5:	vmovaps %xmm12,%xmm0
    2ea9:	lea    0x1(%rax),%edi
    2eac:	shl    $0x3,%rdi
    2eb0:	lea    -0x8(%rdi),%rsi
    2eb4:	shr    $0x3,%rsi
    2eb8:	add    $0x1,%rsi
    2ebc:	and    $0x3,%esi
    2ebf:	je     2f2a <_Z2v1RK4Args+0xaa>
    2ec1:	cmp    $0x1,%rsi
    2ec5:	je     2f07 <_Z2v1RK4Args+0x87>
    2ec7:	cmp    $0x2,%rsi
    2ecb:	je     2ee9 <_Z2v1RK4Args+0x69>
    2ecd:	vbroadcastss (%rcx),%xmm0
    2ed2:	vbroadcastss 0x4(%rcx),%xmm2
    2ed8:	mov    $0x8,%r8d
    2ede:	vfmadd132ps (%rdx),%xmm12,%xmm0
    2ee3:	vfmadd231ps 0x10(%rdx),%xmm2,%xmm12
    2ee9:	vbroadcastss (%rcx,%r8,1),%xmm3
    2eef:	vbroadcastss 0x4(%rcx,%r8,1),%xmm4
    2ef6:	vfmadd231ps (%rdx,%r8,4),%xmm3,%xmm0
    2efc:	vfmadd231ps 0x10(%rdx,%r8,4),%xmm4,%xmm12
    2f03:	add    $0x8,%r8
    2f07:	vbroadcastss (%rcx,%r8,1),%xmm5
    2f0d:	vbroadcastss 0x4(%rcx,%r8,1),%xmm6
    2f14:	vfmadd231ps (%rdx,%r8,4),%xmm5,%xmm0
    2f1a:	vfmadd231ps 0x10(%rdx,%r8,4),%xmm6,%xmm12
    2f21:	add    $0x8,%r8
    2f25:	cmp    %r8,%rdi
    2f28:	je     2fa1 <_Z2v1RK4Args+0x121>
    2f2a:	vbroadcastss (%rcx,%r8,1),%xmm7
    2f30:	vbroadcastss 0x4(%rcx,%r8,1),%xmm8
    2f37:	vfmadd132ps (%rdx,%r8,4),%xmm0,%xmm7
    2f3d:	vfmadd231ps 0x10(%rdx,%r8,4),%xmm8,%xmm12
    2f44:	vbroadcastss 0x8(%rcx,%r8,1),%xmm9
    2f4b:	vbroadcastss 0xc(%rcx,%r8,1),%xmm10
    2f52:	vfmadd132ps 0x20(%rdx,%r8,4),%xmm7,%xmm9
    2f59:	vfmadd132ps 0x30(%rdx,%r8,4),%xmm12,%xmm10
    2f60:	vbroadcastss 0x10(%rcx,%r8,1),%xmm1
    2f67:	vbroadcastss 0x14(%rcx,%r8,1),%xmm11
    2f6e:	vfmadd132ps 0x40(%rdx,%r8,4),%xmm9,%xmm1
    2f75:	vfmadd132ps 0x50(%rdx,%r8,4),%xmm10,%xmm11
    2f7c:	vbroadcastss 0x18(%rcx,%r8,1),%xmm0
    2f83:	vbroadcastss 0x1c(%rcx,%r8,1),%xmm12
    2f8a:	vfmadd132ps 0x60(%rdx,%r8,4),%xmm1,%xmm0
    2f91:	vfmadd132ps 0x70(%rdx,%r8,4),%xmm11,%xmm12
    2f98:	add    $0x20,%r8
    2f9c:	cmp    %r8,%rdi
    2f9f:	jne    2f2a <_Z2v1RK4Args+0xaa>
    2fa1:	vaddps %xmm12,%xmm0,%xmm13
    2fa6:	vshufps $0x55,%xmm13,%xmm13,%xmm15
    2fac:	vaddss %xmm15,%xmm13,%xmm0
    2fb1:	vunpckhps %xmm13,%xmm13,%xmm2
    2fb6:	vshufps $0xff,%xmm13,%xmm13,%xmm4
    2fbc:	vaddss %xmm2,%xmm0,%xmm3
    2fc0:	vaddss %xmm4,%xmm3,%xmm0
    2fc4:	ret    
    2fc5:	nopl   (%rax)
    2fc8:	vxorps %xmm0,%xmm0,%xmm0
    2fcc:	ret    
    2fcd:	nopl   (%rax)

0000000000002fd0 <_Z2v2RK4Args>:
    2fd0:	endbr64 
    2fd4:	mov    0x8(%rdi),%rax
    2fd8:	mov    0x20(%rdi),%esi
    2fdb:	vbroadcastss (%rax),%xmm3
    2fe0:	vbroadcastss 0x4(%rax),%xmm2
    2fe6:	cmp    $0x1,%esi
    2fe9:	jle    3130 <_Z2v2RK4Args+0x160>
    2fef:	mov    (%rdi),%rcx
    2ff2:	sub    $0x2,%esi
    2ff5:	vxorps %xmm0,%xmm0,%xmm0
    2ff9:	mov    0x10(%rdi),%rdx
    2ffd:	shr    %esi
    2fff:	vmovaps %xmm0,%xmm1
    3003:	lea    0x8(%rcx),%r9
    3007:	lea    (%r9,%rsi,8),%rdi
    300b:	and    $0x3,%esi
    300e:	je     30eb <_Z2v2RK4Args+0x11b>
    3014:	movslq (%rcx),%r8
    3017:	movslq 0x4(%rcx),%rcx
    301b:	vmovaps %xmm3,%xmm1
    301f:	shl    $0x4,%rcx
    3023:	shl    $0x4,%r8
    3027:	vfmadd132ps (%rdx,%r8,1),%xmm0,%xmm1
    302d:	vfmadd231ps (%rdx,%rcx,1),%xmm2,%xmm0
    3033:	mov    %r9,%rcx
    3036:	add    $0x8,%r9
    303a:	cmp    $0x1,%rsi
    303e:	je     30eb <_Z2v2RK4Args+0x11b>
    3044:	cmp    $0x2,%rsi
    3048:	je     306c <_Z2v2RK4Args+0x9c>
    304a:	movslq (%rcx),%r10
    304d:	movslq 0x4(%rcx),%r11
    3051:	mov    %r9,%rcx
    3054:	add    $0x8,%r9
    3058:	shl    $0x4,%r10
    305c:	shl    $0x4,%r11
    3060:	vfmadd231ps (%rdx,%r10,1),%xmm3,%xmm1
    3066:	vfmadd231ps (%rdx,%r11,1),%xmm2,%xmm0
    306c:	movslq (%rcx),%rax
    306f:	movslq 0x4(%rcx),%rsi
    3073:	mov    %r9,%rcx
    3076:	add    $0x8,%r9
    307a:	shl    $0x4,%rax
    307e:	shl    $0x4,%rsi
    3082:	vfmadd231ps (%rdx,%rax,1),%xmm3,%xmm1
    3088:	vfmadd231ps (%rdx,%rsi,1),%xmm2,%xmm0
    308e:	jmp    30eb <_Z2v2RK4Args+0x11b>
    3090:	movslq (%r9),%r11
    3093:	movslq 0x4(%r9),%rax
    3097:	movslq 0x8(%r9),%rsi
    309b:	movslq 0xc(%r9),%rcx
    309f:	movslq 0x10(%r9),%r8
    30a3:	movslq 0x14(%r9),%r10
    30a7:	shl    $0x4,%r11
    30ab:	shl    $0x4,%rax
    30af:	shl    $0x4,%rcx
    30b3:	vfmadd231ps (%rdx,%r11,1),%xmm3,%xmm1
    30b9:	shl    $0x4,%rsi
    30bd:	vfmadd231ps (%rdx,%rax,1),%xmm2,%xmm0
    30c3:	vfmadd231ps (%rdx,%rsi,1),%xmm3,%xmm1
    30c9:	shl    $0x4,%r8
    30cd:	vfmadd231ps (%rdx,%rcx,1),%xmm2,%xmm0
    30d3:	shl    $0x4,%r10
    30d7:	lea    0x18(%r9),%rcx
    30db:	add    $0x20,%r9
    30df:	vfmadd231ps (%rdx,%r8,1),%xmm3,%xmm1
    30e5:	vfmadd231ps (%rdx,%r10,1),%xmm2,%xmm0
    30eb:	movslq (%rcx),%r8
    30ee:	movslq 0x4(%rcx),%r10
    30f2:	shl    $0x4,%r8
    30f6:	shl    $0x4,%r10
    30fa:	vfmadd231ps (%rdx,%r8,1),%xmm3,%xmm1
    3100:	vfmadd231ps (%rdx,%r10,1),%xmm2,%xmm0
    3106:	cmp    %r9,%rdi
    3109:	jne    3090 <_Z2v2RK4Args+0xc0>
    310b:	vaddps %xmm1,%xmm0,%xmm4
    310f:	vshufps $0x55,%xmm4,%xmm4,%xmm5
    3114:	vaddss %xmm4,%xmm5,%xmm7
    3118:	vunpckhps %xmm4,%xmm4,%xmm8
    311c:	vshufps $0xff,%xmm4,%xmm4,%xmm10
    3121:	vaddss %xmm8,%xmm7,%xmm9
    3126:	vaddss %xmm10,%xmm9,%xmm0
    312b:	ret    
    312c:	nopl   0x0(%rax)
    3130:	vxorps %xmm0,%xmm0,%xmm0
    3134:	ret    
    3135:	data16 cs nopw 0x0(%rax,%rax,1)

0000000000003140 <_Z2v3RK4Args>:
    3140:	endbr64 
    3144:	mov    0x10(%rdi),%rax
    3148:	mov    0x20(%rdi),%edx
    314b:	vmovups (%rax),%xmm4
    314f:	vmovups 0x10(%rax),%xmm3
    3154:	cmp    $0x1,%edx
    3157:	jle    32f0 <_Z2v3RK4Args+0x1b0>
    315d:	sub    $0x2,%edx
    3160:	vxorps %xmm14,%xmm14,%xmm14
    3165:	mov    (%rdi),%rsi
    3168:	mov    0x8(%rdi),%rcx
    316c:	shr    %edx
    316e:	xor    %r10d,%r10d
    3171:	vmovaps %xmm14,%xmm1
    3175:	xor    %edi,%edi
    3177:	lea    0x1(%rdx),%r8d
    317b:	shl    $0x3,%r8
    317f:	lea    -0x8(%r8),%r9
    3183:	shr    $0x3,%r9
    3187:	add    $0x1,%r9
    318b:	and    $0x3,%r9d
    318f:	je     3217 <_Z2v3RK4Args+0xd7>
    3195:	cmp    $0x1,%r9
    3199:	je     31e8 <_Z2v3RK4Args+0xa8>
    319b:	cmp    $0x2,%r9
    319f:	je     31c1 <_Z2v3RK4Args+0x81>
    31a1:	vbroadcastss (%rcx),%xmm1
    31a6:	mov    0x4(%rsi),%edi
    31a9:	mov    $0x8,%r10d
    31af:	vbroadcastss 0x4(%rcx),%xmm2
    31b5:	add    (%rsi),%edi
    31b7:	vfmadd132ps %xmm4,%xmm14,%xmm1
    31bc:	vfmadd231ps %xmm3,%xmm2,%xmm14
    31c1:	vbroadcastss (%rcx,%r10,1),%xmm5
    31c7:	vbroadcastss 0x4(%rcx,%r10,1),%xmm6
    31ce:	mov    0x4(%rsi,%r10,1),%r11d
    31d3:	add    (%rsi,%r10,1),%r11d
    31d7:	add    $0x8,%r10
    31db:	vfmadd231ps %xmm4,%xmm5,%xmm1
    31e0:	vfmadd231ps %xmm3,%xmm6,%xmm14
    31e5:	add    %r11d,%edi
    31e8:	vbroadcastss (%rcx,%r10,1),%xmm7
    31ee:	vbroadcastss 0x4(%rcx,%r10,1),%xmm8
    31f5:	mov    0x4(%rsi,%r10,1),%eax
    31fa:	add    (%rsi,%r10,1),%eax
    31fe:	add    $0x8,%r10
    3202:	vfmadd231ps %xmm4,%xmm7,%xmm1
    3207:	vfmadd231ps %xmm3,%xmm8,%xmm14
    320c:	add    %eax,%edi
    320e:	cmp    %r10,%r8
    3211:	je     32b3 <_Z2v3RK4Args+0x173>
    3217:	vbroadcastss (%rcx,%r10,1),%xmm9
    321d:	vbroadcastss 0x4(%rcx,%r10,1),%xmm10
    3224:	vbroadcastss 0x8(%rcx,%r10,1),%xmm11
    322b:	vbroadcastss 0xc(%rcx,%r10,1),%xmm12
    3232:	vfmadd132ps %xmm4,%xmm1,%xmm9
    3237:	vfmadd231ps %xmm3,%xmm10,%xmm14
    323c:	vbroadcastss 0x10(%rcx,%r10,1),%xmm0
    3243:	vbroadcastss 0x14(%rcx,%r10,1),%xmm13
    324a:	mov    0x4(%rsi,%r10,1),%edx
    324f:	add    (%rsi,%r10,1),%edx
    3253:	mov    0xc(%rsi,%r10,1),%r9d
    3258:	add    %edi,%edx
    325a:	add    0x8(%rsi,%r10,1),%r9d
    325f:	mov    0x14(%rsi,%r10,1),%edi
    3264:	vbroadcastss 0x18(%rcx,%r10,1),%xmm1
    326b:	add    %r9d,%edx
    326e:	add    0x10(%rsi,%r10,1),%edi
    3273:	vfmadd132ps %xmm3,%xmm14,%xmm12
    3278:	vfmadd132ps %xmm4,%xmm9,%xmm11
    327d:	vbroadcastss 0x1c(%rcx,%r10,1),%xmm14
    3284:	add    %edi,%edx
    3286:	mov    0x1c(%rsi,%r10,1),%edi
    328b:	add    0x18(%rsi,%r10,1),%edi
    3290:	add    $0x20,%r10
    3294:	add    %edx,%edi
    3296:	vfmadd132ps %xmm4,%xmm11,%xmm0
    329b:	vfmadd132ps %xmm3,%xmm12,%xmm13
    32a0:	vfmadd132ps %xmm4,%xmm0,%xmm1
    32a5:	vfmadd132ps %xmm3,%xmm13,%xmm14
    32aa:	cmp    %r10,%r8
    32ad:	jne    3217 <_Z2v3RK4Args+0xd7>
    32b3:	vaddps %xmm1,%xmm14,%xmm15
    32b7:	vxorps %xmm4,%xmm4,%xmm4
    32bb:	vcvtsi2ss %edi,%xmm4,%xmm0
    32bf:	vshufps $0x55,%xmm15,%xmm15,%xmm1
    32c5:	vaddss %xmm15,%xmm1,%xmm2
    32ca:	vunpckhps %xmm15,%xmm15,%xmm5
    32cf:	vshufps $0xff,%xmm15,%xmm15,%xmm7
    32d5:	vaddss %xmm5,%xmm2,%xmm6
    32d9:	vaddss %xmm7,%xmm6,%xmm8
    32dd:	vfmadd132ss 0x1e1a(%rip),%xmm8,%xmm0        # 5100 <_IO_stdin_used+0x100>
    32e6:	ret    
    32e7:	nopw   0x0(%rax,%rax,1)
    32f0:	vxorps %xmm0,%xmm0,%xmm0
    32f4:	ret    
    32f5:	data16 cs nopw 0x0(%rax,%rax,1)

0000000000003300 <_Z2v6RK4Args>:
    3300:	endbr64 
    3304:	push   %rbx
    3305:	mov    0x20(%rdi),%r10d
    3309:	mov    (%rdi),%r8
    330c:	mov    0x8(%rdi),%r9
    3310:	mov    0x10(%rdi),%rdx
    3314:	cmp    $0x3,%r10d
    3318:	jle    35a0 <_Z2v6RK4Args+0x2a0>
    331e:	lea    -0x4(%r10),%r11d
    3322:	lea    0x10(%r8),%rsi
    3326:	vxorps %xmm1,%xmm1,%xmm1
    332a:	mov    %r8,%rax
    332d:	shr    $0x2,%r11d
    3331:	vmovaps %xmm1,%xmm2
    3335:	vmovaps %xmm1,%xmm6
    3339:	mov    %r9,%rcx
    333c:	mov    %r11d,%edi
    333f:	vmovaps %xmm1,%xmm0
    3343:	shl    $0x4,%rdi
    3347:	add    %rsi,%rdi
    334a:	mov    %rdi,%rbx
    334d:	sub    %r8,%rbx
    3350:	and    $0x10,%ebx
    3353:	je     33c0 <_Z2v6RK4Args+0xc0>
    3355:	movslq (%r8),%rcx
    3358:	vbroadcastss (%r9),%xmm0
    335d:	movslq 0x4(%r8),%rax
    3361:	movslq 0x8(%r8),%rbx
    3365:	shl    $0x4,%rcx
    3369:	vbroadcastss 0x4(%r9),%xmm5
    336f:	vbroadcastss 0x8(%r9),%xmm2
    3375:	vfmadd132ps (%rdx,%rcx,1),%xmm1,%xmm0
    337b:	movslq 0xc(%r8),%rcx
    337f:	shl    $0x4,%rax
    3383:	shl    $0x4,%rbx
    3387:	vbroadcastss 0xc(%r9),%xmm3
    338d:	vmovaps %xmm5,%xmm6
    3391:	vfmadd132ps (%rdx,%rbx,1),%xmm1,%xmm2
    3397:	shl    $0x4,%rcx
    339b:	vfmadd132ps (%rdx,%rax,1),%xmm1,%xmm6
    33a1:	mov    %rsi,%rax
    33a4:	vfmadd231ps (%rdx,%rcx,1),%xmm3,%xmm1
    33aa:	lea    0x10(%r9),%rcx
    33ae:	cmp    %rsi,%rdi
    33b1:	je     3474 <_Z2v6RK4Args+0x174>
    33b7:	nopw   0x0(%rax,%rax,1)
    33c0:	movslq (%rax),%rsi
    33c3:	movslq 0x4(%rax),%rbx
    33c7:	add    $0x20,%rax
    33cb:	add    $0x20,%rcx
    33cf:	vbroadcastss -0x20(%rcx),%xmm4
    33d5:	vbroadcastss -0x1c(%rcx),%xmm7
    33db:	shl    $0x4,%rsi
    33df:	shl    $0x4,%rbx
    33e3:	vbroadcastss -0x18(%rcx),%xmm8
    33e9:	vbroadcastss -0x14(%rcx),%xmm9
    33ef:	vfmadd132ps (%rdx,%rsi,1),%xmm0,%xmm4
    33f5:	vfmadd132ps (%rdx,%rbx,1),%xmm6,%xmm7
    33fb:	movslq -0x18(%rax),%rsi
    33ff:	movslq -0x14(%rax),%rbx
    3403:	vbroadcastss -0xc(%rcx),%xmm5
    3409:	vbroadcastss -0x10(%rcx),%xmm0
    340f:	shl    $0x4,%rsi
    3413:	shl    $0x4,%rbx
    3417:	vbroadcastss -0x8(%rcx),%xmm10
    341d:	vbroadcastss -0x4(%rcx),%xmm11
    3423:	vfmadd231ps (%rdx,%rsi,1),%xmm8,%xmm2
    3429:	vfmadd231ps (%rdx,%rbx,1),%xmm9,%xmm1
    342f:	vmovaps %xmm5,%xmm6
    3433:	movslq -0x10(%rax),%rsi
    3437:	movslq -0xc(%rax),%rbx
    343b:	shl    $0x4,%rsi
    343f:	shl    $0x4,%rbx
    3443:	vfmadd132ps (%rdx,%rsi,1),%xmm4,%xmm0
    3449:	vfmadd132ps (%rdx,%rbx,1),%xmm7,%xmm6
    344f:	movslq -0x8(%rax),%rsi
    3453:	movslq -0x4(%rax),%rbx
    3457:	shl    $0x4,%rsi
    345b:	shl    $0x4,%rbx
    345f:	vfmadd231ps (%rdx,%rsi,1),%xmm10,%xmm2
    3465:	vfmadd231ps (%rdx,%rbx,1),%xmm11,%xmm1
    346b:	cmp    %rax,%rdi
    346e:	jne    33c0 <_Z2v6RK4Args+0xc0>
    3474:	vaddps %xmm2,%xmm1,%xmm1
    3478:	lea    0x4(,%r11,4),%eax
    3480:	cmp    %eax,%r10d
    3483:	jle    3576 <_Z2v6RK4Args+0x276>
    3489:	cltq   
    348b:	movslq (%r8,%rax,4),%rdi
    348f:	mov    %rax,%r11
    3492:	lea    0x1(%rax),%rsi
    3496:	vbroadcastss (%r9,%rax,4),%xmm12
    349c:	not    %r11
    349f:	add    %r10d,%r11d
    34a2:	shl    $0x4,%rdi
    34a6:	vfmadd231ps (%rdx,%rdi,1),%xmm12,%xmm0
    34ac:	and    $0x3,%r11d
    34b0:	cmp    %esi,%r10d
    34b3:	jle    3576 <_Z2v6RK4Args+0x276>
    34b9:	test   %r11d,%r11d
    34bc:	je     3517 <_Z2v6RK4Args+0x217>
    34be:	cmp    $0x1,%r11d
    34c2:	je     34fa <_Z2v6RK4Args+0x1fa>
    34c4:	cmp    $0x2,%r11d
    34c8:	je     34e2 <_Z2v6RK4Args+0x1e2>
    34ca:	movslq (%r8,%rsi,4),%rcx
    34ce:	vbroadcastss (%r9,%rsi,4),%xmm13
    34d4:	lea    0x2(%rax),%rsi
    34d8:	shl    $0x4,%rcx
    34dc:	vfmadd231ps (%rdx,%rcx,1),%xmm13,%xmm0
    34e2:	movslq (%r8,%rsi,4),%rax
    34e6:	vbroadcastss (%r9,%rsi,4),%xmm14
    34ec:	add    $0x1,%rsi
    34f0:	shl    $0x4,%rax
    34f4:	vfmadd231ps (%rdx,%rax,1),%xmm14,%xmm0
    34fa:	movslq (%r8,%rsi,4),%rbx
    34fe:	vbroadcastss (%r9,%rsi,4),%xmm15
    3504:	add    $0x1,%rsi
    3508:	shl    $0x4,%rbx
    350c:	vfmadd231ps (%rdx,%rbx,1),%xmm15,%xmm0
    3512:	cmp    %esi,%r10d
    3515:	jle    3576 <_Z2v6RK4Args+0x276>
    3517:	movslq (%r8,%rsi,4),%r11
    351b:	movslq 0x4(%r8,%rsi,4),%rdi
    3520:	movslq 0x8(%r8,%rsi,4),%rcx
    3525:	vbroadcastss (%r9,%rsi,4),%xmm2
    352b:	shl    $0x4,%r11
    352f:	movslq 0xc(%r8,%rsi,4),%rax
    3534:	shl    $0x4,%rdi
    3538:	vbroadcastss 0x8(%r9,%rsi,4),%xmm3
    353f:	vfmadd132ps (%rdx,%r11,1),%xmm0,%xmm2
    3545:	shl    $0x4,%rcx
    3549:	vbroadcastss 0x4(%r9,%rsi,4),%xmm0
    3550:	vfmadd132ps (%rdx,%rdi,1),%xmm2,%xmm0
    3556:	shl    $0x4,%rax
    355a:	vfmadd132ps (%rdx,%rcx,1),%xmm0,%xmm3
    3560:	vbroadcastss 0xc(%r9,%rsi,4),%xmm0
    3567:	add    $0x4,%rsi
    356b:	vfmadd132ps (%rdx,%rax,1),%xmm3,%xmm0
    3571:	cmp    %esi,%r10d
    3574:	jg     3517 <_Z2v6RK4Args+0x217>
    3576:	vaddps %xmm6,%xmm0,%xmm6
    357a:	pop    %rbx
    357b:	vaddps %xmm1,%xmm6,%xmm4
    357f:	vshufps $0x55,%xmm4,%xmm4,%xmm7
    3584:	vaddss %xmm4,%xmm7,%xmm9
    3588:	vunpckhps %xmm4,%xmm4,%xmm5
    358c:	vshufps $0xff,%xmm4,%xmm4,%xmm11
    3591:	vaddss %xmm5,%xmm9,%xmm10
    3595:	vaddss %xmm11,%xmm10,%xmm0
    359a:	ret    
    359b:	nopl   0x0(%rax,%rax,1)
    35a0:	vxorps %xmm1,%xmm1,%xmm1
    35a4:	xor    %eax,%eax
    35a6:	vmovaps %xmm1,%xmm6
    35aa:	vmovaps %xmm1,%xmm0
    35ae:	jmp    3480 <_Z2v6RK4Args+0x180>
    35b3:	data16 cs nopw 0x0(%rax,%rax,1)
    35be:	xchg   %ax,%ax

00000000000035c0 <_Z2v7RK4Args>:
    35c0:	endbr64 
    35c4:	mov    0x20(%rdi),%r10d
    35c8:	mov    (%rdi),%r8
    35cb:	mov    0x8(%rdi),%r9
    35cf:	mov    0x10(%rdi),%rcx
    35d3:	cmp    $0x7,%r10d
    35d7:	jle    3810 <_Z2v7RK4Args+0x250>
    35dd:	lea    -0x8(%r10),%r11d
    35e1:	vxorps %xmm4,%xmm4,%xmm4
    35e5:	mov    %r8,%rax
    35e8:	mov    %r9,%rdx
    35eb:	vmovaps %xmm4,%xmm7
    35ef:	shr    $0x3,%r11d
    35f3:	vmovaps %xmm4,%xmm3
    35f7:	vmovaps %xmm4,%xmm6
    35fb:	vmovaps %xmm4,%xmm5
    35ff:	vmovaps %xmm4,%xmm2
    3603:	vmovaps %xmm4,%xmm0
    3607:	mov    %r11d,%esi
    360a:	shl    $0x5,%rsi
    360e:	vmovaps %xmm4,%xmm1
    3612:	lea    0x20(%r8,%rsi,1),%rdi
    3617:	nopw   0x0(%rax,%rax,1)
    3620:	movslq (%rax),%rsi
    3623:	vbroadcastss (%rdx),%xmm8
    3628:	add    $0x20,%rax
    362c:	add    $0x20,%rdx
    3630:	vbroadcastss -0x1c(%rdx),%xmm9
    3636:	vbroadcastss -0x18(%rdx),%xmm10
    363c:	shl    $0x4,%rsi
    3640:	vbroadcastss -0x14(%rdx),%xmm11
    3646:	vbroadcastss -0x10(%rdx),%xmm12
    364c:	vfmadd231ps (%rcx,%rsi,1),%xmm8,%xmm1
    3652:	movslq -0x1c(%rax),%rsi
    3656:	vbroadcastss -0xc(%rdx),%xmm13
    365c:	vbroadcastss -0x8(%rdx),%xmm14
    3662:	shl    $0x4,%rsi
    3666:	vbroadcastss -0x4(%rdx),%xmm15
    366c:	vfmadd231ps (%rcx,%rsi,1),%xmm9,%xmm0
    3672:	movslq -0x18(%rax),%rsi
    3676:	shl    $0x4,%rsi
    367a:	vfmadd231ps (%rcx,%rsi,1),%xmm10,%xmm2
    3680:	movslq -0x14(%rax),%rsi
    3684:	shl    $0x4,%rsi
    3688:	vfmadd231ps (%rcx,%rsi,1),%xmm11,%xmm5
    368e:	movslq -0x10(%rax),%rsi
    3692:	shl    $0x4,%rsi
    3696:	vfmadd231ps (%rcx,%rsi,1),%xmm12,%xmm6
    369c:	movslq -0xc(%rax),%rsi
    36a0:	shl    $0x4,%rsi
    36a4:	vfmadd231ps (%rcx,%rsi,1),%xmm13,%xmm3
    36aa:	movslq -0x8(%rax),%rsi
    36ae:	shl    $0x4,%rsi
    36b2:	vfmadd231ps (%rcx,%rsi,1),%xmm14,%xmm7
    36b8:	movslq -0x4(%rax),%rsi
    36bc:	shl    $0x4,%rsi
    36c0:	vfmadd231ps (%rcx,%rsi,1),%xmm15,%xmm4
    36c6:	cmp    %rax,%rdi
    36c9:	jne    3620 <_Z2v7RK4Args+0x60>
    36cf:	vaddps %xmm7,%xmm4,%xmm4
    36d3:	vaddps %xmm6,%xmm3,%xmm7
    36d7:	vaddps %xmm5,%xmm2,%xmm2
    36db:	lea    0x8(,%r11,8),%edx
    36e3:	vaddps %xmm7,%xmm4,%xmm9
    36e7:	cmp    %edx,%r10d
    36ea:	jle    37de <_Z2v7RK4Args+0x21e>
    36f0:	movslq %edx,%rdx
    36f3:	movslq (%r8,%rdx,4),%rax
    36f7:	mov    %rdx,%r11
    36fa:	lea    0x1(%rdx),%rsi
    36fe:	vbroadcastss (%r9,%rdx,4),%xmm3
    3704:	not    %r11
    3707:	add    %r10d,%r11d
    370a:	shl    $0x4,%rax
    370e:	vfmadd231ps (%rcx,%rax,1),%xmm3,%xmm1
    3714:	and    $0x3,%r11d
    3718:	cmp    %esi,%r10d
    371b:	jle    37de <_Z2v7RK4Args+0x21e>
    3721:	test   %r11d,%r11d
    3724:	je     377f <_Z2v7RK4Args+0x1bf>
    3726:	cmp    $0x1,%r11d
    372a:	je     3762 <_Z2v7RK4Args+0x1a2>
    372c:	cmp    $0x2,%r11d
    3730:	je     374a <_Z2v7RK4Args+0x18a>
    3732:	movslq (%r8,%rsi,4),%rdi
    3736:	vbroadcastss (%r9,%rsi,4),%xmm6
    373c:	lea    0x2(%rdx),%rsi
    3740:	shl    $0x4,%rdi
    3744:	vfmadd231ps (%rcx,%rdi,1),%xmm6,%xmm1
    374a:	movslq (%r8,%rsi,4),%rdx
    374e:	vbroadcastss (%r9,%rsi,4),%xmm5
    3754:	add    $0x1,%rsi
    3758:	shl    $0x4,%rdx
    375c:	vfmadd231ps (%rcx,%rdx,1),%xmm5,%xmm1
    3762:	movslq (%r8,%rsi,4),%r11
    3766:	vbroadcastss (%r9,%rsi,4),%xmm8
    376c:	add    $0x1,%rsi
    3770:	shl    $0x4,%r11
    3774:	vfmadd231ps (%rcx,%r11,1),%xmm8,%xmm1
    377a:	cmp    %esi,%r10d
    377d:	jle    37de <_Z2v7RK4Args+0x21e>
    377f:	movslq (%r8,%rsi,4),%rax
    3783:	movslq 0x4(%r8,%rsi,4),%rdi
    3788:	movslq 0x8(%r8,%rsi,4),%rdx
    378d:	vbroadcastss (%r9,%rsi,4),%xmm10
    3793:	shl    $0x4,%rax
    3797:	movslq 0xc(%r8,%rsi,4),%r11
    379c:	shl    $0x4,%rdi
    37a0:	vbroadcastss 0x8(%r9,%rsi,4),%xmm11
    37a7:	vfmadd132ps (%rcx,%rax,1),%xmm1,%xmm10
    37ad:	shl    $0x4,%rdx
    37b1:	vbroadcastss 0x4(%r9,%rsi,4),%xmm1
    37b8:	vfmadd132ps (%rcx,%rdi,1),%xmm10,%xmm1
    37be:	shl    $0x4,%r11
    37c2:	vfmadd132ps (%rcx,%rdx,1),%xmm1,%xmm11
    37c8:	vbroadcastss 0xc(%r9,%rsi,4),%xmm1
    37cf:	add    $0x4,%rsi
    37d3:	vfmadd132ps (%rcx,%r11,1),%xmm11,%xmm1
    37d9:	cmp    %esi,%r10d
    37dc:	jg     377f <_Z2v7RK4Args+0x1bf>
    37de:	vaddps %xmm0,%xmm1,%xmm12
    37e2:	vaddps %xmm2,%xmm12,%xmm13
    37e6:	vaddps %xmm9,%xmm13,%xmm14
    37eb:	vshufps $0x55,%xmm14,%xmm14,%xmm15
    37f1:	vaddss %xmm14,%xmm15,%xmm7
    37f6:	vunpckhps %xmm14,%xmm14,%xmm3
    37fb:	vshufps $0xff,%xmm14,%xmm14,%xmm5
    3801:	vaddss %xmm3,%xmm7,%xmm6
    3805:	vaddss %xmm5,%xmm6,%xmm0
    3809:	ret    
    380a:	nopw   0x0(%rax,%rax,1)
    3810:	vxorps %xmm2,%xmm2,%xmm2
    3814:	xor    %edx,%edx
    3816:	vmovaps %xmm2,%xmm9
    381a:	vmovaps %xmm2,%xmm0
    381e:	vmovaps %xmm2,%xmm1
    3822:	jmp    36e7 <_Z2v7RK4Args+0x127>
    3827:	nopw   0x0(%rax,%rax,1)

0000000000003830 <_Z2v4RK4Args>:
    3830:	endbr64 
    3834:	mov    0x20(%rdi),%r8d
    3838:	cmp    $0x3,%r8d
    383c:	jle    3940 <_Z2v4RK4Args+0x110>
    3842:	mov    (%rdi),%rcx
    3845:	lea    -0x4(%r8),%r9d
    3849:	vxorps %xmm2,%xmm2,%xmm2
    384d:	mov    0x10(%rdi),%rsi
    3851:	shr    $0x2,%r9d
    3855:	mov    0x8(%rdi),%rdx
    3859:	mov    0x18(%rdi),%rax
    385d:	vmovaps %xmm2,%xmm3
    3861:	lea    0x10(%rcx),%r10
    3865:	vmovaps %xmm2,%xmm4
    3869:	shl    $0x4,%r9
    386d:	vmovaps %xmm2,%xmm5
    3871:	add    %r10,%r9
    3874:	vmovaps %xmm2,%xmm0
    3878:	vmovaps %xmm2,%xmm1
    387c:	jmp    3884 <_Z2v4RK4Args+0x54>
    387e:	xchg   %ax,%ax
    3880:	add    $0x10,%r10
    3884:	movslq (%rcx),%rdi
    3887:	movslq 0x4(%rcx),%r11
    388b:	add    $0x10,%rdx
    388f:	add    $0x40,%rax
    3893:	movslq 0x8(%rcx),%r8
    3897:	movslq 0xc(%rcx),%rcx
    389b:	shl    $0x4,%rdi
    389f:	shl    $0x4,%r11
    38a3:	vbroadcastss -0x10(%rdx),%xmm6
    38a9:	vbroadcastss -0xc(%rdx),%xmm7
    38af:	shl    $0x4,%rcx
    38b3:	shl    $0x4,%r8
    38b7:	vfmadd231ps (%rsi,%rdi,1),%xmm6,%xmm1
    38bd:	vbroadcastss -0x8(%rdx),%xmm8
    38c3:	vfmadd231ps (%rsi,%r11,1),%xmm7,%xmm0
    38c9:	vbroadcastss -0x4(%rdx),%xmm9
    38cf:	vorps  -0x40(%rax),%xmm5,%xmm5
    38d4:	vfmadd231ps (%rsi,%rcx,1),%xmm9,%xmm0
    38da:	mov    %r10,%rcx
    38dd:	vorps  -0x30(%rax),%xmm4,%xmm4
    38e2:	vorps  -0x20(%rax),%xmm3,%xmm3
    38e7:	vorps  -0x10(%rax),%xmm2,%xmm2
    38ec:	vfmadd231ps (%rsi,%r8,1),%xmm8,%xmm1
    38f2:	cmp    %r10,%r9
    38f5:	jne    3880 <_Z2v4RK4Args+0x50>
    38f7:	vaddps %xmm1,%xmm0,%xmm10
    38fb:	vshufps $0x55,%xmm10,%xmm10,%xmm11
    3901:	vaddss %xmm10,%xmm11,%xmm13
    3906:	vunpckhps %xmm10,%xmm10,%xmm14
    390b:	vshufps $0xff,%xmm10,%xmm10,%xmm0
    3911:	vaddss %xmm14,%xmm13,%xmm15
    3916:	vaddss %xmm0,%xmm15,%xmm0
    391a:	vmovaps %xmm5,0x371e(%rip)        # 7040 <_ZL6g_sink>
    3922:	vmovaps %xmm4,0x3726(%rip)        # 7050 <_ZL6g_sink+0x10>
    392a:	vmovaps %xmm3,0x372e(%rip)        # 7060 <_ZL6g_sink+0x20>
    3932:	vmovaps %xmm2,0x3736(%rip)        # 7070 <_ZL6g_sink+0x30>
    393a:	ret    
    393b:	nopl   0x0(%rax,%rax,1)
    3940:	vxorps %xmm2,%xmm2,%xmm2
    3944:	vxorps %xmm0,%xmm0,%xmm0
    3948:	vmovaps %xmm2,%xmm3
    394c:	vmovaps %xmm2,%xmm4
    3950:	vmovaps %xmm2,%xmm5
    3954:	jmp    391a <_Z2v4RK4Args+0xea>
    3956:	cs nopw 0x0(%rax,%rax,1)

0000000000003960 <_Z3v4bRK4Args>:
    3960:	endbr64 
    3964:	mov    0x20(%rdi),%ecx
    3967:	cmp    $0x3,%ecx
    396a:	jle    3b38 <_Z3v4bRK4Args+0x1d8>
    3970:	mov    (%rdi),%rsi
    3973:	mov    0x10(%rdi),%rdx
    3977:	vxorps %xmm13,%xmm13,%xmm13
    397c:	mov    0x8(%rdi),%rax
    3980:	lea    -0x4(%rcx),%edi
    3983:	vmovaps %xmm13,%xmm12
    3988:	vmovaps %xmm13,%xmm11
    398d:	lea    0x10(%rsi),%r9
    3991:	vmovaps %xmm13,%xmm10
    3996:	shr    $0x2,%edi
    3999:	vmovaps %xmm13,%xmm14
    399e:	shl    $0x4,%rdi
    39a2:	vmovaps %xmm13,%xmm2
    39a6:	lea    (%rdi,%r9,1),%r8
    39aa:	and    $0x10,%edi
    39ad:	je     3a85 <_Z3v4bRK4Args+0x125>
    39b3:	movslq (%rsi),%r10
    39b6:	movslq 0x4(%rsi),%r11
    39ba:	add    $0x10,%rax
    39be:	movslq 0x8(%rsi),%rcx
    39c2:	vbroadcastss -0x10(%rax),%xmm10
    39c8:	movslq 0xc(%rsi),%rsi
    39cc:	shl    $0x4,%r10
    39d0:	shl    $0x4,%r11
    39d4:	vbroadcastss -0xc(%rax),%xmm11
    39da:	vmovaps %xmm10,%xmm2
    39de:	shl    $0x4,%rcx
    39e2:	vfmadd132ps (%rdx,%r10,1),%xmm14,%xmm2
    39e8:	vbroadcastss -0x8(%rax),%xmm12
    39ee:	shl    $0x4,%rsi
    39f2:	vfmadd231ps (%rdx,%r11,1),%xmm11,%xmm14
    39f8:	vbroadcastss -0x4(%rax),%xmm13
    39fe:	vfmadd231ps (%rdx,%rcx,1),%xmm12,%xmm2
    3a04:	vfmadd231ps (%rdx,%rsi,1),%xmm13,%xmm14
    3a0a:	mov    %r9,%rsi
    3a0d:	add    $0x10,%r9
    3a11:	jmp    3a85 <_Z3v4bRK4Args+0x125>
    3a13:	nopl   0x0(%rax,%rax,1)
    3a18:	movslq (%r9),%rsi
    3a1b:	movslq 0x4(%r9),%rdi
    3a1f:	add    $0x20,%rax
    3a23:	vbroadcastss -0x10(%rax),%xmm9
    3a29:	vbroadcastss -0xc(%rax),%xmm4
    3a2f:	movslq 0x8(%r9),%r10
    3a33:	movslq 0xc(%r9),%r11
    3a37:	shl    $0x4,%rsi
    3a3b:	shl    $0x4,%rdi
    3a3f:	vbroadcastss -0x8(%rax),%xmm2
    3a45:	vbroadcastss -0x4(%rax),%xmm14
    3a4b:	vorps  %xmm9,%xmm8,%xmm10
    3a50:	vorps  %xmm4,%xmm7,%xmm11
    3a54:	shl    $0x4,%r10
    3a58:	vfmadd132ps (%rdx,%rdi,1),%xmm0,%xmm4
    3a5e:	shl    $0x4,%r11
    3a62:	vfmadd132ps (%rdx,%rsi,1),%xmm3,%xmm9
    3a68:	vorps  %xmm2,%xmm6,%xmm12
    3a6c:	vorps  %xmm14,%xmm5,%xmm13
    3a71:	vfmadd132ps (%rdx,%r10,1),%xmm9,%xmm2
    3a77:	vfmadd132ps (%rdx,%r11,1),%xmm4,%xmm14
    3a7d:	lea    0x10(%r9),%rsi
    3a81:	add    $0x20,%r9
    3a85:	movslq (%rsi),%rdi
    3a88:	movslq 0x4(%rsi),%r10
    3a8c:	vbroadcastss (%rax),%xmm9
    3a91:	vbroadcastss 0x4(%rax),%xmm4
    3a97:	movslq 0x8(%rsi),%r11
    3a9b:	movslq 0xc(%rsi),%rcx
    3a9f:	shl    $0x4,%rdi
    3aa3:	shl    $0x4,%r10
    3aa7:	vbroadcastss 0x8(%rax),%xmm3
    3aad:	vbroadcastss 0xc(%rax),%xmm0
    3ab3:	vorps  %xmm9,%xmm10,%xmm8
    3ab8:	vorps  %xmm4,%xmm11,%xmm7
    3abc:	shl    $0x4,%r11
    3ac0:	vfmadd132ps (%rdx,%rdi,1),%xmm2,%xmm9
    3ac6:	shl    $0x4,%rcx
    3aca:	vfmadd132ps (%rdx,%r10,1),%xmm14,%xmm4
    3ad0:	vorps  %xmm3,%xmm12,%xmm6
    3ad4:	vorps  %xmm0,%xmm13,%xmm5
    3ad8:	vfmadd132ps (%rdx,%r11,1),%xmm9,%xmm3
    3ade:	vfmadd132ps (%rdx,%rcx,1),%xmm4,%xmm0
    3ae4:	cmp    %r9,%r8
    3ae7:	jne    3a18 <_Z3v4bRK4Args+0xb8>
    3aed:	vaddps %xmm3,%xmm0,%xmm14
    3af1:	vshufps $0x55,%xmm14,%xmm14,%xmm1
    3af7:	vaddss %xmm14,%xmm1,%xmm10
    3afc:	vunpckhps %xmm14,%xmm14,%xmm11
    3b01:	vshufps $0xff,%xmm14,%xmm14,%xmm13
    3b07:	vaddss %xmm11,%xmm10,%xmm12
    3b0c:	vaddss %xmm13,%xmm12,%xmm0
    3b11:	vmovaps %xmm8,0x3527(%rip)        # 7040 <_ZL6g_sink>
    3b19:	vmovaps %xmm7,0x352f(%rip)        # 7050 <_ZL6g_sink+0x10>
    3b21:	vmovaps %xmm6,0x3537(%rip)        # 7060 <_ZL6g_sink+0x20>
    3b29:	vmovaps %xmm5,0x353f(%rip)        # 7070 <_ZL6g_sink+0x30>
    3b31:	ret    
    3b32:	nopw   0x0(%rax,%rax,1)
    3b38:	vxorps %xmm5,%xmm5,%xmm5
    3b3c:	vxorps %xmm0,%xmm0,%xmm0
    3b40:	vmovaps %xmm5,%xmm6
    3b44:	vmovaps %xmm5,%xmm7
    3b48:	vmovaps %xmm5,%xmm8
    3b4c:	jmp    3b11 <_Z3v4bRK4Args+0x1b1>
    3b4e:	xchg   %ax,%ax

0000000000003b50 <_Z3v4cRK4Args>:
    3b50:	endbr64 
    3b54:	mov    0x20(%rdi),%edx
    3b57:	cmp    $0x3,%edx
    3b5a:	jle    3c60 <_Z3v4cRK4Args+0x110>
    3b60:	mov    (%rdi),%rax
    3b63:	lea    -0x4(%rdx),%r10d
    3b67:	vxorps %xmm2,%xmm2,%xmm2
    3b6b:	mov    0x10(%rdi),%r8
    3b6f:	shr    $0x2,%r10d
    3b73:	mov    0x8(%rdi),%rdi
    3b77:	vmovaps %xmm2,%xmm3
    3b7b:	vmovaps %xmm2,%xmm4
    3b7f:	lea    0x10(%rax),%r9
    3b83:	vmovaps %xmm2,%xmm5
    3b87:	shl    $0x4,%r10
    3b8b:	vmovaps %xmm2,%xmm0
    3b8f:	add    %r9,%r10
    3b92:	vmovaps %xmm2,%xmm1
    3b96:	jmp    3ba4 <_Z3v4cRK4Args+0x54>
    3b98:	nopl   0x0(%rax,%rax,1)
    3ba0:	add    $0x10,%r9
    3ba4:	movslq (%rax),%rsi
    3ba7:	movslq 0x4(%rax),%rcx
    3bab:	add    $0x10,%rdi
    3baf:	movslq 0x8(%rax),%r11
    3bb3:	movslq 0xc(%rax),%rax
    3bb7:	shl    $0x4,%rsi
    3bbb:	shl    $0x4,%rcx
    3bbf:	vbroadcastss -0x10(%rdi),%xmm6
    3bc5:	vbroadcastss -0xc(%rdi),%xmm7
    3bcb:	add    %r8,%rsi
    3bce:	add    %r8,%rcx
    3bd1:	shl    $0x4,%r11
    3bd5:	vbroadcastss -0x8(%rdi),%xmm8
    3bdb:	shl    $0x4,%rax
    3bdf:	add    %r8,%r11
    3be2:	vorps  0x10(%rsi),%xmm5,%xmm5
    3be7:	vbroadcastss -0x4(%rdi),%xmm9
    3bed:	add    %r8,%rax
    3bf0:	vfmadd231ps (%rsi),%xmm6,%xmm1
    3bf5:	vorps  0x10(%rcx),%xmm4,%xmm4
    3bfa:	vfmadd231ps (%rcx),%xmm7,%xmm0
    3bff:	vorps  0x10(%rax),%xmm2,%xmm2
    3c04:	vfmadd231ps (%rax),%xmm9,%xmm0
    3c09:	vfmadd231ps (%r11),%xmm8,%xmm1
    3c0e:	mov    %r9,%rax
    3c11:	vorps  0x10(%r11),%xmm3,%xmm3
    3c17:	cmp    %r9,%r10
    3c1a:	jne    3ba0 <_Z3v4cRK4Args+0x50>
    3c1c:	vaddps %xmm1,%xmm0,%xmm10
    3c20:	vshufps $0x55,%xmm10,%xmm10,%xmm11
    3c26:	vaddss %xmm10,%xmm11,%xmm13
    3c2b:	vunpckhps %xmm10,%xmm10,%xmm14
    3c30:	vshufps $0xff,%xmm10,%xmm10,%xmm0
    3c36:	vaddss %xmm14,%xmm13,%xmm15
    3c3b:	vaddss %xmm0,%xmm15,%xmm0
    3c3f:	vmovaps %xmm5,0x33f9(%rip)        # 7040 <_ZL6g_sink>
    3c47:	vmovaps %xmm4,0x3401(%rip)        # 7050 <_ZL6g_sink+0x10>
    3c4f:	vmovaps %xmm3,0x3409(%rip)        # 7060 <_ZL6g_sink+0x20>
    3c57:	vmovaps %xmm2,0x3411(%rip)        # 7070 <_ZL6g_sink+0x30>
    3c5f:	ret    
    3c60:	vxorps %xmm2,%xmm2,%xmm2
    3c64:	vxorps %xmm0,%xmm0,%xmm0
    3c68:	vmovaps %xmm2,%xmm3
    3c6c:	vmovaps %xmm2,%xmm4
    3c70:	vmovaps %xmm2,%xmm5
    3c74:	jmp    3c3f <_Z3v4cRK4Args+0xef>
    3c76:	cs nopw 0x0(%rax,%rax,1)

0000000000003c80 <_Z2v5RK4Args>:
    3c80:	endbr64 
    3c84:	mov    0x20(%rdi),%edx
    3c87:	cmp    $0x3,%edx
    3c8a:	jle    3df0 <_Z2v5RK4Args+0x170>
    3c90:	mov    0x10(%rdi),%rcx
    3c94:	lea    -0x4(%rdx),%esi
    3c97:	vxorps %xmm13,%xmm13,%xmm13
    3c9c:	mov    0x8(%rdi),%rax
    3ca0:	vmovaps %xmm13,%xmm12
    3ca5:	shr    $0x2,%esi
    3ca8:	vmovaps %xmm13,%xmm11
    3cad:	vmovaps %xmm13,%xmm10
    3cb2:	lea    0x40(%rcx),%r8
    3cb6:	vmovaps %xmm13,%xmm14
    3cbb:	shl    $0x6,%rsi
    3cbf:	vmovaps %xmm13,%xmm2
    3cc3:	lea    (%rsi,%r8,1),%rdi
    3cc7:	and    $0x40,%esi
    3cca:	je     3d5e <_Z2v5RK4Args+0xde>
    3cd0:	vbroadcastss (%rax),%xmm10
    3cd5:	vbroadcastss 0x4(%rax),%xmm11
    3cdb:	add    $0x10,%rax
    3cdf:	vbroadcastss -0x8(%rax),%xmm12
    3ce5:	vbroadcastss -0x4(%rax),%xmm13
    3ceb:	vmovaps %xmm10,%xmm2
    3cef:	vfmadd132ps (%rcx),%xmm14,%xmm2
    3cf4:	vfmadd231ps 0x10(%rcx),%xmm11,%xmm14
    3cfa:	vfmadd231ps 0x20(%rcx),%xmm12,%xmm2
    3d00:	vfmadd231ps 0x30(%rcx),%xmm13,%xmm14
    3d06:	mov    %r8,%rcx
    3d09:	add    $0x40,%r8
    3d0d:	jmp    3d5e <_Z2v5RK4Args+0xde>
    3d0f:	nop
    3d10:	vbroadcastss 0x10(%rax),%xmm9
    3d16:	vbroadcastss 0x14(%rax),%xmm4
    3d1c:	lea    0x40(%r8),%rcx
    3d20:	add    $0x20,%rax
    3d24:	vbroadcastss -0x8(%rax),%xmm2
    3d2a:	vbroadcastss -0x4(%rax),%xmm14
    3d30:	sub    $0xffffffffffffff80,%r8
    3d34:	vorps  %xmm9,%xmm8,%xmm10
    3d39:	vfmadd132ps -0x80(%r8),%xmm3,%xmm9
    3d3f:	vorps  %xmm4,%xmm7,%xmm11
    3d43:	vfmadd132ps -0x70(%r8),%xmm0,%xmm4
    3d49:	vorps  %xmm2,%xmm6,%xmm12
    3d4d:	vorps  %xmm14,%xmm5,%xmm13
    3d52:	vfmadd132ps -0x60(%r8),%xmm9,%xmm2
    3d58:	vfmadd132ps -0x50(%r8),%xmm4,%xmm14
    3d5e:	vbroadcastss (%rax),%xmm9
    3d63:	vbroadcastss 0x4(%rax),%xmm4
    3d69:	vbroadcastss 0x8(%rax),%xmm3
    3d6f:	vbroadcastss 0xc(%rax),%xmm0
    3d75:	vorps  %xmm9,%xmm10,%xmm8
    3d7a:	vfmadd132ps (%rcx),%xmm2,%xmm9
    3d7f:	vorps  %xmm4,%xmm11,%xmm7
    3d83:	vfmadd132ps 0x10(%rcx),%xmm14,%xmm4
    3d89:	vorps  %xmm3,%xmm12,%xmm6
    3d8d:	vorps  %xmm0,%xmm13,%xmm5
    3d91:	vfmadd132ps 0x20(%rcx),%xmm9,%xmm3
    3d97:	vfmadd132ps 0x30(%rcx),%xmm4,%xmm0
    3d9d:	cmp    %r8,%rdi
    3da0:	jne    3d10 <_Z2v5RK4Args+0x90>
    3da6:	vaddps %xmm3,%xmm0,%xmm14
    3daa:	vshufps $0x55,%xmm14,%xmm14,%xmm1
    3db0:	vaddss %xmm14,%xmm1,%xmm10
    3db5:	vunpckhps %xmm14,%xmm14,%xmm11
    3dba:	vshufps $0xff,%xmm14,%xmm14,%xmm13
    3dc0:	vaddss %xmm11,%xmm10,%xmm12
    3dc5:	vaddss %xmm13,%xmm12,%xmm0
    3dca:	vmovaps %xmm8,0x326e(%rip)        # 7040 <_ZL6g_sink>
    3dd2:	vmovaps %xmm7,0x3276(%rip)        # 7050 <_ZL6g_sink+0x10>
    3dda:	vmovaps %xmm6,0x327e(%rip)        # 7060 <_ZL6g_sink+0x20>
    3de2:	vmovaps %xmm5,0x3286(%rip)        # 7070 <_ZL6g_sink+0x30>
    3dea:	ret    
    3deb:	nopl   0x0(%rax,%rax,1)
    3df0:	vxorps %xmm5,%xmm5,%xmm5
    3df4:	vxorps %xmm0,%xmm0,%xmm0
    3df8:	vmovaps %xmm5,%xmm6
    3dfc:	vmovaps %xmm5,%xmm7
    3e00:	vmovaps %xmm5,%xmm8
    3e04:	jmp    3dca <_Z2v5RK4Args+0x14a>
    3e06:	cs nopw 0x0(%rax,%rax,1)

0000000000003e10 <_ZNSt6vectorIiSaIiEEC1EmRKS0_>:
    3e10:	endbr64 
    3e14:	push   %r13
    3e16:	mov    %rsi,%rax
    3e19:	push   %r12
    3e1b:	push   %rbp
    3e1c:	push   %rbx
    3e1d:	sub    $0x8,%rsp
    3e21:	shr    $0x3d,%rax
    3e25:	jne    3eda <_ZNSt6vectorIiSaIiEEC1EmRKS0_+0xca>
    3e2b:	vpxor  %xmm0,%xmm0,%xmm0
    3e2f:	movq   $0x0,0x10(%rdi)
    3e37:	mov    %rdi,%rbx
    3e3a:	mov    %rsi,%rbp
    3e3d:	vmovdqu %xmm0,(%rdi)
    3e41:	lea    0x0(,%rsi,4),%r13
    3e49:	test   %rsi,%rsi
    3e4c:	je     3ea0 <_ZNSt6vectorIiSaIiEEC1EmRKS0_+0x90>
    3e4e:	mov    %r13,%rdi
    3e51:	call   1150 <_Znwm@plt>
    3e56:	vmovq  %rax,%xmm1
    3e5b:	lea    (%rax,%r13,1),%r12
    3e5f:	movl   $0x0,(%rax)
    3e65:	lea    0x4(%rax),%rdi
    3e69:	vpunpcklqdq %xmm1,%xmm1,%xmm2
    3e6d:	mov    %r12,0x10(%rbx)
    3e71:	vmovdqu %xmm2,(%rbx)
    3e75:	cmp    $0x1,%rbp
    3e79:	je     3ec8 <_ZNSt6vectorIiSaIiEEC1EmRKS0_+0xb8>
    3e7b:	cmp    %rdi,%r12
    3e7e:	je     3e8b <_ZNSt6vectorIiSaIiEEC1EmRKS0_+0x7b>
    3e80:	lea    -0x4(%r13),%rdx
    3e84:	xor    %esi,%esi
    3e86:	call   1120 <memset@plt>
    3e8b:	mov    %r12,0x8(%rbx)
    3e8f:	add    $0x8,%rsp
    3e93:	pop    %rbx
    3e94:	pop    %rbp
    3e95:	pop    %r12
    3e97:	pop    %r13
    3e99:	ret    
    3e9a:	nopw   0x0(%rax,%rax,1)
    3ea0:	xor    %r12d,%r12d
    3ea3:	movq   $0x0,(%rdi)
    3eaa:	movq   $0x0,0x10(%rdi)
    3eb2:	mov    %r12,0x8(%rbx)
    3eb6:	add    $0x8,%rsp
    3eba:	pop    %rbx
    3ebb:	pop    %rbp
    3ebc:	pop    %r12
    3ebe:	pop    %r13
    3ec0:	ret    
    3ec1:	nopl   0x0(%rax)
    3ec8:	mov    %rdi,%r12
    3ecb:	mov    %r12,0x8(%rbx)
    3ecf:	add    $0x8,%rsp
    3ed3:	pop    %rbx
    3ed4:	pop    %rbp
    3ed5:	pop    %r12
    3ed7:	pop    %r13
    3ed9:	ret    
    3eda:	lea    0x1127(%rip),%rdi        # 5008 <_IO_stdin_used+0x8>
    3ee1:	call   1110 <_ZSt20__throw_length_errorPKc@plt>
    3ee6:	cs nopw 0x0(%rax,%rax,1)

0000000000003ef0 <_ZNSt6vectorIfSaIfEEC1EmRKS0_>:
    3ef0:	endbr64 
    3ef4:	push   %r13
    3ef6:	mov    %rsi,%rax
    3ef9:	push   %r12
    3efb:	push   %rbp
    3efc:	push   %rbx
    3efd:	sub    $0x8,%rsp
    3f01:	shr    $0x3d,%rax
    3f05:	jne    3fba <_ZNSt6vectorIfSaIfEEC1EmRKS0_+0xca>
    3f0b:	vpxor  %xmm0,%xmm0,%xmm0
    3f0f:	movq   $0x0,0x10(%rdi)
    3f17:	mov    %rdi,%rbx
    3f1a:	mov    %rsi,%rbp
    3f1d:	vmovdqu %xmm0,(%rdi)
    3f21:	lea    0x0(,%rsi,4),%r13
    3f29:	test   %rsi,%rsi
    3f2c:	je     3f80 <_ZNSt6vectorIfSaIfEEC1EmRKS0_+0x90>
    3f2e:	mov    %r13,%rdi
    3f31:	call   1150 <_Znwm@plt>
    3f36:	vmovq  %rax,%xmm1
    3f3b:	lea    (%rax,%r13,1),%r12
    3f3f:	movl   $0x0,(%rax)
    3f45:	lea    0x4(%rax),%rdi
    3f49:	vpunpcklqdq %xmm1,%xmm1,%xmm2
    3f4d:	mov    %r12,0x10(%rbx)
    3f51:	vmovdqu %xmm2,(%rbx)
    3f55:	cmp    $0x1,%rbp
    3f59:	je     3fa8 <_ZNSt6vectorIfSaIfEEC1EmRKS0_+0xb8>
    3f5b:	cmp    %rdi,%r12
    3f5e:	je     3f6b <_ZNSt6vectorIfSaIfEEC1EmRKS0_+0x7b>
    3f60:	lea    -0x4(%r13),%rdx
    3f64:	xor    %esi,%esi
    3f66:	call   1120 <memset@plt>
    3f6b:	mov    %r12,0x8(%rbx)
    3f6f:	add    $0x8,%rsp
    3f73:	pop    %rbx
    3f74:	pop    %rbp
    3f75:	pop    %r12
    3f77:	pop    %r13
    3f79:	ret    
    3f7a:	nopw   0x0(%rax,%rax,1)
    3f80:	xor    %r12d,%r12d
    3f83:	movq   $0x0,(%rdi)
    3f8a:	movq   $0x0,0x10(%rdi)
    3f92:	mov    %r12,0x8(%rbx)
    3f96:	add    $0x8,%rsp
    3f9a:	pop    %rbx
    3f9b:	pop    %rbp
    3f9c:	pop    %r12
    3f9e:	pop    %r13
    3fa0:	ret    
    3fa1:	nopl   0x0(%rax)
    3fa8:	mov    %rdi,%r12
    3fab:	mov    %r12,0x8(%rbx)
    3faf:	add    $0x8,%rsp
    3fb3:	pop    %rbx
    3fb4:	pop    %rbp
    3fb5:	pop    %r12
    3fb7:	pop    %r13
    3fb9:	ret    
    3fba:	lea    0x1047(%rip),%rdi        # 5008 <_IO_stdin_used+0x8>
    3fc1:	call   1110 <_ZSt20__throw_length_errorPKc@plt>
    3fc6:	cs nopw 0x0(%rax,%rax,1)

0000000000003fd0 <_ZNSt6vectorIS_IdSaIdEESaIS1_EED1Ev>:
    3fd0:	endbr64 
    3fd4:	push   %r12
    3fd6:	mov    %rdi,%r12
    3fd9:	push   %rbp
    3fda:	push   %rbx
    3fdb:	mov    0x8(%rdi),%rbx
    3fdf:	mov    (%rdi),%rbp
    3fe2:	cmp    %rbp,%rbx
    3fe5:	je     4012 <_ZNSt6vectorIS_IdSaIdEESaIS1_EED1Ev+0x42>
    3fe7:	nopw   0x0(%rax,%rax,1)
    3ff0:	mov    0x0(%rbp),%rdi
    3ff4:	test   %rdi,%rdi
    3ff7:	je     4030 <_ZNSt6vectorIS_IdSaIdEESaIS1_EED1Ev+0x60>
    3ff9:	mov    0x10(%rbp),%rsi
    3ffd:	add    $0x18,%rbp
    4001:	sub    %rdi,%rsi
    4004:	call   1160 <_ZdlPvm@plt>
    4009:	cmp    %rbp,%rbx
    400c:	jne    3ff0 <_ZNSt6vectorIS_IdSaIdEESaIS1_EED1Ev+0x20>
    400e:	mov    (%r12),%rbp
    4012:	test   %rbp,%rbp
    4015:	je     4040 <_ZNSt6vectorIS_IdSaIdEESaIS1_EED1Ev+0x70>
    4017:	mov    0x10(%r12),%rsi
    401c:	pop    %rbx
    401d:	mov    %rbp,%rdi
    4020:	sub    %rbp,%rsi
    4023:	pop    %rbp
    4024:	pop    %r12
    4026:	jmp    1160 <_ZdlPvm@plt>
    402b:	nopl   0x0(%rax,%rax,1)
    4030:	add    $0x18,%rbp
    4034:	cmp    %rbp,%rbx
    4037:	jne    3ff0 <_ZNSt6vectorIS_IdSaIdEESaIS1_EED1Ev+0x20>
    4039:	jmp    400e <_ZNSt6vectorIS_IdSaIdEESaIS1_EED1Ev+0x3e>
    403b:	nopl   0x0(%rax,%rax,1)
    4040:	pop    %rbx
    4041:	pop    %rbp
    4042:	pop    %r12
    4044:	ret    
    4045:	cs nopw 0x0(%rax,%rax,1)
    404f:	nop

0000000000004050 <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_>:
    4050:	endbr64 
    4054:	push   %r15
    4056:	mov    %rdx,%r15
    4059:	movabs $0xfffffffffffffff,%rdx
    4063:	push   %r14
    4065:	push   %r13
    4067:	push   %r12
    4069:	push   %rbp
    406a:	push   %rbx
    406b:	sub    $0x18,%rsp
    406f:	mov    0x8(%rdi),%r12
    4073:	mov    (%rdi),%r14
    4076:	mov    %r12,%rax
    4079:	sub    %r14,%rax
    407c:	sar    $0x3,%rax
    4080:	cmp    %rdx,%rax
    4083:	je     419b <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_+0x14b>
    4089:	test   %rax,%rax
    408c:	mov    $0x1,%ebx
    4091:	mov    %rsi,%rdx
    4094:	mov    %rdi,%rbp
    4097:	cmovne %rax,%rbx
    409b:	xor    %ecx,%ecx
    409d:	mov    %rsi,%r13
    40a0:	add    %rbx,%rax
    40a3:	setb   %cl
    40a6:	sub    %r14,%rdx
    40a9:	test   %rcx,%rcx
    40ac:	jne    4158 <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_+0x108>
    40b2:	test   %rax,%rax
    40b5:	jne    4180 <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_+0x130>
    40bb:	xor    %ebx,%ebx
    40bd:	xor    %edi,%edi
    40bf:	vmovsd (%r15),%xmm0
    40c4:	sub    %r13,%r12
    40c7:	lea    0x8(%rdi,%rdx,1),%r15
    40cc:	vmovq  %rdi,%xmm2
    40d1:	lea    (%r15,%r12,1),%r8
    40d5:	vpinsrq $0x1,%r8,%xmm2,%xmm1
    40db:	vmovsd %xmm0,(%rdi,%rdx,1)
    40e0:	vmovdqa %xmm1,(%rsp)
    40e5:	test   %rdx,%rdx
    40e8:	jg     4118 <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_+0xc8>
    40ea:	test   %r12,%r12
    40ed:	jg     4140 <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_+0xf0>
    40ef:	test   %r14,%r14
    40f2:	jne    4125 <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_+0xd5>
    40f4:	vmovdqa (%rsp),%xmm3
    40f9:	mov    %rbx,0x10(%rbp)
    40fd:	vmovdqu %xmm3,0x0(%rbp)
    4102:	add    $0x18,%rsp
    4106:	pop    %rbx
    4107:	pop    %rbp
    4108:	pop    %r12
    410a:	pop    %r13
    410c:	pop    %r14
    410e:	pop    %r15
    4110:	ret    
    4111:	nopl   0x0(%rax)
    4118:	mov    %r14,%rsi
    411b:	call   1190 <memmove@plt>
    4120:	test   %r12,%r12
    4123:	jg     4140 <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_+0xf0>
    4125:	mov    0x10(%rbp),%rsi
    4129:	mov    %r14,%rdi
    412c:	sub    %r14,%rsi
    412f:	call   1160 <_ZdlPvm@plt>
    4134:	jmp    40f4 <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_+0xa4>
    4136:	cs nopw 0x0(%rax,%rax,1)
    4140:	mov    %r12,%rdx
    4143:	mov    %r13,%rsi
    4146:	mov    %r15,%rdi
    4149:	call   1130 <memcpy@plt>
    414e:	test   %r14,%r14
    4151:	je     40f4 <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_+0xa4>
    4153:	jmp    4125 <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_+0xd5>
    4155:	nopl   (%rax)
    4158:	movabs $0x7ffffffffffffff8,%rbx
    4162:	mov    %rbx,%rdi
    4165:	mov    %rdx,(%rsp)
    4169:	call   1150 <_Znwm@plt>
    416e:	mov    (%rsp),%rdx
    4172:	mov    %rax,%rdi
    4175:	add    %rax,%rbx
    4178:	jmp    40bf <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_+0x6f>
    417d:	nopl   (%rax)
    4180:	movabs $0xfffffffffffffff,%rsi
    418a:	cmp    %rsi,%rax
    418d:	cmova  %rsi,%rax
    4191:	lea    0x0(,%rax,8),%rbx
    4199:	jmp    4162 <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_+0x112>
    419b:	lea    0xe97(%rip),%rdi        # 5039 <_IO_stdin_used+0x39>
    41a2:	call   1110 <_ZSt20__throw_length_errorPKc@plt>

Disassembly of section .fini:

00000000000041a8 <_fini>:
    41a8:	endbr64 
    41ac:	sub    $0x8,%rsp
    41b0:	add    $0x8,%rsp
    41b4:	ret    
