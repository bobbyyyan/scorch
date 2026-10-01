
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
    11c4:	mov    -0x188(%rbp),%rdi
    11cb:	sub    %rdi,%rsi
    11ce:	vzeroupper 
    11d1:	call   1160 <_ZdlPvm@plt>
    11d6:	lea    -0xb0(%rbp),%rdi
    11dd:	call   4830 <_ZNSt6vectorIS_IdSaIdEESaIS1_EED1Ev>
    11e2:	mov    -0x178(%rbp),%rdi
    11e9:	mov    $0x68,%esi
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
    12a9:	jg     28ad <main+0x163d>
    12af:	movl   $0x1000,-0x190(%rbp)
    12b9:	movl   $0x1f,-0x180(%rbp)
    12c3:	movl   $0x1000,-0x16c(%rbp)
    12cd:	movl   $0x3039,-0x18c(%rbp)
    12d7:	mov    -0x16c(%rbp),%r15d
    12de:	lea    -0x70(%rbp),%r13
    12e2:	lea    -0x130(%rbp),%rdi
    12e9:	mov    %r13,%rdx
    12ec:	movslq %r15d,%r14
    12ef:	mov    %r14,%rsi
    12f2:	mov    %r14,%r12
    12f5:	call   4670 <_ZNSt6vectorIiSaIiEEC1EmRKS0_>
    12fa:	lea    -0x110(%rbp),%rdi
    1301:	mov    %r13,%rdx
    1304:	mov    %r14,%rsi
    1307:	call   4750 <_ZNSt6vectorIfSaIfEEC1EmRKS0_>
    130c:	mov    -0x190(%rbp),%ebx
    1312:	lea    -0xf0(%rbp),%rdi
    1319:	mov    %r13,%rdx
    131c:	lea    0x2(%rbx),%esi
    131f:	movslq %esi,%rsi
    1322:	shl    $0x2,%rsi
    1326:	call   4750 <_ZNSt6vectorIfSaIfEEC1EmRKS0_>
    132b:	add    $0x2,%r12d
    132f:	lea    -0xd0(%rbp),%rdi
    1336:	mov    %r13,%rdx
    1339:	movslq %r12d,%rsi
    133c:	shl    $0x2,%rsi
    1340:	call   4750 <_ZNSt6vectorIfSaIfEEC1EmRKS0_>
    1345:	mov    -0x110(%rbp),%rax
    134c:	mov    -0x130(%rbp),%rdx
    1353:	lea    0x1(%r15),%ecx
    1357:	mov    %rax,-0x1a8(%rbp)
    135e:	mov    %rdx,-0x1b0(%rbp)
    1365:	cmp    %ebx,%ecx
    1367:	jge    292e <main+0x16be>
    136d:	mov    -0x18c(%rbp),%ecx
    1373:	test   %r15d,%r15d
    1376:	jle    1898 <main+0x628>
    137c:	mov    -0x16c(%rbp),%edi
    1382:	mov    -0x1b0(%rbp),%r11
    1389:	mov    -0x18c(%rbp),%ecx
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
    13f1:	imul   $0x19660d,-0x18c(%rbp),%ecx
    13fb:	xor    %edx,%edx
    13fd:	mov    -0x1b0(%rbp),%r14
    1404:	mov    %r9,%r12
    1407:	add    $0x3c6ef35f,%ecx
    140d:	mov    %ecx,%eax
    140f:	divl   -0x190(%rbp)
    1415:	mov    %edx,(%r14)
    1418:	imul   $0x19660d,%ecx,%ecx
    141e:	xor    %edx,%edx
    1420:	add    $0x4,%r12
    1424:	add    $0x3c6ef35f,%ecx
    142a:	mov    %ecx,%eax
    142c:	divl   -0x190(%rbp)
    1432:	mov    %edx,-0x4(%r12)
    1437:	imul   $0x19660d,%ecx,%ecx
    143d:	xor    %edx,%edx
    143f:	add    $0x4,%r12
    1443:	add    $0x3c6ef35f,%ecx
    1449:	mov    %ecx,%eax
    144b:	divl   -0x190(%rbp)
    1451:	mov    %edx,-0x4(%r12)
    1456:	imul   $0x19660d,%ecx,%ecx
    145c:	xor    %edx,%edx
    145e:	add    $0x4,%r12
    1462:	add    $0x3c6ef35f,%ecx
    1468:	mov    %ecx,%eax
    146a:	divl   -0x190(%rbp)
    1470:	mov    %edx,-0x4(%r12)
    1475:	imul   $0x19660d,%ecx,%ecx
    147b:	xor    %edx,%edx
    147d:	add    $0x4,%r12
    1481:	add    $0x3c6ef35f,%ecx
    1487:	mov    %ecx,%eax
    1489:	divl   -0x190(%rbp)
    148f:	mov    %edx,-0x4(%r12)
    1494:	imul   $0x19660d,%ecx,%ecx
    149a:	xor    %edx,%edx
    149c:	add    $0x4,%r12
    14a0:	add    $0x3c6ef35f,%ecx
    14a6:	mov    %ecx,%eax
    14a8:	divl   -0x190(%rbp)
    14ae:	mov    %edx,-0x4(%r12)
    14b3:	imul   $0x19660d,%ecx,%ecx
    14b9:	xor    %edx,%edx
    14bb:	add    $0x4,%r12
    14bf:	add    $0x3c6ef35f,%ecx
    14c5:	mov    %ecx,%eax
    14c7:	divl   -0x190(%rbp)
    14cd:	mov    %edx,-0x4(%r12)
    14d2:	cmp    %r8,%r12
    14d5:	je     15b3 <main+0x343>
    14db:	mov    -0x190(%rbp),%ebx
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
    15ba:	vmovss 0x3b4a(%rip),%xmm0        # 510c <_IO_stdin_used+0x10c>
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
    1621:	vxorps %xmm5,%xmm5,%xmm5
    1625:	mov    -0x1a8(%rbp),%rdi
    162c:	add    $0x3c6ef35f,%ecx
    1632:	mov    %ecx,%esi
    1634:	shr    $0x8,%esi
    1637:	movzwl %si,%r11d
    163b:	mov    %r12,%rsi
    163e:	vcvtsi2ss %r11d,%xmm5,%xmm1
    1643:	vmulss %xmm0,%xmm1,%xmm2
    1647:	vmovss %xmm2,(%rdi)
    164b:	imul   $0x19660d,%ecx,%ecx
    1651:	vxorps %xmm6,%xmm6,%xmm6
    1655:	add    $0x4,%rsi
    1659:	add    $0x3c6ef35f,%ecx
    165f:	mov    %ecx,%r9d
    1662:	shr    $0x8,%r9d
    1666:	movzwl %r9w,%r15d
    166a:	vcvtsi2ss %r15d,%xmm6,%xmm3
    166f:	vmulss %xmm0,%xmm3,%xmm4
    1673:	vmovss %xmm4,-0x4(%rsi)
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
    1703:	vxorps %xmm5,%xmm5,%xmm5
    1707:	add    $0x4,%rsi
    170b:	add    $0x3c6ef35f,%ecx
    1711:	mov    %ecx,%r11d
    1714:	shr    $0x8,%r11d
    1718:	movzwl %r11w,%edi
    171c:	vcvtsi2ss %edi,%xmm5,%xmm1
    1720:	vmulss %xmm0,%xmm1,%xmm2
    1724:	vmovss %xmm2,-0x4(%rsi)
    1729:	imul   $0x19660d,%ecx,%ecx
    172f:	vxorps %xmm6,%xmm6,%xmm6
    1733:	add    $0x4,%rsi
    1737:	add    $0x3c6ef35f,%ecx
    173d:	mov    %ecx,%r9d
    1740:	shr    $0x8,%r9d
    1744:	movzwl %r9w,%r15d
    1748:	vcvtsi2ss %r15d,%xmm6,%xmm3
    174d:	vmulss %xmm0,%xmm3,%xmm4
    1751:	vmovss %xmm4,-0x4(%rsi)
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
    1809:	vcvtsi2ss %edx,%xmm7,%xmm5
    180d:	vmulss %xmm0,%xmm14,%xmm15
    1811:	add    $0x3c6ef35f,%r8d
    1818:	imul   $0x19660d,%r8d,%r11d
    181f:	mov    %r8d,%r12d
    1822:	vmovss %xmm13,-0x18(%rsi)
    1827:	shr    $0x8,%r12d
    182b:	movzwl %r12w,%ebx
    182f:	vcvtsi2ss %ebx,%xmm7,%xmm2
    1833:	vmulss %xmm0,%xmm5,%xmm1
    1837:	add    $0x3c6ef35f,%r11d
    183e:	imul   $0x19660d,%r11d,%r15d
    1845:	mov    %r11d,%edi
    1848:	vmovss %xmm15,-0x14(%rsi)
    184d:	shr    $0x8,%edi
    1850:	movzwl %di,%r9d
    1854:	vcvtsi2ss %r9d,%xmm7,%xmm3
    1859:	lea    0x3c6ef35f(%r15),%ecx
    1860:	vmulss %xmm0,%xmm2,%xmm6
    1864:	mov    %ecx,%r14d
    1867:	vmovss %xmm1,-0x10(%rsi)
    186c:	shr    $0x8,%r14d
    1870:	movzwl %r14w,%eax
    1874:	vcvtsi2ss %eax,%xmm7,%xmm7
    1878:	vmulss %xmm0,%xmm3,%xmm4
    187c:	vmovss %xmm6,-0xc(%rsi)
    1881:	vmulss %xmm0,%xmm7,%xmm8
    1885:	vmovss %xmm4,-0x8(%rsi)
    188a:	vmovss %xmm8,-0x4(%rsi)
    188f:	cmp    %rsi,%r10
    1892:	jne    175f <main+0x4ef>
    1898:	mov    -0xf0(%rbp),%r10
    189f:	mov    -0xe8(%rbp),%rsi
    18a6:	sub    %r10,%rsi
    18a9:	mov    %r10,-0x1a0(%rbp)
    18b0:	mov    %rsi,%rdx
    18b3:	shr    $0x2,%rdx
    18b7:	je     1b9e <main+0x92e>
    18bd:	mov    %r10,%r11
    18c0:	add    %rsi,%r10
    18c3:	sub    $0x4,%rsi
    18c7:	vmovss 0x383d(%rip),%xmm0        # 510c <_IO_stdin_used+0x10c>
    18cf:	shr    $0x2,%rsi
    18d3:	add    $0x1,%rsi
    18d7:	and    $0x7,%esi
    18da:	je     1a5e <main+0x7ee>
    18e0:	cmp    $0x1,%rsi
    18e4:	je     1a28 <main+0x7b8>
    18ea:	cmp    $0x2,%rsi
    18ee:	je     19fa <main+0x78a>
    18f4:	cmp    $0x3,%rsi
    18f8:	je     19ce <main+0x75e>
    18fe:	cmp    $0x4,%rsi
    1902:	je     19a4 <main+0x734>
    1908:	cmp    $0x5,%rsi
    190c:	je     1975 <main+0x705>
    190e:	cmp    $0x6,%rsi
    1912:	je     1948 <main+0x6d8>
    1914:	imul   $0x19660d,%ecx,%ecx
    191a:	vxorps %xmm9,%xmm9,%xmm9
    191f:	mov    -0x1a0(%rbp),%rbx
    1926:	lea    0x4(%rbx),%r11
    192a:	add    $0x3c6ef35f,%ecx
    1930:	mov    %ecx,%r8d
    1933:	shr    $0x8,%r8d
    1937:	movzwl %r8w,%r12d
    193b:	vcvtsi2ss %r12d,%xmm9,%xmm10
    1940:	vmulss %xmm0,%xmm10,%xmm11
    1944:	vmovss %xmm11,(%rbx)
    1948:	imul   $0x19660d,%ecx,%ecx
    194e:	vxorps %xmm12,%xmm12,%xmm12
    1953:	add    $0x4,%r11
    1957:	add    $0x3c6ef35f,%ecx
    195d:	mov    %ecx,%edi
    195f:	shr    $0x8,%edi
    1962:	movzwl %di,%r9d
    1966:	vcvtsi2ss %r9d,%xmm12,%xmm13
    196b:	vmulss %xmm0,%xmm13,%xmm14
    196f:	vmovss %xmm14,-0x4(%r11)
    1975:	imul   $0x19660d,%ecx,%ecx
    197b:	vxorps %xmm15,%xmm15,%xmm15
    1980:	add    $0x4,%r11
    1984:	add    $0x3c6ef35f,%ecx
    198a:	mov    %ecx,%r15d
    198d:	shr    $0x8,%r15d
    1991:	movzwl %r15w,%r14d
    1995:	vcvtsi2ss %r14d,%xmm15,%xmm5
    199a:	vmulss %xmm0,%xmm5,%xmm1
    199e:	vmovss %xmm1,-0x4(%r11)
    19a4:	imul   $0x19660d,%ecx,%ecx
    19aa:	vxorps %xmm2,%xmm2,%xmm2
    19ae:	add    $0x4,%r11
    19b2:	add    $0x3c6ef35f,%ecx
    19b8:	mov    %ecx,%eax
    19ba:	shr    $0x8,%eax
    19bd:	movzwl %ax,%esi
    19c0:	vcvtsi2ss %esi,%xmm2,%xmm6
    19c4:	vmulss %xmm0,%xmm6,%xmm3
    19c8:	vmovss %xmm3,-0x4(%r11)
    19ce:	imul   $0x19660d,%ecx,%ecx
    19d4:	vxorps %xmm4,%xmm4,%xmm4
    19d8:	add    $0x4,%r11
    19dc:	add    $0x3c6ef35f,%ecx
    19e2:	mov    %ecx,%edx
    19e4:	shr    $0x8,%edx
    19e7:	movzwl %dx,%r8d
    19eb:	vcvtsi2ss %r8d,%xmm4,%xmm7
    19f0:	vmulss %xmm0,%xmm7,%xmm8
    19f4:	vmovss %xmm8,-0x4(%r11)
    19fa:	imul   $0x19660d,%ecx,%ecx
    1a00:	vxorps %xmm9,%xmm9,%xmm9
    1a05:	add    $0x4,%r11
    1a09:	add    $0x3c6ef35f,%ecx
    1a0f:	mov    %ecx,%r12d
    1a12:	shr    $0x8,%r12d
    1a16:	movzwl %r12w,%ebx
    1a1a:	vcvtsi2ss %ebx,%xmm9,%xmm10
    1a1e:	vmulss %xmm0,%xmm10,%xmm11
    1a22:	vmovss %xmm11,-0x4(%r11)
    1a28:	imul   $0x19660d,%ecx,%ecx
    1a2e:	vxorps %xmm12,%xmm12,%xmm12
    1a33:	add    $0x4,%r11
    1a37:	add    $0x3c6ef35f,%ecx
    1a3d:	mov    %ecx,%edi
    1a3f:	shr    $0x8,%edi
    1a42:	movzwl %di,%r9d
    1a46:	vcvtsi2ss %r9d,%xmm12,%xmm13
    1a4b:	vmulss %xmm0,%xmm13,%xmm14
    1a4f:	vmovss %xmm14,-0x4(%r11)
    1a55:	cmp    %r10,%r11
    1a58:	je     1b9e <main+0x92e>
    1a5e:	imul   $0x19660d,%ecx,%r15d
    1a65:	vxorps %xmm15,%xmm15,%xmm15
    1a6a:	add    $0x20,%r11
    1a6e:	add    $0x3c6ef35f,%r15d
    1a75:	imul   $0x19660d,%r15d,%eax
    1a7c:	mov    %r15d,%ecx
    1a7f:	shr    $0x8,%ecx
    1a82:	movzwl %cx,%r14d
    1a86:	add    $0x3c6ef35f,%eax
    1a8b:	vcvtsi2ss %r14d,%xmm15,%xmm5
    1a90:	imul   $0x19660d,%eax,%r8d
    1a97:	mov    %eax,%esi
    1a99:	shr    $0x8,%esi
    1a9c:	movzwl %si,%edx
    1a9f:	vcvtsi2ss %edx,%xmm15,%xmm2
    1aa3:	vmulss %xmm0,%xmm5,%xmm1
    1aa7:	add    $0x3c6ef35f,%r8d
    1aae:	imul   $0x19660d,%r8d,%edi
    1ab5:	mov    %r8d,%r12d
    1ab8:	shr    $0x8,%r12d
    1abc:	movzwl %r12w,%ebx
    1ac0:	vcvtsi2ss %ebx,%xmm15,%xmm3
    1ac4:	vmulss %xmm0,%xmm2,%xmm6
    1ac8:	add    $0x3c6ef35f,%edi
    1ace:	imul   $0x19660d,%edi,%r14d
    1ad5:	mov    %edi,%r9d
    1ad8:	vmovss %xmm1,-0x20(%r11)
    1ade:	shr    $0x8,%r9d
    1ae2:	movzwl %r9w,%r15d
    1ae6:	vcvtsi2ss %r15d,%xmm15,%xmm7
    1aeb:	vmulss %xmm0,%xmm3,%xmm4
    1aef:	add    $0x3c6ef35f,%r14d
    1af6:	imul   $0x19660d,%r14d,%esi
    1afd:	mov    %r14d,%ecx
    1b00:	vmovss %xmm6,-0x1c(%r11)
    1b06:	shr    $0x8,%ecx
    1b09:	movzwl %cx,%eax
    1b0c:	vcvtsi2ss %eax,%xmm15,%xmm9
    1b10:	vmulss %xmm0,%xmm7,%xmm8
    1b14:	add    $0x3c6ef35f,%esi
    1b1a:	imul   $0x19660d,%esi,%r12d
    1b21:	mov    %esi,%edx
    1b23:	vmovss %xmm4,-0x18(%r11)
    1b29:	shr    $0x8,%edx
    1b2c:	movzwl %dx,%r8d
    1b30:	vcvtsi2ss %r8d,%xmm15,%xmm11
    1b35:	vmulss %xmm0,%xmm9,%xmm10
    1b39:	add    $0x3c6ef35f,%r12d
    1b40:	imul   $0x19660d,%r12d,%r9d
    1b47:	mov    %r12d,%ebx
    1b4a:	vmovss %xmm8,-0x14(%r11)
    1b50:	shr    $0x8,%ebx
    1b53:	movzwl %bx,%edi
    1b56:	vcvtsi2ss %edi,%xmm15,%xmm13
    1b5a:	lea    0x3c6ef35f(%r9),%ecx
    1b61:	vmulss %xmm0,%xmm11,%xmm12
    1b65:	mov    %ecx,%r15d
    1b68:	vmovss %xmm10,-0x10(%r11)
    1b6e:	shr    $0x8,%r15d
    1b72:	movzwl %r15w,%r14d
    1b76:	vcvtsi2ss %r14d,%xmm15,%xmm15
    1b7b:	vmulss %xmm0,%xmm13,%xmm14
    1b7f:	vmovss %xmm12,-0xc(%r11)
    1b85:	vmulss %xmm0,%xmm15,%xmm5
    1b89:	vmovss %xmm14,-0x8(%r11)
    1b8f:	vmovss %xmm5,-0x4(%r11)
    1b95:	cmp    %r10,%r11
    1b98:	jne    1a5e <main+0x7ee>
    1b9e:	mov    -0xd0(%rbp),%r10
    1ba5:	mov    -0xc8(%rbp),%r11
    1bac:	vmovq  -0x1a0(%rbp),%xmm0
    1bb4:	vmovq  -0x1b0(%rbp),%xmm2
    1bbc:	sub    %r10,%r11
    1bbf:	mov    %r10,-0x198(%rbp)
    1bc6:	vpinsrq $0x1,-0x1a8(%rbp),%xmm2,%xmm6
    1bd0:	mov    %r11,%rax
    1bd3:	vpinsrq $0x1,%r10,%xmm0,%xmm1
    1bd9:	shr    $0x2,%rax
    1bdd:	vinserti128 $0x1,%xmm1,%ymm6,%ymm3
    1be3:	je     1edd <main+0xc6d>
    1be9:	mov    -0x198(%rbp),%rsi
    1bf0:	vmovss 0x3514(%rip),%xmm8        # 510c <_IO_stdin_used+0x10c>
    1bf8:	mov    %rsi,%rbx
    1bfb:	add    %r11,%rsi
    1bfe:	sub    $0x4,%r11
    1c02:	shr    $0x2,%r11
    1c06:	add    $0x1,%r11
    1c0a:	and    $0x7,%r11d
    1c0e:	je     1d95 <main+0xb25>
    1c14:	cmp    $0x1,%r11
    1c18:	je     1d5f <main+0xaef>
    1c1e:	cmp    $0x2,%r11
    1c22:	je     1d31 <main+0xac1>
    1c28:	cmp    $0x3,%r11
    1c2c:	je     1d07 <main+0xa97>
    1c32:	cmp    $0x4,%r11
    1c36:	je     1cd9 <main+0xa69>
    1c3c:	cmp    $0x5,%r11
    1c40:	je     1caa <main+0xa3a>
    1c42:	cmp    $0x6,%r11
    1c46:	je     1c7d <main+0xa0d>
    1c48:	imul   $0x19660d,%ecx,%ecx
    1c4e:	vxorps %xmm4,%xmm4,%xmm4
    1c52:	mov    -0x198(%rbp),%r12
    1c59:	lea    0x4(%r12),%rbx
    1c5e:	add    $0x3c6ef35f,%ecx
    1c64:	mov    %ecx,%edx
    1c66:	shr    $0x8,%edx
    1c69:	movzwl %dx,%r8d
    1c6d:	vcvtsi2ss %r8d,%xmm4,%xmm7
    1c72:	vmulss %xmm8,%xmm7,%xmm9
    1c77:	vmovss %xmm9,(%r12)
    1c7d:	imul   $0x19660d,%ecx,%ecx
    1c83:	vxorps %xmm10,%xmm10,%xmm10
    1c88:	add    $0x4,%rbx
    1c8c:	add    $0x3c6ef35f,%ecx
    1c92:	mov    %ecx,%edi
    1c94:	shr    $0x8,%edi
    1c97:	movzwl %di,%r9d
    1c9b:	vcvtsi2ss %r9d,%xmm10,%xmm11
    1ca0:	vmulss %xmm8,%xmm11,%xmm12
    1ca5:	vmovss %xmm12,-0x4(%rbx)
    1caa:	imul   $0x19660d,%ecx,%ecx
    1cb0:	vxorps %xmm13,%xmm13,%xmm13
    1cb5:	add    $0x4,%rbx
    1cb9:	add    $0x3c6ef35f,%ecx
    1cbf:	mov    %ecx,%r15d
    1cc2:	shr    $0x8,%r15d
    1cc6:	movzwl %r15w,%r14d
    1cca:	vcvtsi2ss %r14d,%xmm13,%xmm14
    1ccf:	vmulss %xmm8,%xmm14,%xmm15
    1cd4:	vmovss %xmm15,-0x4(%rbx)
    1cd9:	imul   $0x19660d,%ecx,%ecx
    1cdf:	vxorps %xmm5,%xmm5,%xmm5
    1ce3:	add    $0x4,%rbx
    1ce7:	add    $0x3c6ef35f,%ecx
    1ced:	mov    %ecx,%r10d
    1cf0:	shr    $0x8,%r10d
    1cf4:	movzwl %r10w,%r11d
    1cf8:	vcvtsi2ss %r11d,%xmm5,%xmm0
    1cfd:	vmulss %xmm8,%xmm0,%xmm1
    1d02:	vmovss %xmm1,-0x4(%rbx)
    1d07:	imul   $0x19660d,%ecx,%ecx
    1d0d:	vxorps %xmm2,%xmm2,%xmm2
    1d11:	add    $0x4,%rbx
    1d15:	add    $0x3c6ef35f,%ecx
    1d1b:	mov    %ecx,%eax
    1d1d:	shr    $0x8,%eax
    1d20:	movzwl %ax,%edx
    1d23:	vcvtsi2ss %edx,%xmm2,%xmm6
    1d27:	vmulss %xmm8,%xmm6,%xmm4
    1d2c:	vmovss %xmm4,-0x4(%rbx)
    1d31:	imul   $0x19660d,%ecx,%ecx
    1d37:	vxorps %xmm7,%xmm7,%xmm7
    1d3b:	add    $0x4,%rbx
    1d3f:	add    $0x3c6ef35f,%ecx
    1d45:	mov    %ecx,%r8d
    1d48:	shr    $0x8,%r8d
    1d4c:	movzwl %r8w,%r12d
    1d50:	vcvtsi2ss %r12d,%xmm7,%xmm9
    1d55:	vmulss %xmm8,%xmm9,%xmm10
    1d5a:	vmovss %xmm10,-0x4(%rbx)
    1d5f:	imul   $0x19660d,%ecx,%ecx
    1d65:	vxorps %xmm11,%xmm11,%xmm11
    1d6a:	add    $0x4,%rbx
    1d6e:	add    $0x3c6ef35f,%ecx
    1d74:	mov    %ecx,%edi
    1d76:	shr    $0x8,%edi
    1d79:	movzwl %di,%r9d
    1d7d:	vcvtsi2ss %r9d,%xmm11,%xmm12
    1d82:	vmulss %xmm8,%xmm12,%xmm13
    1d87:	vmovss %xmm13,-0x4(%rbx)
    1d8c:	cmp    %rbx,%rsi
    1d8f:	je     1edd <main+0xc6d>
    1d95:	imul   $0x19660d,%ecx,%r15d
    1d9c:	vxorps %xmm14,%xmm14,%xmm14
    1da1:	add    $0x20,%rbx
    1da5:	add    $0x3c6ef35f,%r15d
    1dac:	imul   $0x19660d,%r15d,%r10d
    1db3:	mov    %r15d,%ecx
    1db6:	shr    $0x8,%ecx
    1db9:	movzwl %cx,%r14d
    1dbd:	add    $0x3c6ef35f,%r10d
    1dc4:	vcvtsi2ss %r14d,%xmm14,%xmm15
    1dc9:	imul   $0x19660d,%r10d,%edx
    1dd0:	mov    %r10d,%r11d
    1dd3:	shr    $0x8,%r11d
    1dd7:	movzwl %r11w,%eax
    1ddb:	vcvtsi2ss %eax,%xmm14,%xmm0
    1ddf:	vmulss %xmm8,%xmm15,%xmm5
    1de4:	add    $0x3c6ef35f,%edx
    1dea:	imul   $0x19660d,%edx,%edi
    1df0:	mov    %edx,%r8d
    1df3:	shr    $0x8,%r8d
    1df7:	movzwl %r8w,%r12d
    1dfb:	vcvtsi2ss %r12d,%xmm14,%xmm2
    1e00:	vmulss %xmm8,%xmm0,%xmm1
    1e05:	add    $0x3c6ef35f,%edi
    1e0b:	imul   $0x19660d,%edi,%r14d
    1e12:	mov    %edi,%r9d
    1e15:	vmovss %xmm5,-0x20(%rbx)
    1e1a:	shr    $0x8,%r9d
    1e1e:	movzwl %r9w,%r15d
    1e22:	vcvtsi2ss %r15d,%xmm14,%xmm4
    1e27:	vmulss %xmm8,%xmm2,%xmm6
    1e2c:	add    $0x3c6ef35f,%r14d
    1e33:	imul   $0x19660d,%r14d,%r11d
    1e3a:	mov    %r14d,%ecx
    1e3d:	vmovss %xmm1,-0x1c(%rbx)
    1e42:	shr    $0x8,%ecx
    1e45:	movzwl %cx,%r10d
    1e49:	vcvtsi2ss %r10d,%xmm14,%xmm9
    1e4e:	vmulss %xmm8,%xmm4,%xmm7
    1e53:	add    $0x3c6ef35f,%r11d
    1e5a:	imul   $0x19660d,%r11d,%r8d
    1e61:	mov    %r11d,%eax
    1e64:	vmovss %xmm6,-0x18(%rbx)
    1e69:	shr    $0x8,%eax
    1e6c:	movzwl %ax,%edx
    1e6f:	vcvtsi2ss %edx,%xmm14,%xmm11
    1e73:	vmulss %xmm8,%xmm9,%xmm10
    1e78:	add    $0x3c6ef35f,%r8d
    1e7f:	imul   $0x19660d,%r8d,%r9d
    1e86:	mov    %r8d,%r12d
    1e89:	vmovss %xmm7,-0x14(%rbx)
    1e8e:	shr    $0x8,%r12d
    1e92:	movzwl %r12w,%edi
    1e96:	vcvtsi2ss %edi,%xmm14,%xmm13
    1e9a:	lea    0x3c6ef35f(%r9),%ecx
    1ea1:	vmulss %xmm8,%xmm11,%xmm12
    1ea6:	mov    %ecx,%r15d
    1ea9:	vmovss %xmm10,-0x10(%rbx)
    1eae:	shr    $0x8,%r15d
    1eb2:	movzwl %r15w,%r14d
    1eb6:	vcvtsi2ss %r14d,%xmm14,%xmm14
    1ebb:	vmulss %xmm8,%xmm13,%xmm15
    1ec0:	vmovss %xmm12,-0xc(%rbx)
    1ec5:	vmulss %xmm8,%xmm14,%xmm5
    1eca:	vmovss %xmm15,-0x8(%rbx)
    1ecf:	vmovss %xmm5,-0x4(%rbx)
    1ed4:	cmp    %rbx,%rsi
    1ed7:	jne    1d95 <main+0xb25>
    1edd:	mov    -0x16c(%rbp),%esi
    1ee3:	mov    $0x1,%edi
    1ee8:	vmovdqa %ymm3,-0x70(%rbp)
    1eed:	lea    -0x90(%rbp),%rbx
    1ef4:	mov    %esi,-0x50(%rbp)
    1ef7:	mov    %rbx,%rsi
    1efa:	vzeroupper 
    1efd:	call   1140 <clock_gettime@plt>
    1f02:	vxorpd %xmm3,%xmm3,%xmm3
    1f06:	mov    %r13,%rdi
    1f09:	vcvtsi2sdq -0x90(%rbp),%xmm3,%xmm8
    1f12:	vcvtsi2sdq -0x88(%rbp),%xmm3,%xmm0
    1f1b:	vfmadd132sd 0x31f4(%rip),%xmm0,%xmm8        # 5118 <_IO_stdin_used+0x118>
    1f24:	vmovsd %xmm8,-0x150(%rbp)
    1f2c:	call   2b20 <_Z2v0RK4Args>
    1f31:	vxorps %xmm2,%xmm2,%xmm2
    1f35:	mov    $0x40,%ecx
    1f3a:	vmovaps %xmm0,%xmm1
    1f3e:	vmovss %xmm2,-0x144(%rbp)
    1f46:	cs nopw 0x0(%rax,%rax,1)
    1f50:	vaddss -0x144(%rbp),%xmm1,%xmm6
    1f58:	vaddss %xmm1,%xmm6,%xmm4
    1f5c:	vaddss %xmm1,%xmm4,%xmm7
    1f60:	vaddss %xmm1,%xmm7,%xmm9
    1f64:	vaddss %xmm1,%xmm9,%xmm10
    1f68:	vaddss %xmm1,%xmm10,%xmm11
    1f6c:	vaddss %xmm1,%xmm11,%xmm12
    1f70:	vaddss %xmm1,%xmm12,%xmm13
    1f74:	vmovss %xmm13,-0x144(%rbp)
    1f7c:	sub    $0x8,%ecx
    1f7f:	jne    1f50 <main+0xce0>
    1f81:	mov    $0x1,%edi
    1f86:	mov    %rbx,%rsi
    1f89:	call   1140 <clock_gettime@plt>
    1f8e:	vxorpd %xmm15,%xmm15,%xmm15
    1f93:	mov    $0x1,%r11d
    1f99:	vmovsd 0x3187(%rip),%xmm1        # 5128 <_IO_stdin_used+0x128>
    1fa1:	vcvtsi2sdq -0x90(%rbp),%xmm15,%xmm14
    1faa:	mov    $0x68,%edi
    1faf:	vcvtsi2sdq -0x88(%rbp),%xmm15,%xmm5
    1fb8:	vfmadd132sd 0x3157(%rip),%xmm5,%xmm14        # 5118 <_IO_stdin_used+0x118>
    1fc1:	vsubsd -0x150(%rbp),%xmm14,%xmm3
    1fc9:	vmulsd 0x314f(%rip),%xmm3,%xmm8        # 5120 <_IO_stdin_used+0x120>
    1fd1:	vmaxsd 0x3137(%rip),%xmm8,%xmm0        # 5110 <_IO_stdin_used+0x110>
    1fd9:	vdivsd %xmm0,%xmm1,%xmm2
    1fdd:	vcvttsd2si %xmm2,%r10d
    1fe1:	test   %r10d,%r10d
    1fe4:	cmovg  %r10d,%r11d
    1fe8:	mov    %r11d,-0x148(%rbp)
    1fef:	call   1150 <_Znwm@plt>
    1ff4:	mov    0x3145(%rip),%rdx        # 5140 <_IO_stdin_used+0x140>
    1ffb:	vpxor  %xmm4,%xmm4,%xmm4
    1fff:	vmovapd 0x3139(%rip),%ymm6        # 5140 <_IO_stdin_used+0x140>
    2007:	mov    $0x138,%edi
    200c:	mov    %rax,-0x178(%rbp)
    2013:	mov    %rdx,0x60(%rax)
    2017:	vmovupd %ymm6,(%rax)
    201b:	vmovupd %ymm6,0x20(%rax)
    2020:	vmovupd %ymm6,0x40(%rax)
    2025:	vmovdqu %xmm4,-0xa8(%rbp)
    202d:	vzeroupper 
    2030:	call   1150 <_Znwm@plt>
    2035:	mov    %rax,%r8
    2038:	vpxor  %xmm7,%xmm7,%xmm7
    203c:	vpxor  %xmm9,%xmm9,%xmm9
    2041:	mov    $0xd,%esi
    2046:	lea    -0x140(%rbp),%r15
    204d:	mov    %rax,-0x168(%rbp)
    2054:	mov    %rbx,%rdi
    2057:	mov    %rax,-0xb0(%rbp)
    205e:	mov    %r15,%rdx
    2061:	lea    0x138(%rax),%rax
    2068:	mov    %rax,-0xa0(%rbp)
    206f:	movq   $0x0,0x130(%r8)
    207a:	mov    %rax,-0xa8(%rbp)
    2081:	vmovdqu %ymm7,(%r8)
    2086:	vmovdqu %ymm7,0x20(%r8)
    208c:	vmovdqu %ymm7,0x40(%r8)
    2092:	vmovdqu %ymm7,0x60(%r8)
    2098:	vmovdqu %ymm7,0x80(%r8)
    20a1:	vmovdqu %ymm7,0xa0(%r8)
    20aa:	vmovdqu %ymm7,0xc0(%r8)
    20b3:	vmovdqu %ymm7,0xe0(%r8)
    20bc:	vmovdqu %ymm7,0x100(%r8)
    20c5:	vmovdqu %xmm9,0x120(%r8)
    20ce:	vzeroupper 
    20d1:	call   4670 <_ZNSt6vectorIiSaIiEEC1EmRKS0_>
    20d6:	mov    -0x90(%rbp),%r12
    20dd:	vmovdqa 0x307b(%rip),%ymm10        # 5160 <_IO_stdin_used+0x160>
    20e5:	movl   $0x0,-0x170(%rbp)
    20ef:	vmovdqa 0x3089(%rip),%xmm11        # 5180 <_IO_stdin_used+0x180>
    20f7:	mov    -0x18c(%rbp),%edi
    20fd:	mov    -0x180(%rbp),%r9d
    2104:	vmovdqu %ymm10,(%r12)
    210a:	movl   $0xc,0x30(%r12)
    2113:	xor    $0x5bf03635,%edi
    2119:	vmovdqu %xmm11,0x20(%r12)
    2120:	mov    %r12,-0x188(%rbp)
    2127:	add    $0x34,%r12
    212b:	mov    %edi,-0x17c(%rbp)
    2131:	mov    %r12,-0x160(%rbp)
    2138:	test   %r9d,%r9d
    213b:	jle    28a5 <main+0x1635>
    2141:	vzeroupper 
    2144:	imul   $0x19660d,-0x17c(%rbp),%r14d
    214f:	mov    -0x188(%rbp),%r8
    2156:	mov    0x30(%r8),%r12d
    215a:	lea    0x3c6ef35f(%r14),%esi
    2161:	imul   $0x4ec4ec4f,%rsi,%rbx
    2168:	mov    %esi,%edx
    216a:	imul   $0x19660d,%esi,%r9d
    2171:	shr    $0x22,%rbx
    2175:	lea    (%rbx,%rbx,2),%ecx
    2178:	lea    0x3c6ef35f(%r9),%r14d
    217f:	lea    (%rbx,%rcx,4),%r11d
    2183:	mov    $0xaaaaaaab,%ebx
    2188:	mov    %r14,%r10
    218b:	sub    %r11d,%edx
    218e:	imul   %rbx,%r14
    2192:	mov    %r10d,%ecx
    2195:	lea    (%r8,%rdx,4),%rax
    2199:	mov    (%rax),%edi
    219b:	shr    $0x23,%r14
    219f:	mov    %edi,0x30(%r8)
    21a3:	lea    (%r14,%r14,2),%esi
    21a7:	mov    $0xba2e8ba3,%r14d
    21ad:	mov    %r12d,(%rax)
    21b0:	imul   $0x19660d,%r10d,%r12d
    21b7:	shl    $0x2,%esi
    21ba:	mov    0x2c(%r8),%edx
    21be:	sub    %esi,%ecx
    21c0:	lea    (%r8,%rcx,4),%r11
    21c4:	lea    0x3c6ef35f(%r12),%edi
    21cc:	mov    (%r11),%eax
    21cf:	mov    %rdi,%r9
    21d2:	imul   %r14,%rdi
    21d6:	mov    $0xcccccccd,%r14d
    21dc:	imul   $0x19660d,%r9d,%r12d
    21e3:	mov    %r9d,%ecx
    21e6:	mov    %eax,0x2c(%r8)
    21ea:	mov    %edx,(%r11)
    21ed:	mov    0x28(%r8),%edx
    21f1:	shr    $0x23,%rdi
    21f5:	lea    (%rdi,%rdi,4),%r10d
    21f9:	lea    (%rdi,%r10,2),%esi
    21fd:	lea    0x3c6ef35f(%r12),%edi
    2205:	sub    %esi,%ecx
    2207:	mov    %rdi,%r9
    220a:	imul   %r14,%rdi
    220e:	lea    (%r8,%rcx,4),%r11
    2212:	mov    %r9d,%esi
    2215:	mov    (%r11),%eax
    2218:	mov    %eax,0x28(%r8)
    221c:	imul   $0x19660d,%r9d,%eax
    2223:	shr    $0x23,%rdi
    2227:	lea    (%rdi,%rdi,4),%r10d
    222b:	mov    %edx,(%r11)
    222e:	mov    0x24(%r8),%ecx
    2232:	add    %r10d,%r10d
    2235:	lea    0x3c6ef35f(%rax),%r12d
    223c:	sub    %r10d,%esi
    223f:	imul   $0x38e38e39,%r12,%r9
    2246:	lea    (%r8,%rsi,4),%r11
    224a:	mov    %r12d,%esi
    224d:	imul   $0x19660d,%r12d,%eax
    2254:	mov    (%r11),%edx
    2257:	shr    $0x21,%r9
    225b:	mov    %edx,0x24(%r8)
    225f:	lea    (%r9,%r9,8),%r10d
    2263:	add    $0x3c6ef35f,%eax
    2268:	mov    %ecx,(%r11)
    226b:	mov    0x20(%r8),%ecx
    226f:	sub    %r10d,%esi
    2272:	mov    %eax,%r12d
    2275:	lea    (%r8,%rsi,4),%r11
    2279:	and    $0x7,%r12d
    227d:	imul   $0x19660d,%eax,%esi
    2283:	mov    (%r11),%edx
    2286:	lea    (%r8,%r12,4),%rdi
    228a:	mov    %edx,0x20(%r8)
    228e:	mov    %ecx,(%r11)
    2291:	lea    0x3c6ef35f(%rsi),%r11d
    2298:	mov    (%rdi),%r10d
    229b:	imul   $0x24924925,%r11,%rcx
    22a2:	mov    %r11d,%edx
    22a5:	mov    0x1c(%r8),%r9d
    22a9:	mov    %r10d,0x1c(%r8)
    22ad:	mov    %r9d,(%rdi)
    22b0:	mov    %r11d,%edi
    22b3:	mov    0x18(%r8),%r10d
    22b7:	shr    $0x20,%rcx
    22bb:	sub    %ecx,%edx
    22bd:	shr    %edx
    22bf:	add    %ecx,%edx
    22c1:	shr    $0x2,%edx
    22c4:	lea    0x0(,%rdx,8),%r12d
    22cc:	sub    %edx,%r12d
    22cf:	imul   $0x19660d,%r11d,%r11d
    22d6:	sub    %r12d,%edi
    22d9:	lea    (%r8,%rdi,4),%r9
    22dd:	lea    0x3c6ef35f(%r11),%ecx
    22e4:	mov    (%r9),%esi
    22e7:	mov    %rcx,%rax
    22ea:	imul   %rbx,%rcx
    22ee:	mov    %esi,0x18(%r8)
    22f2:	mov    %eax,%r12d
    22f5:	imul   $0x19660d,%eax,%esi
    22fb:	mov    %r10d,(%r9)
    22fe:	mov    0x14(%r8),%r9d
    2302:	shr    $0x22,%rcx
    2306:	lea    (%rcx,%rcx,2),%edx
    2309:	lea    0x3c6ef35f(%rsi),%r11d
    2310:	add    %edx,%edx
    2312:	mov    %r11,%rax
    2315:	sub    %edx,%r12d
    2318:	imul   %r14,%r11
    231c:	mov    %eax,%ecx
    231e:	lea    (%r8,%r12,4),%rdi
    2322:	mov    (%rdi),%r10d
    2325:	shr    $0x22,%r11
    2329:	mov    %r10d,0x14(%r8)
    232d:	lea    (%r11,%r11,4),%r14d
    2331:	mov    %r9d,(%rdi)
    2334:	sub    %r14d,%ecx
    2337:	mov    0x10(%r8),%r12d
    233b:	imul   $0x19660d,%eax,%r9d
    2342:	lea    (%r8,%rcx,4),%rdx
    2346:	mov    %r8,-0x150(%rbp)
    234d:	mov    (%rdx),%edi
    234f:	add    $0x3c6ef35f,%r9d
    2356:	imul   $0x19660d,%r9d,%r14d
    235d:	mov    %r9d,%r10d
    2360:	mov    %edi,0x10(%r8)
    2364:	and    $0x3,%r10d
    2368:	mov    %r12d,(%rdx)
    236b:	mov    0xc(%r8),%eax
    236f:	lea    (%r8,%r10,4),%r11
    2373:	lea    0x3c6ef35f(%r14),%ecx
    237a:	mov    (%r11),%esi
    237d:	mov    %rcx,%r12
    2380:	imul   %rbx,%rcx
    2384:	mov    %esi,0xc(%r8)
    2388:	mov    %r12d,%edx
    238b:	mov    %eax,(%r11)
    238e:	imul   $0x19660d,%r12d,%r11d
    2395:	mov    0x8(%r8),%r9d
    2399:	shr    $0x21,%rcx
    239d:	lea    (%rcx,%rcx,2),%ebx
    23a0:	lea    0x3c6ef35f(%r11),%eax
    23a7:	sub    %ebx,%edx
    23a9:	lea    (%r8,%rdx,4),%rdi
    23ad:	mov    %eax,%esi
    23af:	mov    %eax,-0x17c(%rbp)
    23b5:	mov    (%rdi),%r10d
    23b8:	and    $0x1,%esi
    23bb:	lea    (%r8,%rsi,4),%r14
    23bf:	mov    %r10d,0x8(%r8)
    23c3:	mov    %r9d,(%rdi)
    23c6:	mov    0x4(%r8),%r12d
    23ca:	mov    (%r14),%ecx
    23cd:	mov    %ecx,0x4(%r8)
    23d1:	mov    %r12d,(%r14)
    23d4:	nopl   0x0(%rax)
    23d8:	mov    -0x150(%rbp),%r8
    23df:	mov    $0x1,%edi
    23e4:	mov    %r15,%rsi
    23e7:	movslq (%r8),%r14
    23ea:	call   1140 <clock_gettime@plt>
    23ef:	vxorpd %xmm12,%xmm12,%xmm12
    23f4:	mov    -0x148(%rbp),%edi
    23fa:	lea    0x487f(%rip),%rdx        # 6c80 <_ZL4ARMS>
    2401:	vcvtsi2sdq -0x140(%rbp),%xmm12,%xmm13
    240a:	mov    %r14,%rbx
    240d:	vcvtsi2sdq -0x138(%rbp),%xmm12,%xmm15
    2416:	shl    $0x4,%rbx
    241a:	vfmadd132sd 0x2cf5(%rip),%xmm15,%xmm13        # 5118 <_IO_stdin_used+0x118>
    2423:	mov    0x8(%rdx,%rbx,1),%r12
    2428:	xor    %ebx,%ebx
    242a:	vmovsd %xmm13,-0x158(%rbp)
    2432:	and    $0x7,%edi
    2435:	je     251e <main+0x12ae>
    243b:	cmp    $0x1,%edi
    243e:	je     24f9 <main+0x1289>
    2444:	cmp    $0x2,%edi
    2447:	je     24e0 <main+0x1270>
    244d:	cmp    $0x3,%edi
    2450:	je     24c7 <main+0x1257>
    2452:	cmp    $0x4,%edi
    2455:	je     24ae <main+0x123e>
    2457:	cmp    $0x5,%edi
    245a:	je     2495 <main+0x1225>
    245c:	cmp    $0x6,%edi
    245f:	je     247c <main+0x120c>
    2461:	mov    %r13,%rdi
    2464:	call   *%r12
    2467:	vaddss -0x144(%rbp),%xmm0,%xmm14
    246f:	mov    $0x1,%ebx
    2474:	vmovss %xmm14,-0x144(%rbp)
    247c:	mov    %r13,%rdi
    247f:	call   *%r12
    2482:	vaddss -0x144(%rbp),%xmm0,%xmm5
    248a:	add    $0x1,%ebx
    248d:	vmovss %xmm5,-0x144(%rbp)
    2495:	mov    %r13,%rdi
    2498:	call   *%r12
    249b:	vaddss -0x144(%rbp),%xmm0,%xmm3
    24a3:	add    $0x1,%ebx
    24a6:	vmovss %xmm3,-0x144(%rbp)
    24ae:	mov    %r13,%rdi
    24b1:	call   *%r12
    24b4:	vaddss -0x144(%rbp),%xmm0,%xmm8
    24bc:	add    $0x1,%ebx
    24bf:	vmovss %xmm8,-0x144(%rbp)
    24c7:	mov    %r13,%rdi
    24ca:	call   *%r12
    24cd:	vaddss -0x144(%rbp),%xmm0,%xmm0
    24d5:	add    $0x1,%ebx
    24d8:	vmovss %xmm0,-0x144(%rbp)
    24e0:	mov    %r13,%rdi
    24e3:	call   *%r12
    24e6:	vaddss -0x144(%rbp),%xmm0,%xmm1
    24ee:	add    $0x1,%ebx
    24f1:	vmovss %xmm1,-0x144(%rbp)
    24f9:	mov    %r13,%rdi
    24fc:	call   *%r12
    24ff:	vaddss -0x144(%rbp),%xmm0,%xmm2
    2507:	add    $0x1,%ebx
    250a:	vmovss %xmm2,-0x144(%rbp)
    2512:	cmp    %ebx,-0x148(%rbp)
    2518:	je     25e0 <main+0x1370>
    251e:	mov    %r13,%rdi
    2521:	call   *%r12
    2524:	vaddss -0x144(%rbp),%xmm0,%xmm6
    252c:	add    $0x1,%ebx
    252f:	mov    %r13,%rdi
    2532:	vmovss %xmm6,-0x144(%rbp)
    253a:	call   *%r12
    253d:	vaddss -0x144(%rbp),%xmm0,%xmm4
    2545:	mov    %r13,%rdi
    2548:	vmovss %xmm4,-0x144(%rbp)
    2550:	call   *%r12
    2553:	vaddss -0x144(%rbp),%xmm0,%xmm7
    255b:	mov    %r13,%rdi
    255e:	vmovss %xmm7,-0x144(%rbp)
    2566:	call   *%r12
    2569:	vaddss -0x144(%rbp),%xmm0,%xmm9
    2571:	mov    %r13,%rdi
    2574:	vmovss %xmm9,-0x144(%rbp)
    257c:	call   *%r12
    257f:	vaddss -0x144(%rbp),%xmm0,%xmm10
    2587:	mov    %r13,%rdi
    258a:	vmovss %xmm10,-0x144(%rbp)
    2592:	call   *%r12
    2595:	vaddss -0x144(%rbp),%xmm0,%xmm11
    259d:	mov    %r13,%rdi
    25a0:	vmovss %xmm11,-0x144(%rbp)
    25a8:	call   *%r12
    25ab:	vaddss -0x144(%rbp),%xmm0,%xmm12
    25b3:	mov    %r13,%rdi
    25b6:	vmovss %xmm12,-0x144(%rbp)
    25be:	call   *%r12
    25c1:	vaddss -0x144(%rbp),%xmm0,%xmm13
    25c9:	add    $0x7,%ebx
    25cc:	vmovss %xmm13,-0x144(%rbp)
    25d4:	cmp    %ebx,-0x148(%rbp)
    25da:	jne    251e <main+0x12ae>
    25e0:	mov    %r15,%rsi
    25e3:	mov    $0x1,%edi
    25e8:	call   1140 <clock_gettime@plt>
    25ed:	vxorpd %xmm15,%xmm15,%xmm15
    25f2:	mov    -0x168(%rbp),%r10
    25f9:	lea    (%r14,%r14,2),%r9
    25fd:	vcvtsi2sdl -0x148(%rbp),%xmm15,%xmm8
    2605:	vcvtsi2sdl -0x16c(%rbp),%xmm15,%xmm0
    260d:	lea    (%r10,%r9,8),%rdi
    2611:	vcvtsi2sdq -0x140(%rbp),%xmm15,%xmm14
    261a:	mov    0x8(%rdi),%rsi
    261e:	vcvtsi2sdq -0x138(%rbp),%xmm15,%xmm5
    2627:	vfmadd132sd 0x2ae8(%rip),%xmm5,%xmm14        # 5118 <_IO_stdin_used+0x118>
    2630:	vsubsd -0x158(%rbp),%xmm14,%xmm3
    2638:	vmulsd %xmm0,%xmm8,%xmm1
    263c:	vdivsd %xmm1,%xmm3,%xmm2
    2640:	vmovsd %xmm2,-0x140(%rbp)
    2648:	cmp    0x10(%rdi),%rsi
    264c:	je     2890 <main+0x1620>
    2652:	vmovsd %xmm2,(%rsi)
    2656:	add    $0x8,%rsi
    265a:	mov    %rsi,0x8(%rdi)
    265e:	mov    -0x178(%rbp),%r11
    2665:	lea    (%r11,%r14,8),%rax
    2669:	vmovsd (%rax),%xmm6
    266d:	vcomisd %xmm2,%xmm6
    2671:	jbe    2868 <main+0x15f8>
    2677:	addq   $0x4,-0x150(%rbp)
    267f:	mov    -0x150(%rbp),%rcx
    2686:	vmovsd %xmm2,(%rax)
    268a:	cmp    %rcx,-0x160(%rbp)
    2691:	jne    23d8 <main+0x1168>
    2697:	addl   $0x1,-0x170(%rbp)
    269e:	mov    -0x170(%rbp),%r8d
    26a5:	cmp    -0x180(%rbp),%r8d
    26ac:	jne    2144 <main+0xed4>
    26b2:	mov    -0x18c(%rbp),%r13d
    26b9:	sub    $0x8,%rsp
    26bd:	mov    -0x190(%rbp),%r9d
    26c4:	vxorpd %xmm4,%xmm4,%xmm4
    26c8:	mov    -0x180(%rbp),%ecx
    26ce:	mov    -0x148(%rbp),%r8d
    26d5:	lea    0x29e4(%rip),%rsi        # 50c0 <_IO_stdin_used+0xc0>
    26dc:	mov    $0x1,%edi
    26e1:	push   %r13
    26e3:	mov    -0x16c(%rbp),%edx
    26e9:	mov    $0x2,%eax
    26ee:	vcvtss2sd -0x144(%rbp),%xmm0,%xmm0
    26f6:	vcvtss2sd 0x4942(%rip),%xmm4,%xmm1        # 7040 <_ZL6g_sink>
    26fe:	call   1100 <__printf_chk@plt>
    2703:	pop    %rax
    2704:	lea    0x2964(%rip),%rsi        # 506f <_IO_stdin_used+0x6f>
    270b:	pop    %rdx
    270c:	mov    $0x1,%edi
    2711:	xor    %eax,%eax
    2713:	call   1100 <__printf_chk@plt>
    2718:	mov    -0x168(%rbp),%r12
    271f:	lea    0x455a(%rip),%r15        # 6c80 <_ZL4ARMS>
    2726:	lea    0x2956(%rip),%r14        # 5083 <_IO_stdin_used+0x83>
    272d:	add    $0x8,%r12
    2731:	nopl   0x0(%rax)
    2738:	mov    -0x8(%r12),%rdx
    273d:	cmp    (%r12),%rdx
    2741:	je     2786 <main+0x1516>
    2743:	mov    (%r15),%r13
    2746:	xor    %ebx,%ebx
    2748:	nopl   0x0(%rax,%rax,1)
    2750:	vmovsd (%rdx,%rbx,8),%xmm0
    2755:	mov    %rbx,%rcx
    2758:	mov    %r13,%rdx
    275b:	mov    %r14,%rsi
    275e:	mov    $0x1,%edi
    2763:	mov    $0x1,%eax
    2768:	call   1100 <__printf_chk@plt>
    276d:	mov    -0x8(%r12),%rdx
    2772:	mov    (%r12),%rdi
    2776:	add    $0x1,%rbx
    277a:	sub    %rdx,%rdi
    277d:	sar    $0x3,%rdi
    2781:	cmp    %rdi,%rbx
    2784:	jb     2750 <main+0x14e0>
    2786:	add    $0x10,%r15
    278a:	lea    0x45bf(%rip),%r9        # 6d50 <_DYNAMIC>
    2791:	add    $0x18,%r12
    2795:	cmp    %r9,%r15
    2798:	jne    2738 <main+0x14c8>
    279a:	mov    -0x188(%rbp),%rdi
    27a1:	mov    -0x80(%rbp),%rsi
    27a5:	xor    %r12d,%r12d
    27a8:	sub    %rdi,%rsi
    27ab:	call   1160 <_ZdlPvm@plt>
    27b0:	lea    -0xb0(%rbp),%rdi
    27b7:	call   4830 <_ZNSt6vectorIS_IdSaIdEESaIS1_EED1Ev>
    27bc:	mov    -0x178(%rbp),%rdi
    27c3:	mov    $0x68,%esi
    27c8:	call   1160 <_ZdlPvm@plt>
    27cd:	mov    -0x198(%rbp),%rdi
    27d4:	test   %rdi,%rdi
    27d7:	je     27e8 <main+0x1578>
    27d9:	mov    -0xc0(%rbp),%rsi
    27e0:	sub    %rdi,%rsi
    27e3:	call   1160 <_ZdlPvm@plt>
    27e8:	mov    -0x1a0(%rbp),%rdi
    27ef:	test   %rdi,%rdi
    27f2:	je     2803 <main+0x1593>
    27f4:	mov    -0xe0(%rbp),%rsi
    27fb:	sub    %rdi,%rsi
    27fe:	call   1160 <_ZdlPvm@plt>
    2803:	mov    -0x1a8(%rbp),%rdi
    280a:	test   %rdi,%rdi
    280d:	je     281e <main+0x15ae>
    280f:	mov    -0x100(%rbp),%rsi
    2816:	sub    %rdi,%rsi
    2819:	call   1160 <_ZdlPvm@plt>
    281e:	mov    -0x1b0(%rbp),%rdi
    2825:	test   %rdi,%rdi
    2828:	je     2839 <main+0x15c9>
    282a:	mov    -0x120(%rbp),%rsi
    2831:	sub    %rdi,%rsi
    2834:	call   1160 <_ZdlPvm@plt>
    2839:	mov    -0x38(%rbp),%rax
    283d:	sub    %fs:0x28,%rax
    2846:	jne    29aa <main+0x173a>
    284c:	lea    -0x30(%rbp),%rsp
    2850:	mov    %r12d,%eax
    2853:	pop    %rbx
    2854:	pop    %rsi
    2855:	pop    %r12
    2857:	pop    %r13
    2859:	pop    %r14
    285b:	pop    %r15
    285d:	pop    %rbp
    285e:	lea    -0x8(%rsi),%rsp
    2862:	ret    
    2863:	nopl   0x0(%rax,%rax,1)
    2868:	addq   $0x4,-0x150(%rbp)
    2870:	mov    -0x150(%rbp),%rsi
    2877:	cmp    -0x160(%rbp),%rsi
    287e:	jne    23d8 <main+0x1168>
    2884:	jmp    2697 <main+0x1427>
    2889:	nopl   0x0(%rax)
    2890:	mov    %r15,%rdx
    2893:	call   48b0 <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_>
    2898:	vmovsd -0x140(%rbp),%xmm2
    28a0:	jmp    265e <main+0x13ee>
    28a5:	vzeroupper 
    28a8:	jmp    26b2 <main+0x1442>
    28ad:	mov    %edi,%r12d
    28b0:	mov    0x8(%rsi),%rdi
    28b4:	mov    %rsi,%rbx
    28b7:	mov    $0xa,%edx
    28bc:	xor    %esi,%esi
    28be:	call   1180 <strtol@plt>
    28c3:	mov    %eax,-0x16c(%rbp)
    28c9:	cmp    $0x2,%r12d
    28cd:	je     2978 <main+0x1708>
    28d3:	mov    0x10(%rbx),%rdi
    28d7:	mov    $0xa,%edx
    28dc:	xor    %esi,%esi
    28de:	call   1180 <strtol@plt>
    28e3:	mov    %eax,-0x180(%rbp)
    28e9:	cmp    $0x3,%r12d
    28ed:	je     29af <main+0x173f>
    28f3:	mov    0x18(%rbx),%rdi
    28f7:	mov    $0xa,%edx
    28fc:	xor    %esi,%esi
    28fe:	call   1180 <strtol@plt>
    2903:	mov    %eax,-0x190(%rbp)
    2909:	cmp    $0x4,%r12d
    290d:	je     299b <main+0x172b>
    2913:	mov    0x20(%rbx),%rdi
    2917:	mov    $0xa,%edx
    291c:	xor    %esi,%esi
    291e:	call   1180 <strtol@plt>
    2923:	mov    %eax,-0x18c(%rbp)
    2929:	jmp    12d7 <main+0x67>
    292e:	mov    -0x190(%rbp),%ecx
    2934:	mov    %r12d,%r8d
    2937:	mov    $0x1,%esi
    293c:	xor    %eax,%eax
    293e:	mov    0x46db(%rip),%rdi        # 7020 <stderr@GLIBC_2.2.5>
    2945:	lea    0x2707(%rip),%rdx        # 5053 <_IO_stdin_used+0x53>
    294c:	call   11a0 <__fprintf_chk@plt>
    2951:	mov    -0xd0(%rbp),%r10
    2958:	mov    -0xf0(%rbp),%r11
    295f:	mov    $0x2,%r12d
    2965:	mov    %r10,-0x198(%rbp)
    296c:	mov    %r11,-0x1a0(%rbp)
    2973:	jmp    27cd <main+0x155d>
    2978:	movl   $0x1000,-0x190(%rbp)
    2982:	movl   $0x1f,-0x180(%rbp)
    298c:	movl   $0x3039,-0x18c(%rbp)
    2996:	jmp    12d7 <main+0x67>
    299b:	movl   $0x3039,-0x18c(%rbp)
    29a5:	jmp    12d7 <main+0x67>
    29aa:	call   1170 <__stack_chk_fail@plt>
    29af:	movl   $0x1000,-0x190(%rbp)
    29b9:	movl   $0x3039,-0x18c(%rbp)
    29c3:	jmp    12d7 <main+0x67>
    29c8:	endbr64 
    29cc:	mov    %rax,%r14
    29cf:	jmp    11c0 <main.cold>
    29d4:	endbr64 
    29d8:	mov    %rax,%r14
    29db:	vzeroupper 
    29de:	jmp    1244 <main.cold+0x84>
    29e3:	endbr64 
    29e7:	mov    %rax,%r14
    29ea:	vzeroupper 
    29ed:	jmp    120e <main.cold+0x4e>
    29f2:	endbr64 
    29f6:	mov    %rax,%r14
    29f9:	vzeroupper 
    29fc:	jmp    11d6 <main.cold+0x16>
    2a01:	endbr64 
    2a05:	mov    %rax,%r14
    2a08:	vzeroupper 
    2a0b:	jmp    11e2 <main.cold+0x22>
    2a10:	endbr64 
    2a14:	mov    %rax,%r14
    2a17:	vzeroupper 
    2a1a:	jmp    11f3 <main.cold+0x33>
    2a1f:	endbr64 
    2a23:	mov    %rax,%r14
    2a26:	vzeroupper 
    2a29:	jmp    1229 <main.cold+0x69>
    2a2e:	xchg   %ax,%ax

0000000000002a30 <_start>:
    2a30:	endbr64 
    2a34:	xor    %ebp,%ebp
    2a36:	mov    %rdx,%r9
    2a39:	pop    %rsi
    2a3a:	mov    %rsp,%rdx
    2a3d:	and    $0xfffffffffffffff0,%rsp
    2a41:	push   %rax
    2a42:	push   %rsp
    2a43:	xor    %r8d,%r8d
    2a46:	xor    %ecx,%ecx
    2a48:	lea    -0x17df(%rip),%rdi        # 1270 <main>
    2a4f:	call   *0x458b(%rip)        # 6fe0 <__libc_start_main@GLIBC_2.34>
    2a55:	hlt    
    2a56:	cs nopw 0x0(%rax,%rax,1)

0000000000002a60 <deregister_tm_clones>:
    2a60:	lea    0x45b1(%rip),%rdi        # 7018 <__TMC_END__>
    2a67:	lea    0x45aa(%rip),%rax        # 7018 <__TMC_END__>
    2a6e:	cmp    %rdi,%rax
    2a71:	je     2a88 <deregister_tm_clones+0x28>
    2a73:	mov    0x456e(%rip),%rax        # 6fe8 <_ITM_deregisterTMCloneTable@Base>
    2a7a:	test   %rax,%rax
    2a7d:	je     2a88 <deregister_tm_clones+0x28>
    2a7f:	jmp    *%rax
    2a81:	nopl   0x0(%rax)
    2a88:	ret    
    2a89:	nopl   0x0(%rax)

0000000000002a90 <register_tm_clones>:
    2a90:	lea    0x4581(%rip),%rdi        # 7018 <__TMC_END__>
    2a97:	lea    0x457a(%rip),%rsi        # 7018 <__TMC_END__>
    2a9e:	sub    %rdi,%rsi
    2aa1:	mov    %rsi,%rax
    2aa4:	shr    $0x3f,%rsi
    2aa8:	sar    $0x3,%rax
    2aac:	add    %rax,%rsi
    2aaf:	sar    %rsi
    2ab2:	je     2ac8 <register_tm_clones+0x38>
    2ab4:	mov    0x453d(%rip),%rax        # 6ff8 <_ITM_registerTMCloneTable@Base>
    2abb:	test   %rax,%rax
    2abe:	je     2ac8 <register_tm_clones+0x38>
    2ac0:	jmp    *%rax
    2ac2:	nopw   0x0(%rax,%rax,1)
    2ac8:	ret    
    2ac9:	nopl   0x0(%rax)

0000000000002ad0 <__do_global_dtors_aux>:
    2ad0:	endbr64 
    2ad4:	cmpb   $0x0,0x454d(%rip)        # 7028 <completed.0>
    2adb:	jne    2b08 <__do_global_dtors_aux+0x38>
    2add:	push   %rbp
    2ade:	cmpq   $0x0,0x44f2(%rip)        # 6fd8 <__cxa_finalize@GLIBC_2.2.5>
    2ae6:	mov    %rsp,%rbp
    2ae9:	je     2af7 <__do_global_dtors_aux+0x27>
    2aeb:	mov    0x4516(%rip),%rdi        # 7008 <__dso_handle>
    2af2:	call   10f0 <__cxa_finalize@plt>
    2af7:	call   2a60 <deregister_tm_clones>
    2afc:	movb   $0x1,0x4525(%rip)        # 7028 <completed.0>
    2b03:	pop    %rbp
    2b04:	ret    
    2b05:	nopl   (%rax)
    2b08:	ret    
    2b09:	nopl   0x0(%rax)

0000000000002b10 <frame_dummy>:
    2b10:	endbr64 
    2b14:	jmp    2a90 <register_tm_clones>
    2b19:	nopl   0x0(%rax)

0000000000002b20 <_Z2v0RK4Args>:
    2b20:	endbr64 
    2b24:	mov    0x20(%rdi),%eax
    2b27:	cmp    $0x1,%eax
    2b2a:	jle    2ce8 <_Z2v0RK4Args+0x1c8>
    2b30:	sub    $0x2,%eax
    2b33:	mov    0x10(%rdi),%rdx
    2b37:	mov    (%rdi),%rcx
    2b3a:	vxorps %xmm12,%xmm12,%xmm12
    2b3f:	shr    %eax
    2b41:	mov    0x8(%rdi),%rsi
    2b45:	xor    %r11d,%r11d
    2b48:	vmovaps %xmm12,%xmm0
    2b4c:	lea    0x1(%rax),%r8d
    2b50:	shl    $0x3,%r8
    2b54:	lea    -0x8(%r8),%rdi
    2b58:	shr    $0x3,%rdi
    2b5c:	add    $0x1,%rdi
    2b60:	and    $0x3,%edi
    2b63:	je     2c06 <_Z2v0RK4Args+0xe6>
    2b69:	cmp    $0x1,%rdi
    2b6d:	je     2bcf <_Z2v0RK4Args+0xaf>
    2b6f:	cmp    $0x2,%rdi
    2b73:	je     2ba1 <_Z2v0RK4Args+0x81>
    2b75:	movslq (%rcx),%r9
    2b78:	movslq 0x4(%rcx),%r10
    2b7c:	mov    $0x8,%r11d
    2b82:	vbroadcastss (%rsi),%xmm0
    2b87:	vbroadcastss 0x4(%rsi),%xmm2
    2b8d:	shl    $0x4,%r9
    2b91:	shl    $0x4,%r10
    2b95:	vfmadd132ps (%rdx,%r9,1),%xmm12,%xmm0
    2b9b:	vfmadd231ps (%rdx,%r10,1),%xmm2,%xmm12
    2ba1:	movslq (%rcx,%r11,1),%rax
    2ba5:	movslq 0x4(%rcx,%r11,1),%rdi
    2baa:	vbroadcastss (%rsi,%r11,1),%xmm3
    2bb0:	vbroadcastss 0x4(%rsi,%r11,1),%xmm4
    2bb7:	add    $0x8,%r11
    2bbb:	shl    $0x4,%rax
    2bbf:	shl    $0x4,%rdi
    2bc3:	vfmadd231ps (%rdx,%rax,1),%xmm3,%xmm0
    2bc9:	vfmadd231ps (%rdx,%rdi,1),%xmm4,%xmm12
    2bcf:	movslq (%rcx,%r11,1),%r9
    2bd3:	movslq 0x4(%rcx,%r11,1),%r10
    2bd8:	vbroadcastss (%rsi,%r11,1),%xmm5
    2bde:	vbroadcastss 0x4(%rsi,%r11,1),%xmm6
    2be5:	add    $0x8,%r11
    2be9:	shl    $0x4,%r9
    2bed:	shl    $0x4,%r10
    2bf1:	vfmadd231ps (%rdx,%r9,1),%xmm5,%xmm0
    2bf7:	vfmadd231ps (%rdx,%r10,1),%xmm6,%xmm12
    2bfd:	cmp    %r11,%r8
    2c00:	je     2cc1 <_Z2v0RK4Args+0x1a1>
    2c06:	movslq (%rcx,%r11,1),%rax
    2c0a:	movslq 0x4(%rcx,%r11,1),%rdi
    2c0f:	movslq 0x8(%rcx,%r11,1),%r9
    2c14:	movslq 0xc(%rcx,%r11,1),%r10
    2c19:	shl    $0x4,%rax
    2c1d:	shl    $0x4,%rdi
    2c21:	vbroadcastss (%rsi,%r11,1),%xmm7
    2c27:	vbroadcastss 0x4(%rsi,%r11,1),%xmm8
    2c2e:	shl    $0x4,%r9
    2c32:	vfmadd132ps (%rdx,%rax,1),%xmm0,%xmm7
    2c38:	shl    $0x4,%r10
    2c3c:	movslq 0x10(%rcx,%r11,1),%rax
    2c41:	vfmadd231ps (%rdx,%rdi,1),%xmm8,%xmm12
    2c47:	movslq 0x14(%rcx,%r11,1),%rdi
    2c4c:	vbroadcastss 0x8(%rsi,%r11,1),%xmm9
    2c53:	vbroadcastss 0xc(%rsi,%r11,1),%xmm10
    2c5a:	shl    $0x4,%rax
    2c5e:	vfmadd132ps (%rdx,%r10,1),%xmm12,%xmm10
    2c64:	movslq 0x1c(%rcx,%r11,1),%r10
    2c69:	shl    $0x4,%rdi
    2c6d:	vfmadd132ps (%rdx,%r9,1),%xmm7,%xmm9
    2c73:	movslq 0x18(%rcx,%r11,1),%r9
    2c78:	vbroadcastss 0x10(%rsi,%r11,1),%xmm1
    2c7f:	vbroadcastss 0x14(%rsi,%r11,1),%xmm11
    2c86:	shl    $0x4,%r10
    2c8a:	vbroadcastss 0x18(%rsi,%r11,1),%xmm0
    2c91:	vbroadcastss 0x1c(%rsi,%r11,1),%xmm12
    2c98:	shl    $0x4,%r9
    2c9c:	add    $0x20,%r11
    2ca0:	vfmadd132ps (%rdx,%rax,1),%xmm9,%xmm1
    2ca6:	vfmadd132ps (%rdx,%rdi,1),%xmm10,%xmm11
    2cac:	vfmadd132ps (%rdx,%r9,1),%xmm1,%xmm0
    2cb2:	vfmadd132ps (%rdx,%r10,1),%xmm11,%xmm12
    2cb8:	cmp    %r11,%r8
    2cbb:	jne    2c06 <_Z2v0RK4Args+0xe6>
    2cc1:	vaddps %xmm12,%xmm0,%xmm13
    2cc6:	vshufps $0x55,%xmm13,%xmm13,%xmm14
    2ccc:	vaddss %xmm13,%xmm14,%xmm0
    2cd1:	vunpckhps %xmm13,%xmm13,%xmm2
    2cd6:	vshufps $0xff,%xmm13,%xmm13,%xmm4
    2cdc:	vaddss %xmm2,%xmm0,%xmm3
    2ce0:	vaddss %xmm4,%xmm3,%xmm0
    2ce4:	ret    
    2ce5:	nopl   (%rax)
    2ce8:	vxorps %xmm0,%xmm0,%xmm0
    2cec:	ret    
    2ced:	nopl   (%rax)

0000000000002cf0 <_Z3v0bRK4Args>:
    2cf0:	endbr64 
    2cf4:	mov    0x20(%rdi),%eax
    2cf7:	cmp    $0x1,%eax
    2cfa:	jle    2eb8 <_Z3v0bRK4Args+0x1c8>
    2d00:	sub    $0x2,%eax
    2d03:	mov    0x10(%rdi),%rdx
    2d07:	mov    (%rdi),%rcx
    2d0a:	vxorps %xmm12,%xmm12,%xmm12
    2d0f:	shr    %eax
    2d11:	mov    0x8(%rdi),%rsi
    2d15:	xor    %r11d,%r11d
    2d18:	vmovaps %xmm12,%xmm0
    2d1c:	lea    0x1(%rax),%r8d
    2d20:	shl    $0x3,%r8
    2d24:	lea    -0x8(%r8),%rdi
    2d28:	shr    $0x3,%rdi
    2d2c:	add    $0x1,%rdi
    2d30:	and    $0x3,%edi
    2d33:	je     2dd6 <_Z3v0bRK4Args+0xe6>
    2d39:	cmp    $0x1,%rdi
    2d3d:	je     2d9f <_Z3v0bRK4Args+0xaf>
    2d3f:	cmp    $0x2,%rdi
    2d43:	je     2d71 <_Z3v0bRK4Args+0x81>
    2d45:	movslq (%rcx),%r9
    2d48:	movslq 0x4(%rcx),%r10
    2d4c:	mov    $0x8,%r11d
    2d52:	vbroadcastss (%rsi),%xmm0
    2d57:	vbroadcastss 0x4(%rsi),%xmm2
    2d5d:	shl    $0x4,%r9
    2d61:	shl    $0x4,%r10
    2d65:	vfmadd132ps (%rdx,%r9,1),%xmm12,%xmm0
    2d6b:	vfmadd231ps (%rdx,%r10,1),%xmm2,%xmm12
    2d71:	movslq (%rcx,%r11,1),%rax
    2d75:	movslq 0x4(%rcx,%r11,1),%rdi
    2d7a:	vbroadcastss (%rsi,%r11,1),%xmm3
    2d80:	vbroadcastss 0x4(%rsi,%r11,1),%xmm4
    2d87:	add    $0x8,%r11
    2d8b:	shl    $0x4,%rax
    2d8f:	shl    $0x4,%rdi
    2d93:	vfmadd231ps (%rdx,%rax,1),%xmm3,%xmm0
    2d99:	vfmadd231ps (%rdx,%rdi,1),%xmm4,%xmm12
    2d9f:	movslq (%rcx,%r11,1),%r9
    2da3:	movslq 0x4(%rcx,%r11,1),%r10
    2da8:	vbroadcastss (%rsi,%r11,1),%xmm5
    2dae:	vbroadcastss 0x4(%rsi,%r11,1),%xmm6
    2db5:	add    $0x8,%r11
    2db9:	shl    $0x4,%r9
    2dbd:	shl    $0x4,%r10
    2dc1:	vfmadd231ps (%rdx,%r9,1),%xmm5,%xmm0
    2dc7:	vfmadd231ps (%rdx,%r10,1),%xmm6,%xmm12
    2dcd:	cmp    %r11,%r8
    2dd0:	je     2e91 <_Z3v0bRK4Args+0x1a1>
    2dd6:	movslq (%rcx,%r11,1),%rax
    2dda:	movslq 0x4(%rcx,%r11,1),%rdi
    2ddf:	movslq 0x8(%rcx,%r11,1),%r9
    2de4:	movslq 0xc(%rcx,%r11,1),%r10
    2de9:	shl    $0x4,%rax
    2ded:	shl    $0x4,%rdi
    2df1:	vbroadcastss (%rsi,%r11,1),%xmm7
    2df7:	vbroadcastss 0x4(%rsi,%r11,1),%xmm8
    2dfe:	shl    $0x4,%r9
    2e02:	vfmadd132ps (%rdx,%rax,1),%xmm0,%xmm7
    2e08:	shl    $0x4,%r10
    2e0c:	movslq 0x10(%rcx,%r11,1),%rax
    2e11:	vfmadd231ps (%rdx,%rdi,1),%xmm8,%xmm12
    2e17:	movslq 0x14(%rcx,%r11,1),%rdi
    2e1c:	vbroadcastss 0x8(%rsi,%r11,1),%xmm9
    2e23:	vbroadcastss 0xc(%rsi,%r11,1),%xmm10
    2e2a:	shl    $0x4,%rax
    2e2e:	vfmadd132ps (%rdx,%r10,1),%xmm12,%xmm10
    2e34:	movslq 0x1c(%rcx,%r11,1),%r10
    2e39:	shl    $0x4,%rdi
    2e3d:	vfmadd132ps (%rdx,%r9,1),%xmm7,%xmm9
    2e43:	movslq 0x18(%rcx,%r11,1),%r9
    2e48:	vbroadcastss 0x10(%rsi,%r11,1),%xmm1
    2e4f:	vbroadcastss 0x14(%rsi,%r11,1),%xmm11
    2e56:	shl    $0x4,%r10
    2e5a:	vbroadcastss 0x18(%rsi,%r11,1),%xmm0
    2e61:	vbroadcastss 0x1c(%rsi,%r11,1),%xmm12
    2e68:	shl    $0x4,%r9
    2e6c:	add    $0x20,%r11
    2e70:	vfmadd132ps (%rdx,%rax,1),%xmm9,%xmm1
    2e76:	vfmadd132ps (%rdx,%rdi,1),%xmm10,%xmm11
    2e7c:	vfmadd132ps (%rdx,%r9,1),%xmm1,%xmm0
    2e82:	vfmadd132ps (%rdx,%r10,1),%xmm11,%xmm12
    2e88:	cmp    %r11,%r8
    2e8b:	jne    2dd6 <_Z3v0bRK4Args+0xe6>
    2e91:	vaddps %xmm12,%xmm0,%xmm13
    2e96:	vshufps $0x55,%xmm13,%xmm13,%xmm14
    2e9c:	vaddss %xmm13,%xmm14,%xmm0
    2ea1:	vunpckhps %xmm13,%xmm13,%xmm2
    2ea6:	vshufps $0xff,%xmm13,%xmm13,%xmm4
    2eac:	vaddss %xmm2,%xmm0,%xmm3
    2eb0:	vaddss %xmm4,%xmm3,%xmm0
    2eb4:	ret    
    2eb5:	nopl   (%rax)
    2eb8:	vxorps %xmm0,%xmm0,%xmm0
    2ebc:	ret    
    2ebd:	nopl   (%rax)

0000000000002ec0 <_Z2v1RK4Args>:
    2ec0:	endbr64 
    2ec4:	mov    0x20(%rdi),%eax
    2ec7:	cmp    $0x1,%eax
    2eca:	jle    3008 <_Z2v1RK4Args+0x148>
    2ed0:	sub    $0x2,%eax
    2ed3:	mov    0x10(%rdi),%rdx
    2ed7:	mov    0x8(%rdi),%rcx
    2edb:	vxorps %xmm12,%xmm12,%xmm12
    2ee0:	shr    %eax
    2ee2:	xor    %r8d,%r8d
    2ee5:	vmovaps %xmm12,%xmm0
    2ee9:	lea    0x1(%rax),%edi
    2eec:	shl    $0x3,%rdi
    2ef0:	lea    -0x8(%rdi),%rsi
    2ef4:	shr    $0x3,%rsi
    2ef8:	add    $0x1,%rsi
    2efc:	and    $0x3,%esi
    2eff:	je     2f6a <_Z2v1RK4Args+0xaa>
    2f01:	cmp    $0x1,%rsi
    2f05:	je     2f47 <_Z2v1RK4Args+0x87>
    2f07:	cmp    $0x2,%rsi
    2f0b:	je     2f29 <_Z2v1RK4Args+0x69>
    2f0d:	vbroadcastss (%rcx),%xmm0
    2f12:	vbroadcastss 0x4(%rcx),%xmm2
    2f18:	mov    $0x8,%r8d
    2f1e:	vfmadd132ps (%rdx),%xmm12,%xmm0
    2f23:	vfmadd231ps 0x10(%rdx),%xmm2,%xmm12
    2f29:	vbroadcastss (%rcx,%r8,1),%xmm3
    2f2f:	vbroadcastss 0x4(%rcx,%r8,1),%xmm4
    2f36:	vfmadd231ps (%rdx,%r8,4),%xmm3,%xmm0
    2f3c:	vfmadd231ps 0x10(%rdx,%r8,4),%xmm4,%xmm12
    2f43:	add    $0x8,%r8
    2f47:	vbroadcastss (%rcx,%r8,1),%xmm5
    2f4d:	vbroadcastss 0x4(%rcx,%r8,1),%xmm6
    2f54:	vfmadd231ps (%rdx,%r8,4),%xmm5,%xmm0
    2f5a:	vfmadd231ps 0x10(%rdx,%r8,4),%xmm6,%xmm12
    2f61:	add    $0x8,%r8
    2f65:	cmp    %r8,%rdi
    2f68:	je     2fe1 <_Z2v1RK4Args+0x121>
    2f6a:	vbroadcastss (%rcx,%r8,1),%xmm7
    2f70:	vbroadcastss 0x4(%rcx,%r8,1),%xmm8
    2f77:	vfmadd132ps (%rdx,%r8,4),%xmm0,%xmm7
    2f7d:	vfmadd231ps 0x10(%rdx,%r8,4),%xmm8,%xmm12
    2f84:	vbroadcastss 0x8(%rcx,%r8,1),%xmm9
    2f8b:	vbroadcastss 0xc(%rcx,%r8,1),%xmm10
    2f92:	vfmadd132ps 0x20(%rdx,%r8,4),%xmm7,%xmm9
    2f99:	vfmadd132ps 0x30(%rdx,%r8,4),%xmm12,%xmm10
    2fa0:	vbroadcastss 0x10(%rcx,%r8,1),%xmm1
    2fa7:	vbroadcastss 0x14(%rcx,%r8,1),%xmm11
    2fae:	vfmadd132ps 0x40(%rdx,%r8,4),%xmm9,%xmm1
    2fb5:	vfmadd132ps 0x50(%rdx,%r8,4),%xmm10,%xmm11
    2fbc:	vbroadcastss 0x18(%rcx,%r8,1),%xmm0
    2fc3:	vbroadcastss 0x1c(%rcx,%r8,1),%xmm12
    2fca:	vfmadd132ps 0x60(%rdx,%r8,4),%xmm1,%xmm0
    2fd1:	vfmadd132ps 0x70(%rdx,%r8,4),%xmm11,%xmm12
    2fd8:	add    $0x20,%r8
    2fdc:	cmp    %r8,%rdi
    2fdf:	jne    2f6a <_Z2v1RK4Args+0xaa>
    2fe1:	vaddps %xmm12,%xmm0,%xmm13
    2fe6:	vshufps $0x55,%xmm13,%xmm13,%xmm14
    2fec:	vaddss %xmm13,%xmm14,%xmm0
    2ff1:	vunpckhps %xmm13,%xmm13,%xmm2
    2ff6:	vshufps $0xff,%xmm13,%xmm13,%xmm4
    2ffc:	vaddss %xmm2,%xmm0,%xmm3
    3000:	vaddss %xmm4,%xmm3,%xmm0
    3004:	ret    
    3005:	nopl   (%rax)
    3008:	vxorps %xmm0,%xmm0,%xmm0
    300c:	ret    
    300d:	nopl   (%rax)

0000000000003010 <_Z2v2RK4Args>:
    3010:	endbr64 
    3014:	mov    0x8(%rdi),%rax
    3018:	mov    0x20(%rdi),%esi
    301b:	vbroadcastss (%rax),%xmm3
    3020:	vbroadcastss 0x4(%rax),%xmm2
    3026:	cmp    $0x1,%esi
    3029:	jle    3170 <_Z2v2RK4Args+0x160>
    302f:	mov    (%rdi),%rcx
    3032:	sub    $0x2,%esi
    3035:	vxorps %xmm0,%xmm0,%xmm0
    3039:	mov    0x10(%rdi),%rdx
    303d:	shr    %esi
    303f:	vmovaps %xmm0,%xmm1
    3043:	lea    0x8(%rcx),%r9
    3047:	lea    (%r9,%rsi,8),%rdi
    304b:	and    $0x3,%esi
    304e:	je     312b <_Z2v2RK4Args+0x11b>
    3054:	movslq (%rcx),%r8
    3057:	movslq 0x4(%rcx),%rcx
    305b:	vmovaps %xmm3,%xmm1
    305f:	shl    $0x4,%rcx
    3063:	shl    $0x4,%r8
    3067:	vfmadd132ps (%rdx,%r8,1),%xmm0,%xmm1
    306d:	vfmadd231ps (%rdx,%rcx,1),%xmm2,%xmm0
    3073:	mov    %r9,%rcx
    3076:	add    $0x8,%r9
    307a:	cmp    $0x1,%rsi
    307e:	je     312b <_Z2v2RK4Args+0x11b>
    3084:	cmp    $0x2,%rsi
    3088:	je     30ac <_Z2v2RK4Args+0x9c>
    308a:	movslq (%rcx),%r10
    308d:	movslq 0x4(%rcx),%r11
    3091:	mov    %r9,%rcx
    3094:	add    $0x8,%r9
    3098:	shl    $0x4,%r10
    309c:	shl    $0x4,%r11
    30a0:	vfmadd231ps (%rdx,%r10,1),%xmm3,%xmm1
    30a6:	vfmadd231ps (%rdx,%r11,1),%xmm2,%xmm0
    30ac:	movslq (%rcx),%rax
    30af:	movslq 0x4(%rcx),%rsi
    30b3:	mov    %r9,%rcx
    30b6:	add    $0x8,%r9
    30ba:	shl    $0x4,%rax
    30be:	shl    $0x4,%rsi
    30c2:	vfmadd231ps (%rdx,%rax,1),%xmm3,%xmm1
    30c8:	vfmadd231ps (%rdx,%rsi,1),%xmm2,%xmm0
    30ce:	jmp    312b <_Z2v2RK4Args+0x11b>
    30d0:	movslq (%r9),%r11
    30d3:	movslq 0x4(%r9),%rax
    30d7:	movslq 0x8(%r9),%rsi
    30db:	movslq 0xc(%r9),%rcx
    30df:	movslq 0x10(%r9),%r8
    30e3:	movslq 0x14(%r9),%r10
    30e7:	shl    $0x4,%r11
    30eb:	shl    $0x4,%rax
    30ef:	shl    $0x4,%rcx
    30f3:	vfmadd231ps (%rdx,%r11,1),%xmm3,%xmm1
    30f9:	shl    $0x4,%rsi
    30fd:	vfmadd231ps (%rdx,%rax,1),%xmm2,%xmm0
    3103:	vfmadd231ps (%rdx,%rsi,1),%xmm3,%xmm1
    3109:	shl    $0x4,%r8
    310d:	vfmadd231ps (%rdx,%rcx,1),%xmm2,%xmm0
    3113:	shl    $0x4,%r10
    3117:	lea    0x18(%r9),%rcx
    311b:	add    $0x20,%r9
    311f:	vfmadd231ps (%rdx,%r8,1),%xmm3,%xmm1
    3125:	vfmadd231ps (%rdx,%r10,1),%xmm2,%xmm0
    312b:	movslq (%rcx),%r8
    312e:	movslq 0x4(%rcx),%r10
    3132:	shl    $0x4,%r8
    3136:	shl    $0x4,%r10
    313a:	vfmadd231ps (%rdx,%r8,1),%xmm3,%xmm1
    3140:	vfmadd231ps (%rdx,%r10,1),%xmm2,%xmm0
    3146:	cmp    %r9,%rdi
    3149:	jne    30d0 <_Z2v2RK4Args+0xc0>
    314b:	vaddps %xmm1,%xmm0,%xmm4
    314f:	vshufps $0x55,%xmm4,%xmm4,%xmm5
    3154:	vaddss %xmm4,%xmm5,%xmm7
    3158:	vunpckhps %xmm4,%xmm4,%xmm8
    315c:	vshufps $0xff,%xmm4,%xmm4,%xmm10
    3161:	vaddss %xmm8,%xmm7,%xmm9
    3166:	vaddss %xmm10,%xmm9,%xmm0
    316b:	ret    
    316c:	nopl   0x0(%rax)
    3170:	vxorps %xmm0,%xmm0,%xmm0
    3174:	ret    
    3175:	data16 cs nopw 0x0(%rax,%rax,1)

0000000000003180 <_Z2v3RK4Args>:
    3180:	endbr64 
    3184:	mov    0x10(%rdi),%rax
    3188:	mov    0x20(%rdi),%edx
    318b:	vmovups (%rax),%xmm4
    318f:	vmovups 0x10(%rax),%xmm3
    3194:	cmp    $0x1,%edx
    3197:	jle    3330 <_Z2v3RK4Args+0x1b0>
    319d:	sub    $0x2,%edx
    31a0:	vxorps %xmm14,%xmm14,%xmm14
    31a5:	mov    (%rdi),%rsi
    31a8:	mov    0x8(%rdi),%rcx
    31ac:	shr    %edx
    31ae:	xor    %r10d,%r10d
    31b1:	vmovaps %xmm14,%xmm1
    31b5:	xor    %edi,%edi
    31b7:	lea    0x1(%rdx),%r8d
    31bb:	shl    $0x3,%r8
    31bf:	lea    -0x8(%r8),%r9
    31c3:	shr    $0x3,%r9
    31c7:	add    $0x1,%r9
    31cb:	and    $0x3,%r9d
    31cf:	je     3257 <_Z2v3RK4Args+0xd7>
    31d5:	cmp    $0x1,%r9
    31d9:	je     3228 <_Z2v3RK4Args+0xa8>
    31db:	cmp    $0x2,%r9
    31df:	je     3201 <_Z2v3RK4Args+0x81>
    31e1:	vbroadcastss (%rcx),%xmm1
    31e6:	mov    0x4(%rsi),%edi
    31e9:	mov    $0x8,%r10d
    31ef:	vbroadcastss 0x4(%rcx),%xmm2
    31f5:	add    (%rsi),%edi
    31f7:	vfmadd132ps %xmm4,%xmm14,%xmm1
    31fc:	vfmadd231ps %xmm3,%xmm2,%xmm14
    3201:	vbroadcastss (%rcx,%r10,1),%xmm5
    3207:	vbroadcastss 0x4(%rcx,%r10,1),%xmm6
    320e:	mov    0x4(%rsi,%r10,1),%r11d
    3213:	add    (%rsi,%r10,1),%r11d
    3217:	add    $0x8,%r10
    321b:	vfmadd231ps %xmm4,%xmm5,%xmm1
    3220:	vfmadd231ps %xmm3,%xmm6,%xmm14
    3225:	add    %r11d,%edi
    3228:	vbroadcastss (%rcx,%r10,1),%xmm7
    322e:	vbroadcastss 0x4(%rcx,%r10,1),%xmm8
    3235:	mov    0x4(%rsi,%r10,1),%eax
    323a:	add    (%rsi,%r10,1),%eax
    323e:	add    $0x8,%r10
    3242:	vfmadd231ps %xmm4,%xmm7,%xmm1
    3247:	vfmadd231ps %xmm3,%xmm8,%xmm14
    324c:	add    %eax,%edi
    324e:	cmp    %r10,%r8
    3251:	je     32f3 <_Z2v3RK4Args+0x173>
    3257:	vbroadcastss (%rcx,%r10,1),%xmm9
    325d:	vbroadcastss 0x4(%rcx,%r10,1),%xmm10
    3264:	vbroadcastss 0x8(%rcx,%r10,1),%xmm11
    326b:	vbroadcastss 0xc(%rcx,%r10,1),%xmm12
    3272:	vfmadd132ps %xmm4,%xmm1,%xmm9
    3277:	vfmadd231ps %xmm3,%xmm10,%xmm14
    327c:	vbroadcastss 0x10(%rcx,%r10,1),%xmm0
    3283:	vbroadcastss 0x14(%rcx,%r10,1),%xmm13
    328a:	mov    0x4(%rsi,%r10,1),%edx
    328f:	add    (%rsi,%r10,1),%edx
    3293:	mov    0xc(%rsi,%r10,1),%r9d
    3298:	add    %edi,%edx
    329a:	add    0x8(%rsi,%r10,1),%r9d
    329f:	mov    0x14(%rsi,%r10,1),%edi
    32a4:	vbroadcastss 0x18(%rcx,%r10,1),%xmm1
    32ab:	add    %r9d,%edx
    32ae:	add    0x10(%rsi,%r10,1),%edi
    32b3:	vfmadd132ps %xmm3,%xmm14,%xmm12
    32b8:	vfmadd132ps %xmm4,%xmm9,%xmm11
    32bd:	vbroadcastss 0x1c(%rcx,%r10,1),%xmm14
    32c4:	add    %edi,%edx
    32c6:	mov    0x1c(%rsi,%r10,1),%edi
    32cb:	add    0x18(%rsi,%r10,1),%edi
    32d0:	add    $0x20,%r10
    32d4:	add    %edx,%edi
    32d6:	vfmadd132ps %xmm4,%xmm11,%xmm0
    32db:	vfmadd132ps %xmm3,%xmm12,%xmm13
    32e0:	vfmadd132ps %xmm4,%xmm0,%xmm1
    32e5:	vfmadd132ps %xmm3,%xmm13,%xmm14
    32ea:	cmp    %r10,%r8
    32ed:	jne    3257 <_Z2v3RK4Args+0xd7>
    32f3:	vaddps %xmm1,%xmm14,%xmm15
    32f7:	vxorps %xmm4,%xmm4,%xmm4
    32fb:	vcvtsi2ss %edi,%xmm4,%xmm0
    32ff:	vshufps $0x55,%xmm15,%xmm15,%xmm1
    3305:	vaddss %xmm15,%xmm1,%xmm2
    330a:	vunpckhps %xmm15,%xmm15,%xmm5
    330f:	vshufps $0xff,%xmm15,%xmm15,%xmm7
    3315:	vaddss %xmm5,%xmm2,%xmm6
    3319:	vaddss %xmm7,%xmm6,%xmm8
    331d:	vfmadd132ss 0x1de2(%rip),%xmm8,%xmm0        # 5108 <_IO_stdin_used+0x108>
    3326:	ret    
    3327:	nopw   0x0(%rax,%rax,1)
    3330:	vxorps %xmm0,%xmm0,%xmm0
    3334:	ret    
    3335:	data16 cs nopw 0x0(%rax,%rax,1)

0000000000003340 <_Z2v6RK4Args>:
    3340:	endbr64 
    3344:	push   %rbx
    3345:	mov    0x20(%rdi),%r10d
    3349:	mov    (%rdi),%r8
    334c:	mov    0x8(%rdi),%r9
    3350:	mov    0x10(%rdi),%rdx
    3354:	cmp    $0x3,%r10d
    3358:	jle    35e0 <_Z2v6RK4Args+0x2a0>
    335e:	lea    -0x4(%r10),%r11d
    3362:	lea    0x10(%r8),%rsi
    3366:	vxorps %xmm1,%xmm1,%xmm1
    336a:	mov    %r8,%rax
    336d:	shr    $0x2,%r11d
    3371:	vmovaps %xmm1,%xmm2
    3375:	vmovaps %xmm1,%xmm6
    3379:	mov    %r9,%rcx
    337c:	mov    %r11d,%edi
    337f:	vmovaps %xmm1,%xmm0
    3383:	shl    $0x4,%rdi
    3387:	add    %rsi,%rdi
    338a:	mov    %rdi,%rbx
    338d:	sub    %r8,%rbx
    3390:	and    $0x10,%ebx
    3393:	je     3400 <_Z2v6RK4Args+0xc0>
    3395:	movslq (%r8),%rcx
    3398:	vbroadcastss (%r9),%xmm0
    339d:	movslq 0x4(%r8),%rax
    33a1:	movslq 0x8(%r8),%rbx
    33a5:	shl    $0x4,%rcx
    33a9:	vbroadcastss 0x4(%r9),%xmm5
    33af:	vbroadcastss 0x8(%r9),%xmm2
    33b5:	vfmadd132ps (%rdx,%rcx,1),%xmm1,%xmm0
    33bb:	movslq 0xc(%r8),%rcx
    33bf:	shl    $0x4,%rax
    33c3:	shl    $0x4,%rbx
    33c7:	vbroadcastss 0xc(%r9),%xmm3
    33cd:	vmovaps %xmm5,%xmm6
    33d1:	vfmadd132ps (%rdx,%rbx,1),%xmm1,%xmm2
    33d7:	shl    $0x4,%rcx
    33db:	vfmadd132ps (%rdx,%rax,1),%xmm1,%xmm6
    33e1:	mov    %rsi,%rax
    33e4:	vfmadd231ps (%rdx,%rcx,1),%xmm3,%xmm1
    33ea:	lea    0x10(%r9),%rcx
    33ee:	cmp    %rsi,%rdi
    33f1:	je     34b4 <_Z2v6RK4Args+0x174>
    33f7:	nopw   0x0(%rax,%rax,1)
    3400:	movslq (%rax),%rsi
    3403:	movslq 0x4(%rax),%rbx
    3407:	add    $0x20,%rax
    340b:	add    $0x20,%rcx
    340f:	vbroadcastss -0x20(%rcx),%xmm4
    3415:	vbroadcastss -0x1c(%rcx),%xmm7
    341b:	shl    $0x4,%rsi
    341f:	shl    $0x4,%rbx
    3423:	vbroadcastss -0x18(%rcx),%xmm8
    3429:	vbroadcastss -0x14(%rcx),%xmm9
    342f:	vfmadd132ps (%rdx,%rsi,1),%xmm0,%xmm4
    3435:	vfmadd132ps (%rdx,%rbx,1),%xmm6,%xmm7
    343b:	movslq -0x18(%rax),%rsi
    343f:	movslq -0x14(%rax),%rbx
    3443:	vbroadcastss -0xc(%rcx),%xmm5
    3449:	vbroadcastss -0x10(%rcx),%xmm0
    344f:	shl    $0x4,%rsi
    3453:	shl    $0x4,%rbx
    3457:	vbroadcastss -0x8(%rcx),%xmm10
    345d:	vbroadcastss -0x4(%rcx),%xmm11
    3463:	vfmadd231ps (%rdx,%rsi,1),%xmm8,%xmm2
    3469:	vfmadd231ps (%rdx,%rbx,1),%xmm9,%xmm1
    346f:	vmovaps %xmm5,%xmm6
    3473:	movslq -0x10(%rax),%rsi
    3477:	movslq -0xc(%rax),%rbx
    347b:	shl    $0x4,%rsi
    347f:	shl    $0x4,%rbx
    3483:	vfmadd132ps (%rdx,%rsi,1),%xmm4,%xmm0
    3489:	vfmadd132ps (%rdx,%rbx,1),%xmm7,%xmm6
    348f:	movslq -0x8(%rax),%rsi
    3493:	movslq -0x4(%rax),%rbx
    3497:	shl    $0x4,%rsi
    349b:	shl    $0x4,%rbx
    349f:	vfmadd231ps (%rdx,%rsi,1),%xmm10,%xmm2
    34a5:	vfmadd231ps (%rdx,%rbx,1),%xmm11,%xmm1
    34ab:	cmp    %rax,%rdi
    34ae:	jne    3400 <_Z2v6RK4Args+0xc0>
    34b4:	vaddps %xmm2,%xmm1,%xmm1
    34b8:	lea    0x4(,%r11,4),%eax
    34c0:	cmp    %eax,%r10d
    34c3:	jle    35b6 <_Z2v6RK4Args+0x276>
    34c9:	cltq   
    34cb:	movslq (%r8,%rax,4),%rdi
    34cf:	mov    %rax,%r11
    34d2:	lea    0x1(%rax),%rsi
    34d6:	vbroadcastss (%r9,%rax,4),%xmm12
    34dc:	not    %r11
    34df:	add    %r10d,%r11d
    34e2:	shl    $0x4,%rdi
    34e6:	vfmadd231ps (%rdx,%rdi,1),%xmm12,%xmm0
    34ec:	and    $0x3,%r11d
    34f0:	cmp    %esi,%r10d
    34f3:	jle    35b6 <_Z2v6RK4Args+0x276>
    34f9:	test   %r11d,%r11d
    34fc:	je     3557 <_Z2v6RK4Args+0x217>
    34fe:	cmp    $0x1,%r11d
    3502:	je     353a <_Z2v6RK4Args+0x1fa>
    3504:	cmp    $0x2,%r11d
    3508:	je     3522 <_Z2v6RK4Args+0x1e2>
    350a:	movslq (%r8,%rsi,4),%rcx
    350e:	vbroadcastss (%r9,%rsi,4),%xmm13
    3514:	lea    0x2(%rax),%rsi
    3518:	shl    $0x4,%rcx
    351c:	vfmadd231ps (%rdx,%rcx,1),%xmm13,%xmm0
    3522:	movslq (%r8,%rsi,4),%rax
    3526:	vbroadcastss (%r9,%rsi,4),%xmm14
    352c:	add    $0x1,%rsi
    3530:	shl    $0x4,%rax
    3534:	vfmadd231ps (%rdx,%rax,1),%xmm14,%xmm0
    353a:	movslq (%r8,%rsi,4),%rbx
    353e:	vbroadcastss (%r9,%rsi,4),%xmm15
    3544:	add    $0x1,%rsi
    3548:	shl    $0x4,%rbx
    354c:	vfmadd231ps (%rdx,%rbx,1),%xmm15,%xmm0
    3552:	cmp    %esi,%r10d
    3555:	jle    35b6 <_Z2v6RK4Args+0x276>
    3557:	movslq (%r8,%rsi,4),%r11
    355b:	movslq 0x4(%r8,%rsi,4),%rdi
    3560:	movslq 0x8(%r8,%rsi,4),%rcx
    3565:	vbroadcastss (%r9,%rsi,4),%xmm2
    356b:	shl    $0x4,%r11
    356f:	movslq 0xc(%r8,%rsi,4),%rax
    3574:	shl    $0x4,%rdi
    3578:	vbroadcastss 0x8(%r9,%rsi,4),%xmm3
    357f:	vfmadd132ps (%rdx,%r11,1),%xmm0,%xmm2
    3585:	shl    $0x4,%rcx
    3589:	vbroadcastss 0x4(%r9,%rsi,4),%xmm0
    3590:	vfmadd132ps (%rdx,%rdi,1),%xmm2,%xmm0
    3596:	shl    $0x4,%rax
    359a:	vfmadd132ps (%rdx,%rcx,1),%xmm0,%xmm3
    35a0:	vbroadcastss 0xc(%r9,%rsi,4),%xmm0
    35a7:	add    $0x4,%rsi
    35ab:	vfmadd132ps (%rdx,%rax,1),%xmm3,%xmm0
    35b1:	cmp    %esi,%r10d
    35b4:	jg     3557 <_Z2v6RK4Args+0x217>
    35b6:	vaddps %xmm6,%xmm0,%xmm6
    35ba:	pop    %rbx
    35bb:	vaddps %xmm1,%xmm6,%xmm4
    35bf:	vshufps $0x55,%xmm4,%xmm4,%xmm7
    35c4:	vaddss %xmm4,%xmm7,%xmm9
    35c8:	vunpckhps %xmm4,%xmm4,%xmm5
    35cc:	vshufps $0xff,%xmm4,%xmm4,%xmm11
    35d1:	vaddss %xmm5,%xmm9,%xmm10
    35d5:	vaddss %xmm11,%xmm10,%xmm0
    35da:	ret    
    35db:	nopl   0x0(%rax,%rax,1)
    35e0:	vxorps %xmm1,%xmm1,%xmm1
    35e4:	xor    %eax,%eax
    35e6:	vmovaps %xmm1,%xmm6
    35ea:	vmovaps %xmm1,%xmm0
    35ee:	jmp    34c0 <_Z2v6RK4Args+0x180>
    35f3:	data16 cs nopw 0x0(%rax,%rax,1)
    35fe:	xchg   %ax,%ax

0000000000003600 <_Z2v7RK4Args>:
    3600:	endbr64 
    3604:	mov    0x20(%rdi),%r10d
    3608:	mov    (%rdi),%r8
    360b:	mov    0x8(%rdi),%r9
    360f:	mov    0x10(%rdi),%rcx
    3613:	cmp    $0x7,%r10d
    3617:	jle    3850 <_Z2v7RK4Args+0x250>
    361d:	lea    -0x8(%r10),%r11d
    3621:	vxorps %xmm4,%xmm4,%xmm4
    3625:	mov    %r8,%rax
    3628:	mov    %r9,%rdx
    362b:	vmovaps %xmm4,%xmm7
    362f:	shr    $0x3,%r11d
    3633:	vmovaps %xmm4,%xmm3
    3637:	vmovaps %xmm4,%xmm6
    363b:	vmovaps %xmm4,%xmm5
    363f:	vmovaps %xmm4,%xmm2
    3643:	vmovaps %xmm4,%xmm0
    3647:	mov    %r11d,%esi
    364a:	shl    $0x5,%rsi
    364e:	vmovaps %xmm4,%xmm1
    3652:	lea    0x20(%r8,%rsi,1),%rdi
    3657:	nopw   0x0(%rax,%rax,1)
    3660:	movslq (%rax),%rsi
    3663:	vbroadcastss (%rdx),%xmm8
    3668:	add    $0x20,%rax
    366c:	add    $0x20,%rdx
    3670:	vbroadcastss -0x1c(%rdx),%xmm9
    3676:	vbroadcastss -0x18(%rdx),%xmm10
    367c:	shl    $0x4,%rsi
    3680:	vbroadcastss -0x14(%rdx),%xmm11
    3686:	vbroadcastss -0x10(%rdx),%xmm12
    368c:	vfmadd231ps (%rcx,%rsi,1),%xmm8,%xmm1
    3692:	movslq -0x1c(%rax),%rsi
    3696:	vbroadcastss -0xc(%rdx),%xmm13
    369c:	vbroadcastss -0x8(%rdx),%xmm14
    36a2:	shl    $0x4,%rsi
    36a6:	vbroadcastss -0x4(%rdx),%xmm15
    36ac:	vfmadd231ps (%rcx,%rsi,1),%xmm9,%xmm0
    36b2:	movslq -0x18(%rax),%rsi
    36b6:	shl    $0x4,%rsi
    36ba:	vfmadd231ps (%rcx,%rsi,1),%xmm10,%xmm2
    36c0:	movslq -0x14(%rax),%rsi
    36c4:	shl    $0x4,%rsi
    36c8:	vfmadd231ps (%rcx,%rsi,1),%xmm11,%xmm5
    36ce:	movslq -0x10(%rax),%rsi
    36d2:	shl    $0x4,%rsi
    36d6:	vfmadd231ps (%rcx,%rsi,1),%xmm12,%xmm6
    36dc:	movslq -0xc(%rax),%rsi
    36e0:	shl    $0x4,%rsi
    36e4:	vfmadd231ps (%rcx,%rsi,1),%xmm13,%xmm3
    36ea:	movslq -0x8(%rax),%rsi
    36ee:	shl    $0x4,%rsi
    36f2:	vfmadd231ps (%rcx,%rsi,1),%xmm14,%xmm7
    36f8:	movslq -0x4(%rax),%rsi
    36fc:	shl    $0x4,%rsi
    3700:	vfmadd231ps (%rcx,%rsi,1),%xmm15,%xmm4
    3706:	cmp    %rax,%rdi
    3709:	jne    3660 <_Z2v7RK4Args+0x60>
    370f:	vaddps %xmm7,%xmm4,%xmm4
    3713:	vaddps %xmm6,%xmm3,%xmm7
    3717:	vaddps %xmm5,%xmm2,%xmm2
    371b:	lea    0x8(,%r11,8),%edx
    3723:	vaddps %xmm7,%xmm4,%xmm9
    3727:	cmp    %edx,%r10d
    372a:	jle    381e <_Z2v7RK4Args+0x21e>
    3730:	movslq %edx,%rdx
    3733:	movslq (%r8,%rdx,4),%rax
    3737:	mov    %rdx,%r11
    373a:	lea    0x1(%rdx),%rsi
    373e:	vbroadcastss (%r9,%rdx,4),%xmm3
    3744:	not    %r11
    3747:	add    %r10d,%r11d
    374a:	shl    $0x4,%rax
    374e:	vfmadd231ps (%rcx,%rax,1),%xmm3,%xmm1
    3754:	and    $0x3,%r11d
    3758:	cmp    %esi,%r10d
    375b:	jle    381e <_Z2v7RK4Args+0x21e>
    3761:	test   %r11d,%r11d
    3764:	je     37bf <_Z2v7RK4Args+0x1bf>
    3766:	cmp    $0x1,%r11d
    376a:	je     37a2 <_Z2v7RK4Args+0x1a2>
    376c:	cmp    $0x2,%r11d
    3770:	je     378a <_Z2v7RK4Args+0x18a>
    3772:	movslq (%r8,%rsi,4),%rdi
    3776:	vbroadcastss (%r9,%rsi,4),%xmm6
    377c:	lea    0x2(%rdx),%rsi
    3780:	shl    $0x4,%rdi
    3784:	vfmadd231ps (%rcx,%rdi,1),%xmm6,%xmm1
    378a:	movslq (%r8,%rsi,4),%rdx
    378e:	vbroadcastss (%r9,%rsi,4),%xmm5
    3794:	add    $0x1,%rsi
    3798:	shl    $0x4,%rdx
    379c:	vfmadd231ps (%rcx,%rdx,1),%xmm5,%xmm1
    37a2:	movslq (%r8,%rsi,4),%r11
    37a6:	vbroadcastss (%r9,%rsi,4),%xmm8
    37ac:	add    $0x1,%rsi
    37b0:	shl    $0x4,%r11
    37b4:	vfmadd231ps (%rcx,%r11,1),%xmm8,%xmm1
    37ba:	cmp    %esi,%r10d
    37bd:	jle    381e <_Z2v7RK4Args+0x21e>
    37bf:	movslq (%r8,%rsi,4),%rax
    37c3:	movslq 0x4(%r8,%rsi,4),%rdi
    37c8:	movslq 0x8(%r8,%rsi,4),%rdx
    37cd:	vbroadcastss (%r9,%rsi,4),%xmm10
    37d3:	shl    $0x4,%rax
    37d7:	movslq 0xc(%r8,%rsi,4),%r11
    37dc:	shl    $0x4,%rdi
    37e0:	vbroadcastss 0x8(%r9,%rsi,4),%xmm11
    37e7:	vfmadd132ps (%rcx,%rax,1),%xmm1,%xmm10
    37ed:	shl    $0x4,%rdx
    37f1:	vbroadcastss 0x4(%r9,%rsi,4),%xmm1
    37f8:	vfmadd132ps (%rcx,%rdi,1),%xmm10,%xmm1
    37fe:	shl    $0x4,%r11
    3802:	vfmadd132ps (%rcx,%rdx,1),%xmm1,%xmm11
    3808:	vbroadcastss 0xc(%r9,%rsi,4),%xmm1
    380f:	add    $0x4,%rsi
    3813:	vfmadd132ps (%rcx,%r11,1),%xmm11,%xmm1
    3819:	cmp    %esi,%r10d
    381c:	jg     37bf <_Z2v7RK4Args+0x1bf>
    381e:	vaddps %xmm0,%xmm1,%xmm12
    3822:	vaddps %xmm2,%xmm12,%xmm13
    3826:	vaddps %xmm9,%xmm13,%xmm14
    382b:	vshufps $0x55,%xmm14,%xmm14,%xmm15
    3831:	vaddss %xmm14,%xmm15,%xmm7
    3836:	vunpckhps %xmm14,%xmm14,%xmm3
    383b:	vshufps $0xff,%xmm14,%xmm14,%xmm5
    3841:	vaddss %xmm3,%xmm7,%xmm6
    3845:	vaddss %xmm5,%xmm6,%xmm0
    3849:	ret    
    384a:	nopw   0x0(%rax,%rax,1)
    3850:	vxorps %xmm2,%xmm2,%xmm2
    3854:	xor    %edx,%edx
    3856:	vmovaps %xmm2,%xmm9
    385a:	vmovaps %xmm2,%xmm0
    385e:	vmovaps %xmm2,%xmm1
    3862:	jmp    3727 <_Z2v7RK4Args+0x127>
    3867:	nopw   0x0(%rax,%rax,1)

0000000000003870 <_Z2v8RK4Args>:
    3870:	endbr64 
    3874:	mov    0x8(%rdi),%r8
    3878:	mov    0x10(%rdi),%rsi
    387c:	mov    0x20(%rdi),%edi
    387f:	cmp    $0x3,%edi
    3882:	jle    3b70 <_Z2v8RK4Args+0x300>
    3888:	lea    -0x4(%rdi),%r9d
    388c:	lea    0x40(%rsi),%r10
    3890:	vxorps %xmm1,%xmm1,%xmm1
    3894:	mov    %rsi,%rax
    3897:	shr    $0x2,%r9d
    389b:	vmovaps %xmm1,%xmm2
    389f:	vmovaps %xmm1,%xmm6
    38a3:	mov    %r8,%rdx
    38a6:	mov    %r9d,%ecx
    38a9:	vmovaps %xmm1,%xmm0
    38ad:	shl    $0x6,%rcx
    38b1:	add    %r10,%rcx
    38b4:	mov    %rcx,%r11
    38b7:	sub    %rsi,%r11
    38ba:	and    $0x40,%r11d
    38be:	je     3900 <_Z2v8RK4Args+0x90>
    38c0:	vbroadcastss 0x4(%r8),%xmm5
    38c6:	vbroadcastss (%r8),%xmm0
    38cb:	mov    %r10,%rax
    38ce:	lea    0x10(%r8),%rdx
    38d2:	vfmadd132ps (%rsi),%xmm1,%xmm0
    38d7:	vbroadcastss 0x8(%r8),%xmm2
    38dd:	vbroadcastss 0xc(%r8),%xmm3
    38e3:	vfmadd132ps 0x20(%rsi),%xmm1,%xmm2
    38e9:	vmovaps %xmm5,%xmm6
    38ed:	vfmadd132ps 0x10(%rsi),%xmm1,%xmm6
    38f3:	vfmadd231ps 0x30(%rsi),%xmm3,%xmm1
    38f9:	cmp    %r10,%rcx
    38fc:	je     3970 <_Z2v8RK4Args+0x100>
    38fe:	xchg   %ax,%ax
    3900:	vbroadcastss 0x14(%rdx),%xmm5
    3906:	vbroadcastss (%rdx),%xmm4
    390b:	sub    $0xffffffffffffff80,%rax
    390f:	add    $0x20,%rdx
    3913:	vfmadd132ps -0x80(%rax),%xmm0,%xmm4
    3919:	vbroadcastss -0x1c(%rdx),%xmm7
    391f:	vbroadcastss -0x18(%rdx),%xmm8
    3925:	vfmadd132ps -0x70(%rax),%xmm6,%xmm7
    392b:	vmovaps %xmm5,%xmm6
    392f:	vfmadd231ps -0x60(%rax),%xmm8,%xmm2
    3935:	vbroadcastss -0x14(%rdx),%xmm9
    393b:	vfmadd231ps -0x50(%rax),%xmm9,%xmm1
    3941:	vbroadcastss -0x10(%rdx),%xmm0
    3947:	vbroadcastss -0x8(%rdx),%xmm10
    394d:	vbroadcastss -0x4(%rdx),%xmm11
    3953:	vfmadd132ps -0x40(%rax),%xmm4,%xmm0
    3959:	vfmadd132ps -0x30(%rax),%xmm7,%xmm6
    395f:	vfmadd231ps -0x20(%rax),%xmm10,%xmm2
    3965:	vfmadd231ps -0x10(%rax),%xmm11,%xmm1
    396b:	cmp    %rax,%rcx
    396e:	jne    3900 <_Z2v8RK4Args+0x90>
    3970:	vaddps %xmm2,%xmm1,%xmm1
    3974:	lea    0x4(,%r9,4),%edx
    397c:	cmp    %edx,%edi
    397e:	jle    3b43 <_Z2v8RK4Args+0x2d3>
    3984:	movslq %edx,%r9
    3987:	mov    %r9,%r10
    398a:	mov    %r9,%rcx
    398d:	lea    0x1(%r9),%r11
    3991:	vbroadcastss (%r8,%r9,4),%xmm12
    3997:	not    %r10
    399a:	shl    $0x4,%rcx
    399e:	add    %edi,%r10d
    39a1:	vfmadd231ps (%rsi,%rcx,1),%xmm12,%xmm0
    39a7:	and    $0x7,%r10d
    39ab:	cmp    %r11d,%edi
    39ae:	jle    3b43 <_Z2v8RK4Args+0x2d3>
    39b4:	test   %r10d,%r10d
    39b7:	je     3a90 <_Z2v8RK4Args+0x220>
    39bd:	cmp    $0x1,%r10d
    39c1:	je     3a70 <_Z2v8RK4Args+0x200>
    39c7:	cmp    $0x2,%r10d
    39cb:	je     3a59 <_Z2v8RK4Args+0x1e9>
    39d1:	cmp    $0x3,%r10d
    39d5:	je     3a42 <_Z2v8RK4Args+0x1d2>
    39d7:	cmp    $0x4,%r10d
    39db:	je     3a2b <_Z2v8RK4Args+0x1bb>
    39dd:	cmp    $0x5,%r10d
    39e1:	je     3a14 <_Z2v8RK4Args+0x1a4>
    39e3:	cmp    $0x6,%r10d
    39e7:	je     39fd <_Z2v8RK4Args+0x18d>
    39e9:	vbroadcastss (%r8,%r11,4),%xmm13
    39ef:	shl    $0x4,%r11
    39f3:	vfmadd231ps (%rsi,%r11,1),%xmm13,%xmm0
    39f9:	lea    0x2(%r9),%r11
    39fd:	mov    %r11,%rax
    3a00:	vbroadcastss (%r8,%r11,4),%xmm14
    3a06:	add    $0x1,%r11
    3a0a:	shl    $0x4,%rax
    3a0e:	vfmadd231ps (%rsi,%rax,1),%xmm14,%xmm0
    3a14:	mov    %r11,%rdx
    3a17:	vbroadcastss (%r8,%r11,4),%xmm15
    3a1d:	add    $0x1,%r11
    3a21:	shl    $0x4,%rdx
    3a25:	vfmadd231ps (%rsi,%rdx,1),%xmm15,%xmm0
    3a2b:	mov    %r11,%r9
    3a2e:	vbroadcastss (%r8,%r11,4),%xmm2
    3a34:	add    $0x1,%r11
    3a38:	shl    $0x4,%r9
    3a3c:	vfmadd231ps (%rsi,%r9,1),%xmm2,%xmm0
    3a42:	mov    %r11,%r10
    3a45:	vbroadcastss (%r8,%r11,4),%xmm3
    3a4b:	add    $0x1,%r11
    3a4f:	shl    $0x4,%r10
    3a53:	vfmadd231ps (%rsi,%r10,1),%xmm3,%xmm0
    3a59:	mov    %r11,%rcx
    3a5c:	vbroadcastss (%r8,%r11,4),%xmm4
    3a62:	add    $0x1,%r11
    3a66:	shl    $0x4,%rcx
    3a6a:	vfmadd231ps (%rsi,%rcx,1),%xmm4,%xmm0
    3a70:	mov    %r11,%rax
    3a73:	vbroadcastss (%r8,%r11,4),%xmm7
    3a79:	add    $0x1,%r11
    3a7d:	shl    $0x4,%rax
    3a81:	vfmadd231ps (%rsi,%rax,1),%xmm7,%xmm0
    3a87:	cmp    %r11d,%edi
    3a8a:	jle    3b43 <_Z2v8RK4Args+0x2d3>
    3a90:	mov    %r11,%rdx
    3a93:	mov    %r11,%r9
    3a96:	vbroadcastss (%r8,%r11,4),%xmm8
    3a9c:	mov    %r11,%r10
    3a9f:	shl    $0x4,%rdx
    3aa3:	shl    $0x4,%r9
    3aa7:	mov    %r11,%rcx
    3aaa:	mov    %r11,%rax
    3aad:	shl    $0x4,%r10
    3ab1:	vfmadd132ps (%rsi,%rdx,1),%xmm0,%xmm8
    3ab7:	shl    $0x4,%rcx
    3abb:	vbroadcastss 0x4(%r8,%r11,4),%xmm0
    3ac2:	vfmadd132ps 0x10(%rsi,%r9,1),%xmm8,%xmm0
    3ac9:	vbroadcastss 0x8(%r8,%r11,4),%xmm9
    3ad0:	mov    %r11,%rdx
    3ad3:	mov    %r11,%r9
    3ad6:	vbroadcastss 0xc(%r8,%r11,4),%xmm5
    3add:	shl    $0x4,%rax
    3ae1:	shl    $0x4,%rdx
    3ae5:	vfmadd132ps 0x20(%rsi,%r10,1),%xmm0,%xmm9
    3aec:	vfmadd132ps 0x30(%rsi,%rcx,1),%xmm9,%xmm5
    3af3:	vbroadcastss 0x10(%r8,%r11,4),%xmm10
    3afa:	mov    %r11,%r10
    3afd:	shl    $0x4,%r9
    3b01:	vfmadd132ps 0x40(%rsi,%rax,1),%xmm5,%xmm10
    3b08:	vbroadcastss 0x14(%r8,%r11,4),%xmm11
    3b0f:	shl    $0x4,%r10
    3b13:	vfmadd132ps 0x50(%rsi,%rdx,1),%xmm10,%xmm11
    3b1a:	vbroadcastss 0x18(%r8,%r11,4),%xmm12
    3b21:	vbroadcastss 0x1c(%r8,%r11,4),%xmm0
    3b28:	vfmadd132ps 0x60(%rsi,%r9,1),%xmm11,%xmm12
    3b2f:	add    $0x8,%r11
    3b33:	vfmadd132ps 0x70(%rsi,%r10,1),%xmm12,%xmm0
    3b3a:	cmp    %r11d,%edi
    3b3d:	jg     3a90 <_Z2v8RK4Args+0x220>
    3b43:	vaddps %xmm6,%xmm0,%xmm6
    3b47:	vaddps %xmm1,%xmm6,%xmm13
    3b4b:	vshufps $0x55,%xmm13,%xmm13,%xmm14
    3b51:	vaddss %xmm13,%xmm14,%xmm3
    3b56:	vunpckhps %xmm13,%xmm13,%xmm2
    3b5b:	vshufps $0xff,%xmm13,%xmm13,%xmm7
    3b61:	vaddss %xmm2,%xmm3,%xmm4
    3b65:	vaddss %xmm7,%xmm4,%xmm0
    3b69:	ret    
    3b6a:	nopw   0x0(%rax,%rax,1)
    3b70:	vxorps %xmm1,%xmm1,%xmm1
    3b74:	xor    %edx,%edx
    3b76:	vmovaps %xmm1,%xmm6
    3b7a:	vmovaps %xmm1,%xmm0
    3b7e:	jmp    397c <_Z2v8RK4Args+0x10c>
    3b83:	data16 cs nopw 0x0(%rax,%rax,1)
    3b8e:	xchg   %ax,%ax

0000000000003b90 <_Z2v9RK4Args>:
    3b90:	endbr64 
    3b94:	push   %rbx
    3b95:	mov    0x10(%rdi),%rax
    3b99:	mov    0x20(%rdi),%r10d
    3b9d:	mov    (%rdi),%r8
    3ba0:	mov    0x8(%rdi),%r9
    3ba4:	vmovups (%rax),%xmm6
    3ba8:	vmovups 0x10(%rax),%xmm9
    3bad:	vmovups 0x20(%rax),%xmm8
    3bb2:	vmovups 0x30(%rax),%xmm7
    3bb7:	cmp    $0x3,%r10d
    3bbb:	jle    3ea0 <_Z2v9RK4Args+0x310>
    3bc1:	lea    -0x4(%r10),%r11d
    3bc5:	lea    0x10(%r8),%rdx
    3bc9:	vxorps %xmm13,%xmm13,%xmm13
    3bce:	xor    %eax,%eax
    3bd0:	shr    $0x2,%r11d
    3bd4:	mov    %r8,%rsi
    3bd7:	vmovaps %xmm13,%xmm1
    3bdb:	mov    %r9,%rcx
    3bde:	mov    %r11d,%edi
    3be1:	vmovaps %xmm13,%xmm10
    3be6:	vmovaps %xmm13,%xmm0
    3bea:	shl    $0x4,%rdi
    3bee:	add    %rdx,%rdi
    3bf1:	mov    %rdi,%rbx
    3bf4:	sub    %r8,%rbx
    3bf7:	and    $0x10,%ebx
    3bfa:	je     3c50 <_Z2v9RK4Args+0xc0>
    3bfc:	vbroadcastss (%r9),%xmm0
    3c01:	mov    0x4(%r8),%eax
    3c05:	mov    %rdx,%rsi
    3c08:	lea    0x10(%r9),%rcx
    3c0c:	vbroadcastss 0x4(%r9),%xmm10
    3c12:	vbroadcastss 0x8(%r9),%xmm1
    3c18:	vbroadcastss 0xc(%r9),%xmm3
    3c1e:	vfmadd132ps %xmm6,%xmm13,%xmm0
    3c23:	add    (%r8),%eax
    3c26:	vfmadd132ps %xmm9,%xmm13,%xmm10
    3c2b:	vfmadd132ps %xmm8,%xmm13,%xmm1
    3c30:	add    0x8(%r8),%eax
    3c34:	add    0xc(%r8),%eax
    3c38:	vfmadd231ps %xmm7,%xmm3,%xmm13
    3c3d:	cmp    %rdx,%rdi
    3c40:	je     3cd0 <_Z2v9RK4Args+0x140>
    3c46:	cs nopw 0x0(%rax,%rax,1)
    3c50:	vbroadcastss (%rcx),%xmm5
    3c55:	mov    0x4(%rsi),%edx
    3c58:	add    $0x20,%rsi
    3c5c:	add    $0x20,%rcx
    3c60:	vbroadcastss -0x1c(%rcx),%xmm4
    3c66:	vbroadcastss -0x18(%rcx),%xmm11
    3c6c:	vbroadcastss -0x14(%rcx),%xmm12
    3c72:	vfmadd132ps %xmm6,%xmm0,%xmm5
    3c77:	add    -0x20(%rsi),%edx
    3c7a:	vfmadd132ps %xmm9,%xmm10,%xmm4
    3c7f:	vfmadd231ps %xmm8,%xmm11,%xmm1
    3c84:	add    -0x18(%rsi),%edx
    3c87:	vbroadcastss -0x10(%rcx),%xmm0
    3c8d:	vfmadd132ps %xmm7,%xmm13,%xmm12
    3c92:	add    -0x14(%rsi),%edx
    3c95:	vbroadcastss -0xc(%rcx),%xmm10
    3c9b:	vbroadcastss -0x8(%rcx),%xmm2
    3ca1:	vbroadcastss -0x4(%rcx),%xmm13
    3ca7:	add    %eax,%edx
    3ca9:	mov    -0xc(%rsi),%eax
    3cac:	add    -0x10(%rsi),%eax
    3caf:	add    -0x8(%rsi),%eax
    3cb2:	vfmadd132ps %xmm6,%xmm5,%xmm0
    3cb7:	add    -0x4(%rsi),%eax
    3cba:	vfmadd132ps %xmm9,%xmm4,%xmm10
    3cbf:	vfmadd231ps %xmm8,%xmm2,%xmm1
    3cc4:	add    %edx,%eax
    3cc6:	vfmadd132ps %xmm7,%xmm12,%xmm13
    3ccb:	cmp    %rsi,%rdi
    3cce:	jne    3c50 <_Z2v9RK4Args+0xc0>
    3cd0:	vaddps %xmm1,%xmm13,%xmm1
    3cd4:	lea    0x4(,%r11,4),%esi
    3cdc:	cmp    %esi,%r10d
    3cdf:	jle    3e5b <_Z2v9RK4Args+0x2cb>
    3ce5:	movslq %esi,%r11
    3ce8:	mov    %r11,%rdi
    3ceb:	vbroadcastss (%r9,%r11,4),%xmm14
    3cf1:	lea    0x1(%r11),%rbx
    3cf5:	add    (%r8,%r11,4),%eax
    3cf9:	not    %rdi
    3cfc:	add    %r10d,%edi
    3cff:	vfmadd231ps %xmm6,%xmm14,%xmm0
    3d04:	and    $0x7,%edi
    3d07:	cmp    %ebx,%r10d
    3d0a:	jle    3e5b <_Z2v9RK4Args+0x2cb>
    3d10:	test   %edi,%edi
    3d12:	je     3dc8 <_Z2v9RK4Args+0x238>
    3d18:	cmp    $0x1,%edi
    3d1b:	je     3dac <_Z2v9RK4Args+0x21c>
    3d21:	cmp    $0x2,%edi
    3d24:	je     3d99 <_Z2v9RK4Args+0x209>
    3d26:	cmp    $0x3,%edi
    3d29:	je     3d86 <_Z2v9RK4Args+0x1f6>
    3d2b:	cmp    $0x4,%edi
    3d2e:	je     3d73 <_Z2v9RK4Args+0x1e3>
    3d30:	cmp    $0x5,%edi
    3d33:	je     3d60 <_Z2v9RK4Args+0x1d0>
    3d35:	cmp    $0x6,%edi
    3d38:	je     3d4d <_Z2v9RK4Args+0x1bd>
    3d3a:	vbroadcastss (%r9,%rbx,4),%xmm15
    3d40:	add    (%r8,%rbx,4),%eax
    3d44:	lea    0x2(%r11),%rbx
    3d48:	vfmadd231ps %xmm6,%xmm15,%xmm0
    3d4d:	vbroadcastss (%r9,%rbx,4),%xmm9
    3d53:	add    (%r8,%rbx,4),%eax
    3d57:	add    $0x1,%rbx
    3d5b:	vfmadd231ps %xmm6,%xmm9,%xmm0
    3d60:	vbroadcastss (%r9,%rbx,4),%xmm8
    3d66:	add    (%r8,%rbx,4),%eax
    3d6a:	add    $0x1,%rbx
    3d6e:	vfmadd231ps %xmm6,%xmm8,%xmm0
    3d73:	vbroadcastss (%r9,%rbx,4),%xmm7
    3d79:	add    (%r8,%rbx,4),%eax
    3d7d:	add    $0x1,%rbx
    3d81:	vfmadd231ps %xmm6,%xmm7,%xmm0
    3d86:	vbroadcastss (%r9,%rbx,4),%xmm3
    3d8c:	add    (%r8,%rbx,4),%eax
    3d90:	add    $0x1,%rbx
    3d94:	vfmadd231ps %xmm6,%xmm3,%xmm0
    3d99:	vbroadcastss (%r9,%rbx,4),%xmm5
    3d9f:	add    (%r8,%rbx,4),%eax
    3da3:	add    $0x1,%rbx
    3da7:	vfmadd231ps %xmm6,%xmm5,%xmm0
    3dac:	vbroadcastss (%r9,%rbx,4),%xmm4
    3db2:	add    (%r8,%rbx,4),%eax
    3db6:	add    $0x1,%rbx
    3dba:	vfmadd231ps %xmm6,%xmm4,%xmm0
    3dbf:	cmp    %ebx,%r10d
    3dc2:	jle    3e5b <_Z2v9RK4Args+0x2cb>
    3dc8:	vbroadcastss (%r9,%rbx,4),%xmm11
    3dce:	vbroadcastss 0x8(%r9,%rbx,4),%xmm12
    3dd5:	vbroadcastss 0xc(%r9,%rbx,4),%xmm13
    3ddc:	vbroadcastss 0x10(%r9,%rbx,4),%xmm2
    3de3:	vfmadd132ps %xmm6,%xmm0,%xmm11
    3de8:	vbroadcastss 0x4(%r9,%rbx,4),%xmm0
    3def:	vbroadcastss 0x14(%r9,%rbx,4),%xmm14
    3df6:	vbroadcastss 0x18(%r9,%rbx,4),%xmm15
    3dfd:	add    (%r8,%rbx,4),%eax
    3e01:	add    0x4(%r8,%rbx,4),%eax
    3e06:	add    0x8(%r8,%rbx,4),%eax
    3e0b:	add    0xc(%r8,%rbx,4),%eax
    3e10:	add    0x10(%r8,%rbx,4),%eax
    3e15:	add    0x14(%r8,%rbx,4),%eax
    3e1a:	add    0x18(%r8,%rbx,4),%eax
    3e1f:	vfmadd132ps %xmm6,%xmm11,%xmm0
    3e24:	add    0x1c(%r8,%rbx,4),%eax
    3e29:	vfmadd132ps %xmm6,%xmm0,%xmm12
    3e2e:	vbroadcastss 0x1c(%r9,%rbx,4),%xmm0
    3e35:	add    $0x8,%rbx
    3e39:	vfmadd132ps %xmm6,%xmm12,%xmm13
    3e3e:	vfmadd132ps %xmm6,%xmm13,%xmm2
    3e43:	vfmadd132ps %xmm6,%xmm2,%xmm14
    3e48:	vfmadd132ps %xmm6,%xmm14,%xmm15
    3e4d:	vfmadd132ps %xmm6,%xmm15,%xmm0
    3e52:	cmp    %ebx,%r10d
    3e55:	jg     3dc8 <_Z2v9RK4Args+0x238>
    3e5b:	vaddps %xmm10,%xmm0,%xmm6
    3e60:	vxorps %xmm9,%xmm9,%xmm9
    3e65:	pop    %rbx
    3e66:	vcvtsi2ss %eax,%xmm9,%xmm0
    3e6a:	vaddps %xmm1,%xmm6,%xmm10
    3e6e:	vshufps $0x55,%xmm10,%xmm10,%xmm8
    3e74:	vaddss %xmm10,%xmm8,%xmm5
    3e79:	vunpckhps %xmm10,%xmm10,%xmm3
    3e7e:	vshufps $0xff,%xmm10,%xmm10,%xmm1
    3e84:	vaddss %xmm3,%xmm5,%xmm4
    3e88:	vaddss %xmm1,%xmm4,%xmm11
    3e8c:	vfmadd132ss 0x1273(%rip),%xmm11,%xmm0        # 5108 <_IO_stdin_used+0x108>
    3e95:	ret    
    3e96:	cs nopw 0x0(%rax,%rax,1)
    3ea0:	vxorps %xmm1,%xmm1,%xmm1
    3ea4:	xor    %esi,%esi
    3ea6:	xor    %eax,%eax
    3ea8:	vmovaps %xmm1,%xmm10
    3eac:	vmovaps %xmm1,%xmm0
    3eb0:	jmp    3cdc <_Z2v9RK4Args+0x14c>
    3eb5:	data16 cs nopw 0x0(%rax,%rax,1)

0000000000003ec0 <_Z2v4RK4Args>:
    3ec0:	endbr64 
    3ec4:	mov    0x20(%rdi),%esi
    3ec7:	cmp    $0x3,%esi
    3eca:	jle    40b8 <_Z2v4RK4Args+0x1f8>
    3ed0:	mov    (%rdi),%r8
    3ed3:	mov    0x10(%rdi),%rcx
    3ed7:	vxorps %xmm3,%xmm3,%xmm3
    3edb:	mov    0x8(%rdi),%rdx
    3edf:	mov    0x18(%rdi),%rax
    3ee3:	lea    -0x4(%rsi),%edi
    3ee6:	vmovaps %xmm3,%xmm10
    3eea:	lea    0x10(%r8),%r10
    3eee:	vmovaps %xmm3,%xmm9
    3ef2:	shr    $0x2,%edi
    3ef5:	vmovaps %xmm3,%xmm8
    3ef9:	shl    $0x4,%rdi
    3efd:	vmovaps %xmm3,%xmm15
    3f01:	vmovaps %xmm3,%xmm11
    3f05:	lea    (%rdi,%r10,1),%r9
    3f09:	and    $0x10,%edi
    3f0c:	je     4003 <_Z2v4RK4Args+0x143>
    3f12:	movslq (%r8),%r11
    3f15:	movslq 0x4(%r8),%rsi
    3f19:	add    $0x10,%rdx
    3f1d:	add    $0x40,%rax
    3f21:	movslq 0x8(%r8),%rdi
    3f25:	movslq 0xc(%r8),%r8
    3f29:	shl    $0x4,%r11
    3f2d:	shl    $0x4,%rsi
    3f31:	vbroadcastss -0x10(%rdx),%xmm2
    3f37:	vbroadcastss -0xc(%rdx),%xmm1
    3f3d:	vfmadd132ps (%rcx,%r11,1),%xmm15,%xmm2
    3f43:	shl    $0x4,%r8
    3f47:	vfmadd231ps (%rcx,%rsi,1),%xmm1,%xmm15
    3f4d:	shl    $0x4,%rdi
    3f51:	vbroadcastss -0x8(%rdx),%xmm11
    3f57:	vbroadcastss -0x4(%rdx),%xmm12
    3f5d:	vmovups -0x40(%rax),%xmm8
    3f62:	vfmadd231ps (%rcx,%r8,1),%xmm12,%xmm15
    3f68:	mov    %r10,%r8
    3f6b:	add    $0x10,%r10
    3f6f:	vmovups -0x30(%rax),%xmm9
    3f74:	vmovups -0x20(%rax),%xmm10
    3f79:	vmovups -0x10(%rax),%xmm3
    3f7e:	vfmadd132ps (%rcx,%rdi,1),%xmm2,%xmm11
    3f84:	jmp    4003 <_Z2v4RK4Args+0x143>
    3f86:	cs nopw 0x0(%rax,%rax,1)
    3f90:	movslq (%r10),%r11
    3f93:	movslq 0x8(%r10),%rdi
    3f97:	add    $0x20,%rdx
    3f9b:	sub    $0xffffffffffffff80,%rax
    3f9f:	vorps  -0x10(%rax),%xmm7,%xmm3
    3fa4:	vbroadcastss -0x10(%rdx),%xmm7
    3faa:	shl    $0x4,%r11
    3fae:	shl    $0x4,%rdi
    3fb2:	movslq 0x4(%r10),%rsi
    3fb6:	movslq 0xc(%r10),%r8
    3fba:	vfmadd132ps (%rcx,%r11,1),%xmm15,%xmm7
    3fc0:	vbroadcastss -0x8(%rdx),%xmm11
    3fc6:	vfmadd132ps (%rcx,%rdi,1),%xmm7,%xmm11
    3fcc:	shl    $0x4,%rsi
    3fd0:	vorps  -0x20(%rax),%xmm4,%xmm10
    3fd5:	shl    $0x4,%r8
    3fd9:	vorps  -0x40(%rax),%xmm6,%xmm8
    3fde:	vbroadcastss -0xc(%rdx),%xmm4
    3fe4:	vorps  -0x30(%rax),%xmm5,%xmm9
    3fe9:	vfmadd132ps (%rcx,%rsi,1),%xmm0,%xmm4
    3fef:	vbroadcastss -0x4(%rdx),%xmm15
    3ff5:	vfmadd132ps (%rcx,%r8,1),%xmm4,%xmm15
    3ffb:	lea    0x10(%r10),%r8
    3fff:	add    $0x20,%r10
    4003:	movslq (%r8),%r11
    4006:	movslq 0x4(%r8),%rsi
    400a:	movslq 0x8(%r8),%rdi
    400e:	movslq 0xc(%r8),%r8
    4012:	shl    $0x4,%r11
    4016:	shl    $0x4,%rsi
    401a:	vorps  (%rax),%xmm8,%xmm6
    401e:	vbroadcastss (%rdx),%xmm13
    4023:	vbroadcastss 0x4(%rdx),%xmm14
    4029:	shl    $0x4,%rdi
    402d:	vfmadd132ps (%rcx,%rsi,1),%xmm15,%xmm14
    4033:	shl    $0x4,%r8
    4037:	vfmadd132ps (%rcx,%r11,1),%xmm11,%xmm13
    403d:	vbroadcastss 0x8(%rdx),%xmm15
    4043:	vbroadcastss 0xc(%rdx),%xmm0
    4049:	vfmadd132ps (%rcx,%rdi,1),%xmm13,%xmm15
    404f:	vfmadd132ps (%rcx,%r8,1),%xmm14,%xmm0
    4055:	vorps  0x10(%rax),%xmm9,%xmm5
    405a:	vorps  0x20(%rax),%xmm10,%xmm4
    405f:	vorps  0x30(%rax),%xmm3,%xmm7
    4064:	cmp    %r10,%r9
    4067:	jne    3f90 <_Z2v4RK4Args+0xd0>
    406d:	vaddps %xmm15,%xmm0,%xmm8
    4072:	vshufps $0x55,%xmm8,%xmm8,%xmm10
    4078:	vaddss %xmm10,%xmm8,%xmm3
    407d:	vunpckhps %xmm8,%xmm8,%xmm2
    4082:	vshufps $0xff,%xmm8,%xmm8,%xmm11
    4088:	vaddss %xmm2,%xmm3,%xmm1
    408c:	vaddss %xmm11,%xmm1,%xmm0
    4091:	vmovaps %xmm6,0x2fa7(%rip)        # 7040 <_ZL6g_sink>
    4099:	vmovaps %xmm5,0x2faf(%rip)        # 7050 <_ZL6g_sink+0x10>
    40a1:	vmovaps %xmm4,0x2fb7(%rip)        # 7060 <_ZL6g_sink+0x20>
    40a9:	vmovaps %xmm7,0x2fbf(%rip)        # 7070 <_ZL6g_sink+0x30>
    40b1:	ret    
    40b2:	nopw   0x0(%rax,%rax,1)
    40b8:	vxorps %xmm7,%xmm7,%xmm7
    40bc:	vxorps %xmm0,%xmm0,%xmm0
    40c0:	vmovaps %xmm7,%xmm4
    40c4:	vmovaps %xmm7,%xmm5
    40c8:	vmovaps %xmm7,%xmm6
    40cc:	jmp    4091 <_Z2v4RK4Args+0x1d1>
    40ce:	xchg   %ax,%ax

00000000000040d0 <_Z3v4bRK4Args>:
    40d0:	endbr64 
    40d4:	mov    0x20(%rdi),%ecx
    40d7:	cmp    $0x3,%ecx
    40da:	jle    42a8 <_Z3v4bRK4Args+0x1d8>
    40e0:	mov    (%rdi),%rsi
    40e3:	mov    0x10(%rdi),%rdx
    40e7:	vxorps %xmm13,%xmm13,%xmm13
    40ec:	mov    0x8(%rdi),%rax
    40f0:	lea    -0x4(%rcx),%edi
    40f3:	vmovaps %xmm13,%xmm12
    40f8:	vmovaps %xmm13,%xmm11
    40fd:	lea    0x10(%rsi),%r9
    4101:	vmovaps %xmm13,%xmm10
    4106:	shr    $0x2,%edi
    4109:	vmovaps %xmm13,%xmm14
    410e:	shl    $0x4,%rdi
    4112:	vmovaps %xmm13,%xmm2
    4116:	lea    (%rdi,%r9,1),%r8
    411a:	and    $0x10,%edi
    411d:	je     41f5 <_Z3v4bRK4Args+0x125>
    4123:	movslq (%rsi),%r10
    4126:	movslq 0x4(%rsi),%r11
    412a:	add    $0x10,%rax
    412e:	movslq 0x8(%rsi),%rcx
    4132:	vbroadcastss -0x10(%rax),%xmm10
    4138:	movslq 0xc(%rsi),%rsi
    413c:	shl    $0x4,%r10
    4140:	shl    $0x4,%r11
    4144:	vbroadcastss -0xc(%rax),%xmm11
    414a:	vmovaps %xmm10,%xmm2
    414e:	shl    $0x4,%rcx
    4152:	vfmadd132ps (%rdx,%r10,1),%xmm14,%xmm2
    4158:	vbroadcastss -0x8(%rax),%xmm12
    415e:	shl    $0x4,%rsi
    4162:	vfmadd231ps (%rdx,%r11,1),%xmm11,%xmm14
    4168:	vbroadcastss -0x4(%rax),%xmm13
    416e:	vfmadd231ps (%rdx,%rcx,1),%xmm12,%xmm2
    4174:	vfmadd231ps (%rdx,%rsi,1),%xmm13,%xmm14
    417a:	mov    %r9,%rsi
    417d:	add    $0x10,%r9
    4181:	jmp    41f5 <_Z3v4bRK4Args+0x125>
    4183:	nopl   0x0(%rax,%rax,1)
    4188:	movslq (%r9),%rsi
    418b:	movslq 0x4(%r9),%rdi
    418f:	add    $0x20,%rax
    4193:	vbroadcastss -0x10(%rax),%xmm9
    4199:	vbroadcastss -0xc(%rax),%xmm4
    419f:	movslq 0x8(%r9),%r10
    41a3:	movslq 0xc(%r9),%r11
    41a7:	shl    $0x4,%rsi
    41ab:	shl    $0x4,%rdi
    41af:	vbroadcastss -0x8(%rax),%xmm2
    41b5:	vbroadcastss -0x4(%rax),%xmm14
    41bb:	vorps  %xmm9,%xmm8,%xmm10
    41c0:	vorps  %xmm4,%xmm7,%xmm11
    41c4:	shl    $0x4,%r10
    41c8:	vfmadd132ps (%rdx,%rdi,1),%xmm0,%xmm4
    41ce:	shl    $0x4,%r11
    41d2:	vfmadd132ps (%rdx,%rsi,1),%xmm3,%xmm9
    41d8:	vorps  %xmm2,%xmm6,%xmm12
    41dc:	vorps  %xmm14,%xmm5,%xmm13
    41e1:	vfmadd132ps (%rdx,%r10,1),%xmm9,%xmm2
    41e7:	vfmadd132ps (%rdx,%r11,1),%xmm4,%xmm14
    41ed:	lea    0x10(%r9),%rsi
    41f1:	add    $0x20,%r9
    41f5:	movslq (%rsi),%rdi
    41f8:	movslq 0x4(%rsi),%r10
    41fc:	vbroadcastss (%rax),%xmm9
    4201:	vbroadcastss 0x4(%rax),%xmm4
    4207:	movslq 0x8(%rsi),%r11
    420b:	movslq 0xc(%rsi),%rcx
    420f:	shl    $0x4,%rdi
    4213:	shl    $0x4,%r10
    4217:	vbroadcastss 0x8(%rax),%xmm3
    421d:	vbroadcastss 0xc(%rax),%xmm0
    4223:	vorps  %xmm9,%xmm10,%xmm8
    4228:	vorps  %xmm4,%xmm11,%xmm7
    422c:	shl    $0x4,%r11
    4230:	vfmadd132ps (%rdx,%rdi,1),%xmm2,%xmm9
    4236:	shl    $0x4,%rcx
    423a:	vfmadd132ps (%rdx,%r10,1),%xmm14,%xmm4
    4240:	vorps  %xmm3,%xmm12,%xmm6
    4244:	vorps  %xmm0,%xmm13,%xmm5
    4248:	vfmadd132ps (%rdx,%r11,1),%xmm9,%xmm3
    424e:	vfmadd132ps (%rdx,%rcx,1),%xmm4,%xmm0
    4254:	cmp    %r9,%r8
    4257:	jne    4188 <_Z3v4bRK4Args+0xb8>
    425d:	vaddps %xmm3,%xmm0,%xmm14
    4261:	vshufps $0x55,%xmm14,%xmm14,%xmm15
    4267:	vaddss %xmm15,%xmm14,%xmm10
    426c:	vunpckhps %xmm14,%xmm14,%xmm11
    4271:	vshufps $0xff,%xmm14,%xmm14,%xmm13
    4277:	vaddss %xmm11,%xmm10,%xmm12
    427c:	vaddss %xmm13,%xmm12,%xmm0
    4281:	vmovaps %xmm8,0x2db7(%rip)        # 7040 <_ZL6g_sink>
    4289:	vmovaps %xmm7,0x2dbf(%rip)        # 7050 <_ZL6g_sink+0x10>
    4291:	vmovaps %xmm6,0x2dc7(%rip)        # 7060 <_ZL6g_sink+0x20>
    4299:	vmovaps %xmm5,0x2dcf(%rip)        # 7070 <_ZL6g_sink+0x30>
    42a1:	ret    
    42a2:	nopw   0x0(%rax,%rax,1)
    42a8:	vxorps %xmm5,%xmm5,%xmm5
    42ac:	vxorps %xmm0,%xmm0,%xmm0
    42b0:	vmovaps %xmm5,%xmm6
    42b4:	vmovaps %xmm5,%xmm7
    42b8:	vmovaps %xmm5,%xmm8
    42bc:	jmp    4281 <_Z3v4bRK4Args+0x1b1>
    42be:	xchg   %ax,%ax

00000000000042c0 <_Z3v4cRK4Args>:
    42c0:	endbr64 
    42c4:	mov    0x20(%rdi),%ecx
    42c7:	cmp    $0x3,%ecx
    42ca:	jle    44c8 <_Z3v4cRK4Args+0x208>
    42d0:	mov    (%rdi),%rsi
    42d3:	mov    0x10(%rdi),%rdx
    42d7:	vxorps %xmm3,%xmm3,%xmm3
    42db:	mov    0x8(%rdi),%rax
    42df:	lea    -0x4(%rcx),%edi
    42e2:	vmovaps %xmm3,%xmm10
    42e6:	vmovaps %xmm3,%xmm9
    42ea:	lea    0x10(%rsi),%r11
    42ee:	vmovaps %xmm3,%xmm8
    42f2:	shr    $0x2,%edi
    42f5:	vmovaps %xmm3,%xmm15
    42f9:	shl    $0x4,%rdi
    42fd:	vmovaps %xmm3,%xmm11
    4301:	lea    (%rdi,%r11,1),%r10
    4305:	and    $0x10,%edi
    4308:	je     4409 <_Z3v4cRK4Args+0x149>
    430e:	movslq (%rsi),%r9
    4311:	movslq 0x4(%rsi),%r8
    4315:	add    $0x10,%rax
    4319:	movslq 0x8(%rsi),%rcx
    431d:	movslq 0xc(%rsi),%rsi
    4321:	shl    $0x4,%r9
    4325:	shl    $0x4,%r8
    4329:	vbroadcastss -0x10(%rax),%xmm2
    432f:	vbroadcastss -0xc(%rax),%xmm1
    4335:	add    %rdx,%r9
    4338:	add    %rdx,%r8
    433b:	shl    $0x4,%rcx
    433f:	vbroadcastss -0x8(%rax),%xmm11
    4345:	shl    $0x4,%rsi
    4349:	vfmadd132ps (%r9),%xmm15,%xmm2
    434e:	add    %rdx,%rcx
    4351:	vbroadcastss -0x4(%rax),%xmm12
    4357:	add    %rdx,%rsi
    435a:	vfmadd231ps (%r8),%xmm1,%xmm15
    435f:	vmovups 0x10(%r9),%xmm8
    4365:	vfmadd231ps (%rsi),%xmm12,%xmm15
    436a:	vmovups 0x10(%rsi),%xmm3
    436f:	mov    %r11,%rsi
    4372:	add    $0x10,%r11
    4376:	vfmadd132ps (%rcx),%xmm2,%xmm11
    437b:	vmovups 0x10(%r8),%xmm9
    4381:	vmovups 0x10(%rcx),%xmm10
    4386:	jmp    4409 <_Z3v4cRK4Args+0x149>
    438b:	nopl   0x0(%rax,%rax,1)
    4390:	movslq (%r11),%r9
    4393:	movslq 0xc(%r11),%rsi
    4397:	add    $0x20,%rax
    439b:	movslq 0x8(%r11),%rdi
    439f:	movslq 0x4(%r11),%r8
    43a3:	shl    $0x4,%r9
    43a7:	shl    $0x4,%rsi
    43ab:	vbroadcastss -0x8(%rax),%xmm11
    43b1:	shl    $0x4,%rdi
    43b5:	add    %rdx,%rsi
    43b8:	add    %rdx,%r9
    43bb:	shl    $0x4,%r8
    43bf:	vorps  0x10(%rsi),%xmm7,%xmm3
    43c4:	add    %rdx,%rdi
    43c7:	add    %rdx,%r8
    43ca:	vbroadcastss -0x10(%rax),%xmm7
    43d0:	vfmadd132ps (%r9),%xmm15,%xmm7
    43d5:	vorps  0x10(%rdi),%xmm4,%xmm10
    43da:	vfmadd132ps (%rdi),%xmm7,%xmm11
    43df:	vbroadcastss -0xc(%rax),%xmm4
    43e5:	vorps  0x10(%r9),%xmm6,%xmm8
    43eb:	vfmadd132ps (%r8),%xmm0,%xmm4
    43f0:	vorps  0x10(%r8),%xmm5,%xmm9
    43f6:	vbroadcastss -0x4(%rax),%xmm15
    43fc:	vfmadd132ps (%rsi),%xmm4,%xmm15
    4401:	lea    0x10(%r11),%rsi
    4405:	add    $0x20,%r11
    4409:	movslq (%rsi),%r9
    440c:	movslq 0x4(%rsi),%r8
    4410:	movslq 0x8(%rsi),%rdi
    4414:	movslq 0xc(%rsi),%rcx
    4418:	shl    $0x4,%r9
    441c:	shl    $0x4,%r8
    4420:	vbroadcastss (%rax),%xmm13
    4425:	vbroadcastss 0x4(%rax),%xmm14
    442b:	add    %rdx,%r9
    442e:	add    %rdx,%r8
    4431:	shl    $0x4,%rdi
    4435:	vbroadcastss 0xc(%rax),%xmm0
    443b:	shl    $0x4,%rcx
    443f:	vfmadd132ps (%r8),%xmm15,%xmm14
    4444:	add    %rdx,%rdi
    4447:	vbroadcastss 0x8(%rax),%xmm15
    444d:	add    %rdx,%rcx
    4450:	vfmadd132ps (%r9),%xmm11,%xmm13
    4455:	vorps  0x10(%rdi),%xmm10,%xmm4
    445a:	vfmadd132ps (%rdi),%xmm13,%xmm15
    445f:	vfmadd132ps (%rcx),%xmm14,%xmm0
    4464:	vorps  0x10(%r9),%xmm8,%xmm6
    446a:	vorps  0x10(%r8),%xmm9,%xmm5
    4470:	vorps  0x10(%rcx),%xmm3,%xmm7
    4475:	cmp    %r11,%r10
    4478:	jne    4390 <_Z3v4cRK4Args+0xd0>
    447e:	vaddps %xmm15,%xmm0,%xmm8
    4483:	vshufps $0x55,%xmm8,%xmm8,%xmm9
    4489:	vaddss %xmm8,%xmm9,%xmm3
    448e:	vunpckhps %xmm8,%xmm8,%xmm2
    4493:	vshufps $0xff,%xmm8,%xmm8,%xmm11
    4499:	vaddss %xmm2,%xmm3,%xmm1
    449d:	vaddss %xmm11,%xmm1,%xmm0
    44a2:	vmovaps %xmm6,0x2b96(%rip)        # 7040 <_ZL6g_sink>
    44aa:	vmovaps %xmm5,0x2b9e(%rip)        # 7050 <_ZL6g_sink+0x10>
    44b2:	vmovaps %xmm4,0x2ba6(%rip)        # 7060 <_ZL6g_sink+0x20>
    44ba:	vmovaps %xmm7,0x2bae(%rip)        # 7070 <_ZL6g_sink+0x30>
    44c2:	ret    
    44c3:	nopl   0x0(%rax,%rax,1)
    44c8:	vxorps %xmm7,%xmm7,%xmm7
    44cc:	vxorps %xmm0,%xmm0,%xmm0
    44d0:	vmovaps %xmm7,%xmm4
    44d4:	vmovaps %xmm7,%xmm5
    44d8:	vmovaps %xmm7,%xmm6
    44dc:	jmp    44a2 <_Z3v4cRK4Args+0x1e2>
    44de:	xchg   %ax,%ax

00000000000044e0 <_Z2v5RK4Args>:
    44e0:	endbr64 
    44e4:	mov    0x20(%rdi),%edx
    44e7:	cmp    $0x3,%edx
    44ea:	jle    4650 <_Z2v5RK4Args+0x170>
    44f0:	mov    0x10(%rdi),%rcx
    44f4:	lea    -0x4(%rdx),%esi
    44f7:	vxorps %xmm13,%xmm13,%xmm13
    44fc:	mov    0x8(%rdi),%rax
    4500:	vmovaps %xmm13,%xmm12
    4505:	shr    $0x2,%esi
    4508:	vmovaps %xmm13,%xmm11
    450d:	vmovaps %xmm13,%xmm10
    4512:	lea    0x40(%rcx),%r8
    4516:	vmovaps %xmm13,%xmm14
    451b:	shl    $0x6,%rsi
    451f:	vmovaps %xmm13,%xmm2
    4523:	lea    (%rsi,%r8,1),%rdi
    4527:	and    $0x40,%esi
    452a:	je     45be <_Z2v5RK4Args+0xde>
    4530:	vbroadcastss (%rax),%xmm10
    4535:	vbroadcastss 0x4(%rax),%xmm11
    453b:	add    $0x10,%rax
    453f:	vbroadcastss -0x8(%rax),%xmm12
    4545:	vbroadcastss -0x4(%rax),%xmm13
    454b:	vmovaps %xmm10,%xmm2
    454f:	vfmadd132ps (%rcx),%xmm14,%xmm2
    4554:	vfmadd231ps 0x10(%rcx),%xmm11,%xmm14
    455a:	vfmadd231ps 0x20(%rcx),%xmm12,%xmm2
    4560:	vfmadd231ps 0x30(%rcx),%xmm13,%xmm14
    4566:	mov    %r8,%rcx
    4569:	add    $0x40,%r8
    456d:	jmp    45be <_Z2v5RK4Args+0xde>
    456f:	nop
    4570:	vbroadcastss 0x10(%rax),%xmm9
    4576:	vbroadcastss 0x14(%rax),%xmm4
    457c:	lea    0x40(%r8),%rcx
    4580:	add    $0x20,%rax
    4584:	vbroadcastss -0x8(%rax),%xmm2
    458a:	vbroadcastss -0x4(%rax),%xmm14
    4590:	sub    $0xffffffffffffff80,%r8
    4594:	vorps  %xmm9,%xmm8,%xmm10
    4599:	vfmadd132ps -0x80(%r8),%xmm3,%xmm9
    459f:	vorps  %xmm4,%xmm7,%xmm11
    45a3:	vfmadd132ps -0x70(%r8),%xmm0,%xmm4
    45a9:	vorps  %xmm2,%xmm6,%xmm12
    45ad:	vorps  %xmm14,%xmm5,%xmm13
    45b2:	vfmadd132ps -0x60(%r8),%xmm9,%xmm2
    45b8:	vfmadd132ps -0x50(%r8),%xmm4,%xmm14
    45be:	vbroadcastss (%rax),%xmm9
    45c3:	vbroadcastss 0x4(%rax),%xmm4
    45c9:	vbroadcastss 0x8(%rax),%xmm3
    45cf:	vbroadcastss 0xc(%rax),%xmm0
    45d5:	vorps  %xmm9,%xmm10,%xmm8
    45da:	vfmadd132ps (%rcx),%xmm2,%xmm9
    45df:	vorps  %xmm4,%xmm11,%xmm7
    45e3:	vfmadd132ps 0x10(%rcx),%xmm14,%xmm4
    45e9:	vorps  %xmm3,%xmm12,%xmm6
    45ed:	vorps  %xmm0,%xmm13,%xmm5
    45f1:	vfmadd132ps 0x20(%rcx),%xmm9,%xmm3
    45f7:	vfmadd132ps 0x30(%rcx),%xmm4,%xmm0
    45fd:	cmp    %r8,%rdi
    4600:	jne    4570 <_Z2v5RK4Args+0x90>
    4606:	vaddps %xmm3,%xmm0,%xmm14
    460a:	vshufps $0x55,%xmm14,%xmm14,%xmm15
    4610:	vaddss %xmm15,%xmm14,%xmm10
    4615:	vunpckhps %xmm14,%xmm14,%xmm11
    461a:	vshufps $0xff,%xmm14,%xmm14,%xmm13
    4620:	vaddss %xmm11,%xmm10,%xmm12
    4625:	vaddss %xmm13,%xmm12,%xmm0
    462a:	vmovaps %xmm8,0x2a0e(%rip)        # 7040 <_ZL6g_sink>
    4632:	vmovaps %xmm7,0x2a16(%rip)        # 7050 <_ZL6g_sink+0x10>
    463a:	vmovaps %xmm6,0x2a1e(%rip)        # 7060 <_ZL6g_sink+0x20>
    4642:	vmovaps %xmm5,0x2a26(%rip)        # 7070 <_ZL6g_sink+0x30>
    464a:	ret    
    464b:	nopl   0x0(%rax,%rax,1)
    4650:	vxorps %xmm5,%xmm5,%xmm5
    4654:	vxorps %xmm0,%xmm0,%xmm0
    4658:	vmovaps %xmm5,%xmm6
    465c:	vmovaps %xmm5,%xmm7
    4660:	vmovaps %xmm5,%xmm8
    4664:	jmp    462a <_Z2v5RK4Args+0x14a>
    4666:	cs nopw 0x0(%rax,%rax,1)

0000000000004670 <_ZNSt6vectorIiSaIiEEC1EmRKS0_>:
    4670:	endbr64 
    4674:	push   %r13
    4676:	mov    %rsi,%rax
    4679:	push   %r12
    467b:	push   %rbp
    467c:	push   %rbx
    467d:	sub    $0x8,%rsp
    4681:	shr    $0x3d,%rax
    4685:	jne    473a <_ZNSt6vectorIiSaIiEEC1EmRKS0_+0xca>
    468b:	vpxor  %xmm0,%xmm0,%xmm0
    468f:	movq   $0x0,0x10(%rdi)
    4697:	mov    %rdi,%rbx
    469a:	mov    %rsi,%rbp
    469d:	vmovdqu %xmm0,(%rdi)
    46a1:	lea    0x0(,%rsi,4),%r13
    46a9:	test   %rsi,%rsi
    46ac:	je     4700 <_ZNSt6vectorIiSaIiEEC1EmRKS0_+0x90>
    46ae:	mov    %r13,%rdi
    46b1:	call   1150 <_Znwm@plt>
    46b6:	vmovq  %rax,%xmm1
    46bb:	lea    (%rax,%r13,1),%r12
    46bf:	movl   $0x0,(%rax)
    46c5:	lea    0x4(%rax),%rdi
    46c9:	vpunpcklqdq %xmm1,%xmm1,%xmm2
    46cd:	mov    %r12,0x10(%rbx)
    46d1:	vmovdqu %xmm2,(%rbx)
    46d5:	cmp    $0x1,%rbp
    46d9:	je     4728 <_ZNSt6vectorIiSaIiEEC1EmRKS0_+0xb8>
    46db:	cmp    %rdi,%r12
    46de:	je     46eb <_ZNSt6vectorIiSaIiEEC1EmRKS0_+0x7b>
    46e0:	lea    -0x4(%r13),%rdx
    46e4:	xor    %esi,%esi
    46e6:	call   1120 <memset@plt>
    46eb:	mov    %r12,0x8(%rbx)
    46ef:	add    $0x8,%rsp
    46f3:	pop    %rbx
    46f4:	pop    %rbp
    46f5:	pop    %r12
    46f7:	pop    %r13
    46f9:	ret    
    46fa:	nopw   0x0(%rax,%rax,1)
    4700:	xor    %r12d,%r12d
    4703:	movq   $0x0,(%rdi)
    470a:	movq   $0x0,0x10(%rdi)
    4712:	mov    %r12,0x8(%rbx)
    4716:	add    $0x8,%rsp
    471a:	pop    %rbx
    471b:	pop    %rbp
    471c:	pop    %r12
    471e:	pop    %r13
    4720:	ret    
    4721:	nopl   0x0(%rax)
    4728:	mov    %rdi,%r12
    472b:	mov    %r12,0x8(%rbx)
    472f:	add    $0x8,%rsp
    4733:	pop    %rbx
    4734:	pop    %rbp
    4735:	pop    %r12
    4737:	pop    %r13
    4739:	ret    
    473a:	lea    0x8c7(%rip),%rdi        # 5008 <_IO_stdin_used+0x8>
    4741:	call   1110 <_ZSt20__throw_length_errorPKc@plt>
    4746:	cs nopw 0x0(%rax,%rax,1)

0000000000004750 <_ZNSt6vectorIfSaIfEEC1EmRKS0_>:
    4750:	endbr64 
    4754:	push   %r13
    4756:	mov    %rsi,%rax
    4759:	push   %r12
    475b:	push   %rbp
    475c:	push   %rbx
    475d:	sub    $0x8,%rsp
    4761:	shr    $0x3d,%rax
    4765:	jne    481a <_ZNSt6vectorIfSaIfEEC1EmRKS0_+0xca>
    476b:	vpxor  %xmm0,%xmm0,%xmm0
    476f:	movq   $0x0,0x10(%rdi)
    4777:	mov    %rdi,%rbx
    477a:	mov    %rsi,%rbp
    477d:	vmovdqu %xmm0,(%rdi)
    4781:	lea    0x0(,%rsi,4),%r13
    4789:	test   %rsi,%rsi
    478c:	je     47e0 <_ZNSt6vectorIfSaIfEEC1EmRKS0_+0x90>
    478e:	mov    %r13,%rdi
    4791:	call   1150 <_Znwm@plt>
    4796:	vmovq  %rax,%xmm1
    479b:	lea    (%rax,%r13,1),%r12
    479f:	movl   $0x0,(%rax)
    47a5:	lea    0x4(%rax),%rdi
    47a9:	vpunpcklqdq %xmm1,%xmm1,%xmm2
    47ad:	mov    %r12,0x10(%rbx)
    47b1:	vmovdqu %xmm2,(%rbx)
    47b5:	cmp    $0x1,%rbp
    47b9:	je     4808 <_ZNSt6vectorIfSaIfEEC1EmRKS0_+0xb8>
    47bb:	cmp    %rdi,%r12
    47be:	je     47cb <_ZNSt6vectorIfSaIfEEC1EmRKS0_+0x7b>
    47c0:	lea    -0x4(%r13),%rdx
    47c4:	xor    %esi,%esi
    47c6:	call   1120 <memset@plt>
    47cb:	mov    %r12,0x8(%rbx)
    47cf:	add    $0x8,%rsp
    47d3:	pop    %rbx
    47d4:	pop    %rbp
    47d5:	pop    %r12
    47d7:	pop    %r13
    47d9:	ret    
    47da:	nopw   0x0(%rax,%rax,1)
    47e0:	xor    %r12d,%r12d
    47e3:	movq   $0x0,(%rdi)
    47ea:	movq   $0x0,0x10(%rdi)
    47f2:	mov    %r12,0x8(%rbx)
    47f6:	add    $0x8,%rsp
    47fa:	pop    %rbx
    47fb:	pop    %rbp
    47fc:	pop    %r12
    47fe:	pop    %r13
    4800:	ret    
    4801:	nopl   0x0(%rax)
    4808:	mov    %rdi,%r12
    480b:	mov    %r12,0x8(%rbx)
    480f:	add    $0x8,%rsp
    4813:	pop    %rbx
    4814:	pop    %rbp
    4815:	pop    %r12
    4817:	pop    %r13
    4819:	ret    
    481a:	lea    0x7e7(%rip),%rdi        # 5008 <_IO_stdin_used+0x8>
    4821:	call   1110 <_ZSt20__throw_length_errorPKc@plt>
    4826:	cs nopw 0x0(%rax,%rax,1)

0000000000004830 <_ZNSt6vectorIS_IdSaIdEESaIS1_EED1Ev>:
    4830:	endbr64 
    4834:	push   %r12
    4836:	mov    %rdi,%r12
    4839:	push   %rbp
    483a:	push   %rbx
    483b:	mov    0x8(%rdi),%rbx
    483f:	mov    (%rdi),%rbp
    4842:	cmp    %rbp,%rbx
    4845:	je     4872 <_ZNSt6vectorIS_IdSaIdEESaIS1_EED1Ev+0x42>
    4847:	nopw   0x0(%rax,%rax,1)
    4850:	mov    0x0(%rbp),%rdi
    4854:	test   %rdi,%rdi
    4857:	je     4890 <_ZNSt6vectorIS_IdSaIdEESaIS1_EED1Ev+0x60>
    4859:	mov    0x10(%rbp),%rsi
    485d:	add    $0x18,%rbp
    4861:	sub    %rdi,%rsi
    4864:	call   1160 <_ZdlPvm@plt>
    4869:	cmp    %rbp,%rbx
    486c:	jne    4850 <_ZNSt6vectorIS_IdSaIdEESaIS1_EED1Ev+0x20>
    486e:	mov    (%r12),%rbp
    4872:	test   %rbp,%rbp
    4875:	je     48a0 <_ZNSt6vectorIS_IdSaIdEESaIS1_EED1Ev+0x70>
    4877:	mov    0x10(%r12),%rsi
    487c:	pop    %rbx
    487d:	mov    %rbp,%rdi
    4880:	sub    %rbp,%rsi
    4883:	pop    %rbp
    4884:	pop    %r12
    4886:	jmp    1160 <_ZdlPvm@plt>
    488b:	nopl   0x0(%rax,%rax,1)
    4890:	add    $0x18,%rbp
    4894:	cmp    %rbp,%rbx
    4897:	jne    4850 <_ZNSt6vectorIS_IdSaIdEESaIS1_EED1Ev+0x20>
    4899:	jmp    486e <_ZNSt6vectorIS_IdSaIdEESaIS1_EED1Ev+0x3e>
    489b:	nopl   0x0(%rax,%rax,1)
    48a0:	pop    %rbx
    48a1:	pop    %rbp
    48a2:	pop    %r12
    48a4:	ret    
    48a5:	cs nopw 0x0(%rax,%rax,1)
    48af:	nop

00000000000048b0 <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_>:
    48b0:	endbr64 
    48b4:	push   %r15
    48b6:	mov    %rdx,%r15
    48b9:	movabs $0xfffffffffffffff,%rdx
    48c3:	push   %r14
    48c5:	push   %r13
    48c7:	push   %r12
    48c9:	push   %rbp
    48ca:	push   %rbx
    48cb:	sub    $0x18,%rsp
    48cf:	mov    0x8(%rdi),%r12
    48d3:	mov    (%rdi),%r14
    48d6:	mov    %r12,%rax
    48d9:	sub    %r14,%rax
    48dc:	sar    $0x3,%rax
    48e0:	cmp    %rdx,%rax
    48e3:	je     49fb <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_+0x14b>
    48e9:	test   %rax,%rax
    48ec:	mov    $0x1,%ebx
    48f1:	mov    %rsi,%rdx
    48f4:	mov    %rdi,%rbp
    48f7:	cmovne %rax,%rbx
    48fb:	xor    %ecx,%ecx
    48fd:	mov    %rsi,%r13
    4900:	add    %rbx,%rax
    4903:	setb   %cl
    4906:	sub    %r14,%rdx
    4909:	test   %rcx,%rcx
    490c:	jne    49b8 <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_+0x108>
    4912:	test   %rax,%rax
    4915:	jne    49e0 <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_+0x130>
    491b:	xor    %ebx,%ebx
    491d:	xor    %edi,%edi
    491f:	vmovsd (%r15),%xmm0
    4924:	sub    %r13,%r12
    4927:	lea    0x8(%rdi,%rdx,1),%r15
    492c:	vmovq  %rdi,%xmm2
    4931:	lea    (%r15,%r12,1),%r8
    4935:	vpinsrq $0x1,%r8,%xmm2,%xmm1
    493b:	vmovsd %xmm0,(%rdi,%rdx,1)
    4940:	vmovdqa %xmm1,(%rsp)
    4945:	test   %rdx,%rdx
    4948:	jg     4978 <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_+0xc8>
    494a:	test   %r12,%r12
    494d:	jg     49a0 <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_+0xf0>
    494f:	test   %r14,%r14
    4952:	jne    4985 <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_+0xd5>
    4954:	vmovdqa (%rsp),%xmm3
    4959:	mov    %rbx,0x10(%rbp)
    495d:	vmovdqu %xmm3,0x0(%rbp)
    4962:	add    $0x18,%rsp
    4966:	pop    %rbx
    4967:	pop    %rbp
    4968:	pop    %r12
    496a:	pop    %r13
    496c:	pop    %r14
    496e:	pop    %r15
    4970:	ret    
    4971:	nopl   0x0(%rax)
    4978:	mov    %r14,%rsi
    497b:	call   1190 <memmove@plt>
    4980:	test   %r12,%r12
    4983:	jg     49a0 <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_+0xf0>
    4985:	mov    0x10(%rbp),%rsi
    4989:	mov    %r14,%rdi
    498c:	sub    %r14,%rsi
    498f:	call   1160 <_ZdlPvm@plt>
    4994:	jmp    4954 <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_+0xa4>
    4996:	cs nopw 0x0(%rax,%rax,1)
    49a0:	mov    %r12,%rdx
    49a3:	mov    %r13,%rsi
    49a6:	mov    %r15,%rdi
    49a9:	call   1130 <memcpy@plt>
    49ae:	test   %r14,%r14
    49b1:	je     4954 <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_+0xa4>
    49b3:	jmp    4985 <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_+0xd5>
    49b5:	nopl   (%rax)
    49b8:	movabs $0x7ffffffffffffff8,%rbx
    49c2:	mov    %rbx,%rdi
    49c5:	mov    %rdx,(%rsp)
    49c9:	call   1150 <_Znwm@plt>
    49ce:	mov    (%rsp),%rdx
    49d2:	mov    %rax,%rdi
    49d5:	add    %rax,%rbx
    49d8:	jmp    491f <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_+0x6f>
    49dd:	nopl   (%rax)
    49e0:	movabs $0xfffffffffffffff,%rsi
    49ea:	cmp    %rsi,%rax
    49ed:	cmova  %rsi,%rax
    49f1:	lea    0x0(,%rax,8),%rbx
    49f9:	jmp    49c2 <_ZNSt6vectorIdSaIdEE17_M_realloc_insertIJRKdEEEvN9__gnu_cxx17__normal_iteratorIPdS1_EEDpOT_+0x112>
    49fb:	lea    0x637(%rip),%rdi        # 5039 <_IO_stdin_used+0x39>
    4a02:	call   1110 <_ZSt20__throw_length_errorPKc@plt>

Disassembly of section .fini:

0000000000004a08 <_fini>:
    4a08:	endbr64 
    4a0c:	sub    $0x8,%rsp
    4a10:	add    $0x8,%rsp
    4a14:	ret    
