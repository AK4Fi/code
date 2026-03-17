from transformers import RobertaTokenizer


def tokenize_opcodes(opcode_str):
    # 加载 CodeBERT 分词器
    tokenizer = RobertaTokenizer.from_pretrained("../model/codebert-base")

    # 分割操作码
    opcodes = [op.strip() for op in opcode_str.split(",")]

    # 存储结果
    tokenization_results = []

    for op in opcodes:
        # 分词并解码（去除特殊符号）
        tokens = tokenizer.tokenize(op)
        clean_tokens = [t.replace('Ġ', '') for t in tokens]  # 去除 RoBERTa 风格的空格符号

        # 格式化输出
        tokenization_results.append(f"{op} → {clean_tokens}")

    return tokenization_results


if __name__ == "__main__":
    # 输入示例（可替换为任意操作码序列）
    input_opcodes = "aaa, aad, aam, aas, adc, add, addpd, addps, addsd, addss, afx, an, and, andnpd, andnps, andpd, andps, apphelp, applied, arpl, asc, available, bad, bit, bound, bsf, bsr, bstr, bswap, bt, btc, btr, bts, bytes, call, called, capabilities, cbw, cc, cccc, ccdd, cdq, ch, character, characters, clc, cld, clear, cli, clsid, clts, cmc, cmova, cmovb, cmovbe, cmovg, cmovge, cmovl, cmovle, cmovnb, cmovno, cmovnp, cmovns, cmovnz, cmovo, cmovp, cmovs, cmovz, cmp, cmpeqsd, cmpeqss, cmpleps, cmplesd, cmpltpd, cmpltps, cmpltsd, cmpneqpd, cmpneqps, cmpnlepd, cmpnlesd, cmpps, cmps, cmpsb, cmpsd, cmpsw, cmpxchg, color, combined, comdlg, comisd, comiss, completed, copied, cp, cpuid, cr, cvtdq, cvtpd, cvtpi, cvtps, cvtsd, cvtsi, cvtss, cvttps, cvttsd, cvttss, cwd, cwde, cy, daa, das, dbl, ddcc, dddd, dddddd, dec, delete, deleted, delta, determined, div, divps, divsd, divss, done, downloaded, dropped, dw, dwmapi, emms, endp, enter, enterw, established, evenexp, ex, extractps, fabs, fadd, faddp, fbld, fbstp, fchs, fclex, fcmovb, fcmovbe, fcmove, fcmovnb, fcmovnbe, fcmovne, fcmovnu, fcmovu, fcom, fcomi, fcomip, fcomp, fcompp, fcos, fdecstp, fdiv, fdivp, fdivr, fdivrp, femms, ffree, ffreep, fiadd, ficom, ficomp, fidiv, fidivr, fild, fimul, fincstp, fist, fistp, fisttp, fisub, fisubr, fld, fldcw, fldenv, fldl, fldlg, fldln, fldpi, fldz, flt, fmul, fmulp, fnclex, fndisi, fneni, fninit, fnop, fnsave, fnstcw, fnstenv, fnstsw, followed, found, fpatan, fprem, fptan, fptc, frndint, frstor, fsave, fscale, fsetpm, fsin, fsincos, fsqrt, fst, fstcw, fstp, fstsw, fsub, fsubp, fsubr, fsubrp, ftst, fucom, fucomi, fucomip, fucomp, fucompp, functional, fxam, fxch, fxsave, fxtract, fyl, generated, getsec, guid, hlt, hnt, ht, hung, icebp, idiv, iid, imul, in, inaccessible, inaccurate, inc, included, ins, insb, insd, inseng, insertps, inst, installable, installed, installing, insw, int, intentional, into, invd, iret, iretw, item, ja, jb, jbe, jcxz, jecxz, jg, jge, jl, jle, jmp, jnb, jno, jnp, jns, jnz, jo, jp, js, jz, lahf, lar, launched, lddqu, ldmxcsr, lds, lea, leave, leavew, les, lfs, lgdt, lgs, lib, lidt, lldt, loaded, lock, lods, lodsb, lodsd, lodsw, loop, loope, loopne, loopw, loopwe, loopwne, lp, lpsz, lsl, lss, made, maxps, maxss, message, migrated, minps, minsd, minss, minutes, missing, mov, movapd, movaps, movd, movdq, movdqa, movdqu, movhlps, movhpd, movhps, movlhps, movlpd, movlps, movmskpd, movmskps, movntdq, movnti, movntps, movntq, movq, movs, movsb, movsd, movss, movsw, movsx, movups, movzx, mpsadbw, mscoree, msnmetal, mul, mulpd, mulps, mulsd, mulss, name, neg, nop, not, off, offset, ole, oledlg, opened, or, orpd, orps, out, outs, outsb, outsd, outsw, ov, overwritten, pabsw, packssdw, packsswb, packusdw, packuswb, paddb, paddd, paddq, paddsb, paddsw, paddusb, paddusw, paddw, palignr, pand, pandn, pause, pavgb, pavgusb, pavgw, pb, pblendvb, pcinit, pclsid, pcmpeqb, pcmpeqd, pcmpeqw, pcmpgtb, pcmpgtd, pcmpgtw, pextrb, pextrd, pextrw, pf, pfacc, pfadd, pfcmpeq, pfcmpge, pfcmpgt, pfmax, pfmin, pfmul, pfnacc, pfpnacc, pfrcp, pfrcpit, pfrsqit, pfrsqrt, pfsub, pfsubr, phaddd, phaddw, phk, phminposuw, phsubd, pi, pinit, pinsrd, pinsrw, pmaddubsw, pmaddwd, pmaxsw, pmaxub, pminsw, pminub, pminud, pminuw, pmovmskb, pmovzxbd, pmovzxwd, pmulhrsw, pmulhuw, pmulhw, pmullw, pmuludq, pop, popa, popaw, popf, popfw, por, pqit, prefetch, prefetchnta, prefetcht, prefetchw, present, proc, processed, proto, psadbw, pshufb, pshufd, pshufhw, pshuflw, pshufw, psignw, pslld, pslldq, psllq, psllw, psrad, psraw, psrld, psrldq, psrlq, psrlw, psubb, psubd, psubq, psubsb, psubsw, psubusb, psubusw, psubw, pswapd, psz, ptest, punpckhbw, punpckhdq, punpckhqdq, punpckhwd, punpcklbw, punpckldq, punpcklqdq, punpcklwd, push, pusha, pushaw, pushf, pushfw, put, pwsz, pxor, qword, rc, rcl, rclsid, rcpps, rcpss, rcr, rdmsr, rdpmc, rdtsc, re, read, recopied, rein, rep, repe, replaced, repne, represented, required, retf, retfw, retn, retnw, retrieved, rg, rglpsz, rguid, riid, rol, ror, roundps, rsldt, rsm, rsqrtps, rsqrtss, rsts, rva, sahf, sal, sar, sbb, scas, scasb, scasd, scasw, separaters, service, set, setalc, setb, setbe, setl, setle, setnb, setnbe, setnl, setnle, setno, setnp, setns, setnz, seto, setp, sets, setz, sfence, sgdt, shl, shld, shown, shr, shrd, shufpd, shufps, sidt, sldt, specified, sqrtps, sqrtsd, sqrtss, stale, start, stc, std, sti, stmxcsr, stos, stosb, stosd, stosw, str, stru, sub, subpd, subps, subsd, subss, svldt, svts, syscall, sysenter, sysexit, sysret, sz, tbyte, terminated, test, to, topic, tstmetal, txt, ucomisd, ucomiss, ud, unicode, uninstalled, unk, unpckhpd, unpckhps, unpcklpd, unpcklps, unreliable, unusable, upda, urlmon, used, vaddps, valid, vandnpd, vblendps, vcmppd, vcmpss, vcomisd, vcvtpd, vcvtps, vcvtss, vcvttpd, verw, vextractf, vinsertf, vmovapd, vmovaps, vmovd, vmovddup, vmovdqa, vmovdqu, vmovlhps, vmovntpd, vmovntps, vmovss, vmovupd, vmread, vmulps, vmwrite, vorpd, vpaddsb, vpaddusw, vpaddw, vpandn, vpcext, vpcmpeqw, vperm, vpermilps, vpextrw, vpmaddwd, vpmaxub, vpmulhw, vpmullw, vpshufhw, vpshuflw, vpsllw, vpsrlq, vpunpckhbw, vpunpckhdq, vpunpckhqdq, vrcpss, vrsqrtss, vsansi, vshufps, vsqrtpd, vsqrtps, vsqrtsd, vsubps, vunpckhps, vunpcklps, vxorps, vzeroupper, wait, wbinvd, wer, where, wrmsr, xabort, xadd, xbegin, xchg, xlat, xmmword, xor, xorpd, xorps, xrstor"

    # 执行分词
    results = tokenize_opcodes(input_opcodes)

    # 打印结果
    print("操作码分词结果：")
    for line in results:
        print(line)