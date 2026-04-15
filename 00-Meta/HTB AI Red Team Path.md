# HTB AI Red Team Path

Master progress index across the 11 modules in the HackTheBox AI Red Team path.

## Progress

**Overall: 0 / 11 modules completed (0%)**

| # | Module | Difficulty | Tier | Sections | Interactive | Est. Time | Progress | Status |
|---|---|---|---|---|---|---|---|---|
| 01 | [[01-Fundamentals-of-AI/01-Fundamentals-of-AI\|Fundamentals of AI]] | Medium | — | 24 | 1 | — | 0% | not-started |
| 02 | [[02-Applications-of-AI-in-InfoSec/02-Applications-of-AI-in-InfoSec\|Applications of AI in InfoSec]] | — | — | 25 | 17 | — | 0% | not-started |
| 03 | [[03-Introduction-to-Red-Teaming-AI/03-Introduction-to-Red-Teaming-AI\|Introduction to Red Teaming AI]] | — | — | 11 | 3 | — | 0% | not-started |
| 04 | [[04-Prompt-Injection-Attacks/04-Prompt-Injection-Attacks\|Prompt Injection Attacks]] | Medium | 2 | 12 | 6 | 8h | 0% | not-started |
| 05 | [[05-LLM-Output-Attacks/05-LLM-Output-Attacks\|LLM Output Attacks]] | Medium | 2 | 14 | 6 | 8h | 0% | not-started |
| 06 | [[06-AI-Data-Attacks/06-AI-Data-Attacks\|AI Data Attacks]] | Hard | 2 | 25 | 22 | 3d | 0% | not-started |
| 07 | [[07-Attacking-AI-Application-and-System/07-Attacking-AI-Application-and-System\|Attacking AI — Application and System]] | Medium | 2 | 14 | 8 | 8h | 0% | not-started |
| 08 | [[08-AI-Evasion-Foundations/08-AI-Evasion-Foundations\|AI Evasion — Foundations]] | Medium | 2 | 12 | 2 | 8h | 0% | not-started |
| 09 | [[09-AI-Evasion-First-Order-Attacks/09-AI-Evasion-First-Order-Attacks\|AI Evasion — First-Order Attacks]] | Hard | 2 | 23 | 4 | 2d | 0% | not-started |
| 10 | [[10-AI-Evasion-Sparsity-Attacks/10-AI-Evasion-Sparsity-Attacks\|AI Evasion — Sparsity Attacks]] | Hard | 2 | 28 | 3 | 3d | 0% | not-started |
| 11 | [[11-AI-Privacy/11-AI-Privacy\|AI Privacy]] | Medium | 2 | 21 | 3 | 2d | 0% | not-started |

## Learning path

1. **Foundations (01, 02):** theory + hands-on infosec applications of ML (spam classifier, anomaly detection, malware classification).
2. **Red-team intro (03):** ML OWASP Top 10 + LLM OWASP Top 10.
3. **Prompt + output attacks (04, 05):** direct/indirect prompt injection, jailbreaks, insecure output handling (XSS/SQLi/command-injection via LLM).
4. **Data attacks (06):** label flipping, clean-label, trojans/backdoors, tensor steganography.
5. **App + system attacks (07):** model reverse engineering, insecure components, MCP server vulnerabilities.
6. **Evasion (08, 09, 10):** GoodWords → FGSM/DeepFool → ElasticNet/JSMA.
7. **Privacy (11):** membership inference attacks, DP-SGD, PATE.

Each module ends with a Skills Assessment (`Skills-Assessment.md` inside the module folder).

## References

- Spec: [[docs/superpowers/specs/2026-04-14-htb-ai-redteam-vault-design|Vault design]]
- Plan: [[docs/superpowers/plans/2026-04-14-htb-ai-redteam-vault-implementation|Implementation plan]]
- Repo: https://github.com/syn-systema/htb-ai-redteam
