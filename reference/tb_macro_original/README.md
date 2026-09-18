# Vendored original

Read-only snapshot of `monash-emu/tb_macroeconomics` at `1d4fbd6` (2026-05-05),
the last commit that still has both `passive_detection` and `detection`.

`reference/make_golden.py` imports only `utils.py` (the triangular seed and the
tanh scale-up). It does not call `epi.py`: that file, at this commit, already
uses eight age bands. The golden model is the three-age reconstruction in the
port plan.

Do not edit these files. Regenerate goldens with `pixi run -e reference golden`.
