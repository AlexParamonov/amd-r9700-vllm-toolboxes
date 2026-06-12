# Lessons Learned

## fp8-gemm-bench - 2026-06-12

**What worked:**
- Issue.md was well-structured with clear acceptance criteria and constraints
- TDD approach caught config format issues early (missing kpack/matrix_instr_nonkdim)
- Code review loop found real bugs (dead code, wrong test assertions)
- Refactor loop reduced line count without changing behavior
- Direct SSH + docker exec for manual testing on remote hardware

**What failed:**
- Search space was 20x too large (5120 vs 256 configs). Builder didn't check existing MI300X configs to calibrate. Estimated 8+ hours runtime, actual 24 min after fix.
- `num_stages` was added to search space but isn't in the output config format. Wasted 4x multiplier.
- Manual test couldn't verify AC #5 (no warnings on vLLM startup) because vLLM takes 5+ min to load and logs are hard to capture from detached docker exec.
- Merged on local machine, but target is remote (localai). Had to scp files separately.

**Next time:**
- Check existing configs in vLLM install dir BEFORE designing search space
- For remote targets, consider building on the target or using git push/pull workflow
- For long-running services (vLLM), capture startup logs to a file and check after
- Search space design should match the output format exactly (no extra params)
- Keep `tasks/lessons.md` in the main repo for cross-issue knowledge
