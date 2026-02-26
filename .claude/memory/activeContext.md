# Active Context

**Last Updated:** 2026-02-26

## Current Focus
- **All cluster jobs finished** — both Gautschi and Gilbreth queues empty
- **Study project:** Created ~/projects/study/ with 18-lesson UPGD codebase teaching plan
- **Lesson 1 in progress:** Iterator protocol (concept-example-quiz format)

## Active Jobs
None — all clusters idle.

## Completed Since Last Update
- Walker2d-v4 20M: all 4 methods × 20 seeds (was running on Gautschi)
- HumanoidStandup-v4 20M: all 4 methods × 20 seeds (split Gautschi + Gilbreth)
- SlipperyAnt ablation: all combo methods
- All previously active Gautschi/Gilbreth jobs

## Study Project
- Location: `~/projects/study/`
- 18 lessons covering full UPGD rebuild (supervised + gridworld DQN + PPO MuJoCo)
- Plan at: `~/projects/study/plan/`
- Format: concept → example → quiz (interactive, one question at a time)
- Current progress: Lesson 01 — iterator delegation concept

## Key Findings (from completed runs)
- Walker2d: hidden_only ≈ full >> Adam; output_only below Adam at 8M
- HumanoidStandup: full > Adam clearly; hidden_only tracking near full
- SlipperyAnt ranking: cbp ≈ shrink_head > cbp_h1 > cbp_h2 > cbp_no_gnt > upgd_full > std

## Next Actions
- Collect all 20M results from WandB for Walker2d + HumanoidStandup
- Generate final 4-method plots
- Decide on 40M reruns (plan at `plan/checkpoint_resume_40m.md`)
- Check HalfCheetah status on Gilbreth
- Continue UPGD lesson plan (Lesson 01 → 18)
