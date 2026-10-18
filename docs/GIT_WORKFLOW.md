# Git workflow (solo, simple, safe)

**Rules**
1. `main` always works. Never commit broken code to it.
2. One branch per task: `week02/resnet-baseline`, `week03/pgd-eval`. Name = `weekNN/short-topic`.
3. Small commits, imperative messages with a type prefix:
   `feat:` new code, `fix:` bug, `exp:` experiment config/result, `docs:`, `chore:`.
   Example: `exp: add PGD-AT config, 3 seeds`.
4. Merge to `main` with a pull request (even alone) so each week is one reviewable unit; squash is fine.
5. Tag each finished week: `git tag week02-done && git push --tags`.
6. Never commit data, checkpoints or API keys. Results tables (CSV) and figures (PNG/PDF) go in `results/`.
7. Every number in the paper must come from a run whose W&B page shows the git hash.

**Daily loop**
```bash
git switch main && git pull
git switch -c week02/resnet-baseline
# work, then:
git add -A && git status        # check what you are adding
git commit -m "feat: resnet18 baseline training"
git push -u origin week02/resnet-baseline
# open PR on GitHub, merge, then:
git switch main && git pull && git branch -d week02/resnet-baseline
```

**On Kaggle/Colab:** clone the repo, run, and commit only small result files. Code changes happen locally, not in the notebook, so history stays clean. Store the GitHub token in Kaggle Secrets / Colab Secrets, never in a cell.
