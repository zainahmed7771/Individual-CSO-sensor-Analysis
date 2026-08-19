# GitHub upload instructions

Local Git is initialized by the release build. No remote is configured and nothing is pushed.

After reviewing `docs/RELEASE_AUDIT.md`, choosing a licence, and adding the approved presentation:
```bash
git status
git add .
git commit -m "Initial professor-ready CSO spatial-drivers release"
git branch -M main
git remote add origin https://github.com/<OWNER>/<REPOSITORY>.git
git push -u origin main
```

GitHub CLI was not detected during the build. Optional later alternative after installing/authenticating `gh`:
```bash
gh repo create <REPOSITORY> --source . --remote origin --private
git push -u origin main
```
Do not make the repository public until licence and data-review items are resolved.
