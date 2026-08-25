# Maintainer release procedure

Do development on a branch and require a clean validation run before merging:

```powershell
git switch -c feature/descriptive-name
python -m pip install -e ".[full,test]"
python scripts\run_release_checks.py
git status
git add .
git commit -m "Describe the reproducibility change"
git push -u origin feature/descriptive-name
```

Open a pull request, confirm both Windows and Ubuntu CI jobs pass, review the generated diff and merge without rewriting scientific history. Tag stable releases from `main`:

```powershell
git switch main
git pull --ff-only
git tag -a v1.1.0 -m "Executable end-to-end teaching and scientific-core release"
git push origin v1.1.0
```

Never commit `config/scientific.yaml`, raw provider data, credentials or machine-local outputs.
