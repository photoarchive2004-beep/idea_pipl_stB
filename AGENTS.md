# Codex instructions for this repo

- Target OS: Windows. Launchers: .bat and PowerShell.
- All console messages and generated reports must be in Russian (UTF-8).
- Secrets are stored in config/secrets.env. Never hardcode keys in code.
- Stage B must be universal: no domain-specific dictionaries (no “fish-specific” assumptions).
- Stage B must be robust: if any source API fails, continue and finish with status DEGRADED, not crash.
- Stage B must produce:
  - out/corpus.csv
  - out/search_log.json
  - out/prisma_lite.md
  - out/field_map.md
  - out/module_B.log
  - out/_moduleB_checkpoint.json
  - out/stageB_summary.txt (same summary printed in console)
- Required dedup priority: DOI > PMID > OpenAlexID > arXivID > (title+year+first_author).
- After work, print a short final summary in console and pause so the window stays open.