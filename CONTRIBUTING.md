# Contributing to CleanCanvas Studio

By submitting a contribution, you agree that it may be distributed under this
repository's Apache License 2.0 unless a separate written agreement states otherwise.

## Before submitting

- Search existing issues and documentation first.
- Do not attach private media, credentials, model caches or personal data.
- Keep each change focused and preserve upstream copyright/license notices.
- Do not commit model weights, virtual environments, generated artifacts or IDE files.
- Document user-visible behavior and add or update tests.

Run before opening a pull request:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s test -p "test_*.py" -v
.\.venv\Scripts\python.exe -m compileall -q backend ui gui.py
```

## Third-party code and models

Before adding third-party code, assets or model wrappers, record its upstream
URL, exact version, license, required NOTICE/attribution text, and whether its
model weights have a separate redistribution license. Update
`THIRD_PARTY_NOTICES.md` and `MODIFICATIONS.md` when relevant.
