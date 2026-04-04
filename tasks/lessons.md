# Lessons

- Verify the exact on-disk filename and path before concluding a file is missing or misnamed.
- Treat user-supplied paths as dirty input; strip leading and trailing whitespace before storing or resolving them.
- When the failure is caused by stale generated artifacts, prefer deleting and regenerating them over adding backward-compatibility code.
- For user-specified response formats, do not add presentation wrappers or inferred metadata beyond the requested fields; mirror the requested shape as literally as possible.
- When exposing identifiers in API or CLI output, use the canonical stored ID unless the user explicitly asks for a derived or display-only alias.
- Before concluding an environment dependency is missing, verify the active interpreter and re-check inside the project's required conda environment.
- When extracting hashtags as metadata, strip them from chunk text before embedding so tag markers do not pollute vector search or snippets.
- When the user specifies an existing CLI or executable name, preserve that name and do not introduce a new alias unless explicitly requested.
