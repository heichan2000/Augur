# Doc-drift check

A **non-optional** step when finishing any change that touches code: before you
open a PR (or merge), grep the docs and docstrings for claims the change
invalidates, and fix or file every hit.

This exists because prose does not fail a test suite. A README line, a module
docstring, or a design-doc sentence can go silently false the moment the code
beneath it changes, and nothing catches it until a reader trusts the stale
claim. Reading for drift does not scale and has already missed it twice (see
below). Grepping does.

## When

After the code is written and green, before the PR. Same checkpoint as running
the suite — treat a skipped doc-drift check the way you'd treat skipped tests.

## The mechanical step

1. List what this change altered: renamed or removed identifiers (functions,
   classes, constants, config keys, event fields), changed behaviour, and any
   new **user-facing copy** (notice text, error strings, CLI help).

2. Grep each of those, case-insensitively, across prose and docstrings — not
   just the file you edited:

   ```bash
   # Changed identifiers and behaviour claims
   git grep -in 'RENAMED_OR_CHANGED_IDENTIFIER' -- '*.md' '*.py' '*.tsx' '*.ts'

   # A distinctive phrase from any new user-facing copy — collisions with an
   # existing, now-contradictory claim are the ones reading misses
   git grep -in 'saved to the conversation' -- '*.md' '*.py' '*.tsx' '*.ts'
   ```

   Include `.py`/`.tsx`/`.ts` alongside `.md`: docstrings and comments drift
   exactly like Markdown does, and the worst cases are two comments in one file
   that now contradict each other.

3. For every hit, decide: is the claim still true? If not, fix it in this
   change, or — if the fix is a separate decision — file an issue and reference
   it. Do not leave a known-false claim standing.

## Why it's non-optional (it earned its place twice)

- **#22 / PR #36 (CORS):** `README.md:57` said "the backend needs no CORS
  policy" — true before that branch, false after. Missed by reading; caught
  only at the final whole-branch review. The plan's file-structure table never
  asked which docs the change would falsify, so no task step could have caught
  it.
- **#27 / PR #38 (truncation):** two hits, both missed by reading. A module
  docstring in `observability.py` claimed `tool_name` was emitted by
  `augur.agent` alone after the provider started emitting it too; and
  `UnsavedNotice`'s docstring in `turn-error.tsx` said a turn "cut off" is *not*
  in the model's context, while the new `TruncatedNotice` seventy lines below —
  copy: *saved to the conversation* — gave "cut off" the opposite meaning in the
  same file.

Both branches already carried a "grep the docs" note as a retro suggestion. It
stayed a suggestion, stayed manual, and still missed all three. That is why it
is written here as a step, not advice.
