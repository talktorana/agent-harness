# PROJECT MEMORY — starting notes

This is the project's long-term memory. The head of this file is put in front
of the agent at the start of every task, so the newest material is what it sees
first. Older material is reachable with `memory_search`.

Notes **prepend** — put new ones at the top.

---

# How to use this file

Record things that would save the NEXT session time:

- a root cause and its fix
- a rule that is not obvious from the code
- a deploy or test recipe that took work to discover
- a trap that nearly cost something
- a decision, and **why** — the reasoning outlives the code

Prefer short and specific over long and general. A note that saves an hour is
worth more than a paragraph of narration.

Correct things when they turn out to be wrong: `memory_revise` for a detail,
`memory_retire` for a whole note. Do not stack a correction on top — a wrong
note left in place is a mistake chosen for repetition.

---

# Example note — replace with your own

## Test suites

Run with `python -m pytest -q` from the project root. The integration suite
needs `docker compose up -d db` first and takes ~90s; the unit suite is 4s.

## Why we use X instead of Y

We picked the simpler option deliberately — see the note below for the
reasoning. If you are about to "improve" it, read that first.
