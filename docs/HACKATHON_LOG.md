# Hackathon build log

This log records what was built during the Meta Global AI Developer Hackathon building period (October 27 to November 4, 2026). Dates match the commit history.

| Date | What I built |
| --- | --- |
| Oct 26 (kickoff) | Chose the project idea, an AI maintainer assistant for open-source repositories, and ideated an initial coding scaffold. No code was committed before the building period began. |
| Oct 27 | Committed the first version: a tool-calling agent loop on Muse Spark, read-only repository and issue tools, the three workflows (triage, reproduce, onboard), a deterministic verifier for each workflow, an action gate that only allows labeling and commenting, a dry-run default, and unit tests that use a scripted model. |
| Oct 28 | Made `.env` loading work on Windows as well as macOS and Linux. |
| Oct 29 | Added a second model backend for the open-weights Muse Spark Open. I tried GroqCloud and OpenRouter first, then settled on Fireworks, a hackathon inference partner. Added streaming for long responses, a per-call output token cap, and immediate failure with a clear message on billing and authentication errors. |
| Oct 30 | Added support for projects that do not use pytest, starting with Django's own test runner. The reproduction verifier now accepts dotted test names and only counts a reproduction when the test fails on an assertion rather than crashing. Fixed command parsing for Windows paths. |
| Oct 31 | Ran the first evaluations. A TypeScript repository showed that reproduction needs a runnable test environment, and the verifier caught the agent claiming reproductions it never ran. |
| Nov 2 | Ran evaluations on scikit-learn, pandas and psf/black. Fixed issue screenshots failing on GitHub's redirecting image links by downloading images and sending them inline. Made evaluations continue past individual failures, save a full report automatically, and support rescoring without new model calls. Fixed the label scoring so process labels such as "Needs Triage" are excluded on both sides, capped triage at three labels, and added duplicate-focused datasets. |
| Nov 2 | On psf/black, gave the agent each label's description, fixed a verifier bug where pytest's shortened summary lines hid an `AssertionError`, and treated Black's "R:" and "S:" labels as workflow labels. Found that most "false alarm" duplicates were issues maintainers had closed as AI-generated spam, and that the agent applied the "spam / ai" label on its own. |
| Nov 3 | Made long evaluation runs resilient to provider outages: patient retries that respect the provider's `Retry-After` header, progress saved after every issue, a `--resume` option, and an early stop that only triggers when no issue has succeeded. Added clear explanations for failed GitHub writes. |
| Nov 3 | Ran Muse Spark and Muse Spark Open on the same psf/black issues for a fair comparison, and added a rescore option that limits scoring to a shared set of issues. |
| Nov 3 | Wrote the demo video script, forked psf/black for the one demo step that posts to GitHub, and set up a token with write access limited to the fork. |
| Nov 4 | Final evaluation numbers in the README, demo video, and submission. |
