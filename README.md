# Custodian

**An AI maintainer assistant that triages issues, reproduces bugs as failing tests and writes onboarding guides for first-time contributors, built on Meta's Muse models.**

Hackathon track: **Build for Your World** (Meta Global AI Developer Hackathon, Oct 27 to Nov 4, 2026).
Bonus categories targeted: **Best Agentic Use-Case** and **Best Build on Meta Open Weight Models**.

---

## The problem

- During open-source contribution events such as GSoC and Hacktoberfest, maintainers receive far more issues than they can read, and many of those issues are duplicates or are missing the details needed to act on them.
- New contributors often pick up an issue, and then stall because they cannot tell which files matter or how to run the relevant tests.
- I have experienced both sides of this as an open-source contributor, which is why I chose this particular problem for the "Build for Your World" track.

## What Custodian does

Custodian runs three workflows on any GitHub issue:

- **Triage:** the agent reads the issue and attached screenshots (if any), explores the code to find the affected area, searches past issues for duplicates, picks labels from the repository's existing label set and drafts polite questions for whatever information is missing.
- **Reproduce:** the agent explores the code, writes a minimal test that asserts the behavior the reporter expected and runs it in a sandboxed checkout until it fails for the reason the issue describes.
- **Onboard:** the agent writes a short, issue-specific guide for a newcomer that lists the files to read, explains how they fit together and then shows how to run the relevant tests.

## The core idea: the agent proposes, code verifies, a gate decides

The agent never has write access to GitHub. Every workflow ends with a structured JSON proposal and deterministic code checks it before anything is posted.

- **Triage verification:** labels that do not exist in the repository are dropped, and a "duplicate of #N" claim is discarded unless #N is a real, different issue.
- **Reproduction verification:** Custodian re-runs the agent's test itself. A reproduction only counts if the test really fails, and failures caused by import errors or pytest collection errors are rejected. Unverified claims are never posted, and the evaluation reports how often verification caught the agent over-claiming.
- **Onboarding verification:** each file path the agent mentions is examined against the checkout, and paths that do not exist are removed or marked as unverified.
- **Action gate:** the only actions that can ever reach GitHub are adding existing labels and posting a comment. Closing, merging, assigning and deleting are impossible. Everything executes as a dry run unless you pass `--apply`, and every comment carries a footer saying it was drafted by an AI.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design.

## How it uses Meta's models

- **Muse Spark (Meta Model API)** drives the multi-step tool-calling loop for all three workflows.
- **Image understanding:** screenshots embedded in an issue are sent to Muse Spark alongside the text, so the agent can read error dialogs and broken UI that the reporter never described in words.
- **Muse Spark Open (open weights, via Fireworks):** the same agent runs on the open-weights model by setting `CUST_PROVIDER=fireworks`, allowing maintainers who cannot send code to a hosted API self-host the whole pipeline. The evaluation compares both providers.

## Quick start

```bash
git clone https://github.com/madsysharma/custodian && cd custodian
python -m venv .venv
source .venv/bin/activate        # On Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[dev]"
cp .env.example .env             # Then fill in MODEL_API_KEY (or GROQ_API_KEY) and GITHUB_TOKEN

# Dry run: prints the proposed labels and comments without posting anything.
custodian all --repo OWNER/REPO --issue 123

# Post the verified actions to GitHub (use this only on repositories you maintain).
custodian all --repo OWNER/REPO --issue 123 --apply
```

Custodian loads `.env` automatically from the folder you run it in, so no shell-specific setup is needed on Windows, macOS or Linux.

To run automatically on every new issue, start the optional webhook server with `pip install -e ".[server]"` and `uvicorn custodian.server:app`, then point a GitHub webhook for "Issues" events at `/webhook`.

## Evaluation

The evaluation replays the agent on a repository's past, closed issues and compares its output with what the maintainers actually did. Issue comments are hidden from the agent during evaluation, so the maintainers' own replies cannot leak the answer.

```bash
custodian build-eval --repo OWNER/REPO --limit 10 --duplicates 4
custodian eval --repo OWNER/REPO --dataset .custodian/eval/OWNER__REPO.jsonl --repro-issues ...
```

The report includes label precision, recall and F1; recall on true duplicates and the false-alarm rate on non-duplicates; and, for reproduction, the number of verified reproductions alongside the number of over-claims that verification caught.

| Metric | Muse Spark 1.3 | Muse Spark Open |
| --- | --- | --- |
| Label F1 | 0.605 | 0.565 |
| Duplicate recall | 0.25 | 0.0 |
| Duplicate false-alarm rate | 0.0 | 0.2 |
| Verified reproductions | 2 | 2 |
| Over-claims caught | 0 | 0 |

## Built during the hackathon

All code in this repository was written during the hackathon building period (October 27 to November 4, 2026). The day-by-day log is in [docs/HACKATHON_LOG.md](docs/HACKATHON_LOG.md), and the commit history shows the same timeline.

## Safety notes

- The reproduction sandbox runs code from the target repository. For the demo it runs locally with an allow-list of test commands, no shell and a timeout. For real use it should run inside a container or a Modal Sandbox.
- Issue text is treated as untrusted input, and the prompts tell the model never to follow instructions found inside an issue. The action gate means that even a successful prompt injection can only produce a labeled comment, never a destructive action.
- Please only use `--apply` on repositories you maintain or where the maintainers have agreed to try it.

## Project layout

```
custodian/
  agent.py          tool-calling loop
  workflows.py      triage, reproduce, onboard, and their verifiers
  policy.py         ActionGate: the only path to GitHub
  tools.py          read-only repository and issue tools, plus sandbox tools
  workspace.py      safe access to the local checkout
  sandbox.py        allow-listed test runner
  search.py         BM25 index for duplicate candidates
  evaluate.py       offline evaluation on past issues
  llm.py, config.py Muse Spark / Muse Spark Open client and settings
  server.py         optional GitHub webhook
tests/              unit tests with a scripted model (no API key needed)
docs/               architecture, demo script, hackathon log, submission checklist
```

## License

MIT
