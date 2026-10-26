# Architecture

## Overview

```mermaid
flowchart LR
    GH[(GitHub issue)] --> CTX[Context builder]
    CTX -->|issue text + screenshots| AG[Muse Spark agent loop]
    IDX[BM25 issue index] --> AG
    WS[Local repo checkout] --> AG
    SB[Sandbox test runner] <--> AG
    AG -->|JSON proposal| VER[Deterministic verifier]
    VER -->|verified result| GATE[ActionGate]
    GATE -->|dry run| LOG[Proposed actions for a maintainer]
    GATE -->|--apply| GH
```

## Components

- **Context builder (`workflows.build_context`):** this fetches the issue, its comments, the repository's labels and up to 300 recent issues, and it makes a shallow clone of the repository. The issue list is cached under `.custodian/cache` so repeated runs are fast.
- **Agent loop (`agent.py`):** this is a plain tool-calling loop with a step budget. When the budget runs out, the model is asked for its final answer without tools, so a run always ends with a proposal.
- **Tools (`tools.py`):** the agent can list, read, and grep files, search and read other issues, write test files in the local checkout and run allow-listed test commands. None of these tools can change anything on GitHub.
- **Verifiers (`workflows.verify_*`):** each workflow has a verifier that checks the proposal against reality. This is the part of the system that makes the agent's output trustworthy and it is where most of the engineering effort goes.
- **ActionGate (`policy.py`):** this is the only code that holds GitHub write access. It allows two action types (adding existing labels and posting a comment), applies limits and records every decision so a maintainer can audit it.

## Why verification is deterministic

- A second model call that "checks" the first one can share the same blind spots, so the verifiers use facts the system can observe directly: whether a file exists, whether a label exists and whether a test really fails when we run it.
- This also gives the evaluation a useful number: the over-claim rate, which is how often the agent said it succeeded when verification showed it had not.

## Threat model

- **Prompt injection through issue text:** issue content is wrapped in tags and labeled as untrusted in the system prompt. Even if an injection succeeds, the gate means the worst outcome is an unhelpful comment.
- **Untrusted code execution:** the reproduction step runs the target repository's tests. The local runner uses an allow-list, no shell, path guards that stop the agent from writing anything other than test files and a timeout. A production deployment should move this step into a container or a Modal Sandbox.
- **Noise for maintainers:** dry run is the default, comments are short and labeled as AI-drafted and unverified reproductions are never posted.

## Stretch goals (in priority order)

1. Run the sandbox inside Modal Sandboxes so reproduction is safe on any repository.
2. Add a "suggested fix" workflow that proposes a patch as a draft pull request, gated by the reproduction test passing afterward.
3. Support JavaScript repositories by allow-listing `npm test` and `npx vitest`.
4. Accept voice bug reports, transcribed with Muse Voice Transcribe, for reporters who find it easier to describe a problem out loud.
