# STYLE.md

How to write text in this repository: code comments, the README, and the
docs in `.agents/`. Read this before writing or editing any of them. These
rules exist so the repo reads like a person wrote it, and so nobody has to
re-litigate them in review.

## The one rule that matters

Write for the reader of each text.

This is living documentation, not a changelog. It states the rules that are
current now; it does not narrate how they evolved. When two entries conflict,
the most recent decision wins, and the older text gets rewritten or deleted,
never annotated with "previously we did X". The same applies to every doc in
this repo: if you find prose telling a story about what used to happen, fix
the prose to state what is true now.

- A **code comment** is read by a maintainer who is already inside the
  code. It states a short, objective fact: what the function does, or why
  it does it that way. It is not a changelog, not a blog post, not a
  launch announcement.
- The **README** is read by a human deciding whether to use the library.
  Write like a person talking to that person.
- The **`.agents/` docs** are read by agents (and humans) working on the
  repo. Orient them: what exists, what is deliberate, what not to break.

If a sentence would sound at home in a press release, it does not belong
in any of the three.

## Hard rules (enforced in review)

1. **No typographic Unicode in prose.** No em-dashes (`—`), en-dashes
   (`–`), curly quotes (`'` `'` `"` `"`), ellipsis character (`…`), or
   arrows (`→`) in comments, docs, commit messages, PR bodies, or tag
   messages. Use commas, colons, semicolons, parentheses, `...`, and a
   plain ASCII hyphen for ranges (`D1-D5`, `v1.19.0 -> v1.61.0`).
   Diagrams (ASCII art in docs) may keep box-drawing characters: a
   diagram is not prose.

2. **No references to `DECISIONS.md` entries outside `.agents/`.** IDs
   like `O8`, `A11`, `D5` exist for people reading the decisions doc.
   They never appear in code comments, commit messages, PR bodies, or
   tag messages. If a comment needs the rationale, state the rationale
   in one line instead of pointing at an ID.

3. **Name components in words, not paths.** Write "the OpenAI adapter",
   "the Langfuse plugin", "the Redis session service". Do not write
   `genai/openai/completions` or `plugin/langfuse` in prose aimed at a reader;
   paths are for technical context (file headers, test names).

4. **Fail-loud over silent, and say so when relevant.** If a function
   returns an error where it used to drop something silently, that is
   worth one line. It is the kind of fact a maintainer needs.

## Code comments

A code comment answers one of two questions, in as few lines as possible:

- **What** does this do, when the name alone is not enough.
- **Why** this way, when the choice is not obvious from the code.

Guidelines:

- **Short.** If a comment needs nine lines, the problem is the comment,
  not the reader. Most functions need zero to four lines.
- **Objective.** State what is true now. Do not narrate history
  ("previously this dropped X") or the journey ("after trying Y we
  settled on Z"): that belongs in commit messages and `DECISIONS.md`.
- **No marketing.** Banned vocabulary in code comments includes
  "flagship", "natural", "elegant", "seamless", "robust", and any
  sentence built around an em-dash to sound dramatic.
- **References are rare.** A link or a doc citation is justified only
  when the behaviour implements an external contract the reader cannot
  be expected to know (an API quirk, a provider's documented rule).
  One pointer, not a bibliography.

Example, from this repo:

```go
// convertFileDataToBlock maps a FileData part to an image block with a URL
// source; the URI goes through verbatim, nothing is downloaded. Images only:
// other media types need uploaded bytes (InlineData). Plain http is allowed
// for gateways on Config.BaseURL; anthropic.com enforces https itself.
```

Four lines, states the contract, keeps the one non-obvious why. That is the
bar.

## README

- Write for a human evaluating the library. Short sentences, plain words.
- Feature lists say what the thing does, not how impressive it is.
- Examples should be copy-pasteable and boring. Boring is a compliment.

## `.agents/` docs

- These orient the next worker. Lead with what exists and what is
  deliberate, so nobody "fixes" a load-bearing behaviour back into a bug.
- Referencing decision IDs (`O8`, `A11`, ...) is fine here, and only here.
- Keep the same no-Unicode and no-marketing rules as everywhere else.

## Commit, PR, and tag messages

- Same hard rules: no typographic Unicode, no decision IDs, component
  names in words.
- Tag messages: what changed, per component, plus a behaviour note if
  observable behaviour changes. Short. The audience is someone deciding
  whether to upgrade.
- PR bodies stand alone: do not cite other PRs or re-argue merged work
  unless there is a real dependency.
