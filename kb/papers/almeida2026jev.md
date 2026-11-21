---
id: almeida2026jev
title: Introducing System One Models & Jev
authors: [Diogo Almeida]
year: 2026
venue: TypeSafe blog post
url: https://typesafe.ai/blog/introducing-system-one-models-and-jev
arxiv:
doi:
pdf_sha256:
tags: [decision-model, calibration, structured-output, typed-actions, non-autoregressive, real-time, game-agents, vendor-claims]
status: read
read: 2026-11-21
---

# Introducing System One Models & Jev

<!-- Source note: no paper exists. Read from web pages: the launch post (dated 2026-09-15) was
fetched 2026-11-20 23:26 local time (the saved copy's mtime is 2026-11-21, when it was copied in);
the linked typesafe.ai docs, evals and home pages were fetched 2026-11-21. Saved HTML and text in
<main>/papers/almeida2026jev/: launch.txt holds the visible text plus the page's embedded JSON
(with the collapsed FAQ answers); evals.txt holds the plot point labels. Section labels are the
reader's numbering, mapped under Method. All claims are vendor claims; none is independently
reproduced. -->

## TL;DR

Vendor launch post (TypeSafe), not a paper: Jev answers typed questions (yes/no, Score, Choice of up
to 255 options) about a text state with calibrated probabilities, in parallel in one call, trained
by "RLCD"; vendor evals: 67.8% (best 74.1%) at 0.4 s, $0.0004 per case; Doom demo on text state.

## Problem

- The post's framing (Sec. 1, Sec. 2): LLMs post-trained with RLHF or RLVR produce strings, so
  software must parse and validate their output, they can hallucinate and make type errors, they
  are slow (seconds to minutes) and expensive when embedded in code, and their stated confidence is
  overconfident and inconsistent. Automating a task needs a model that says when it is in the 5% of
  cases it gets wrong (Sec. 2, "Confidence" row).
- RLHF optimizes "the text that a human rater prefers", which the authors call the wrong task for
  automation; RLVR fits only programmatically checkable tasks and, they say, gives spiky,
  non-robust intelligence (Sec. 11, FAQ "Why was a new training algorithm needed?"). The AI
  primer adds that RLHF rewards sycophancy and causes mode dropping (Sec. 14).
- Survey lens (reader's framing, not the post's): a policy that reads a structured game state and
  scores a fixed, typed action set in one forward pass, with usable probabilities, at frame-rate
  latency. That is the shape of a game-playing policy head, which is why this note is in the KB.

## Method

What is disclosed (very little about the model itself):
- **Interface.** A request is a `state` (natural-language text, a JSON object or an array; PNG or
  JPEG images since the Jev 1.14 release, also on 1.13) plus any number of named, typed questions (Sec. 12, Sec. 15).
  Types: Choice (one option from a set, with a probability for each option and a confidence),
  Score (ordered levels: score, per-level probabilities, confidence), Noul (yes/no: one
  probability), Box (image bounding box, Jev 1.14 only) (Sec. 12, Sec. 13, Sec. 20). Outputs are
  constrained to the declared types; there is no text generation (Sec. 13).
- **Parallel, non-autoregressive answering.** The post contrasts LLMs' token-by-token sampling
  with Jev "generates all outputs in a single query" (Sec. 2, "Sampling" row; Sec. 4). The docs
  say the state is ingested once and every question is evaluated in parallel and in isolation
  against it (Sec. 12, Sec. 15, Sec. 17). The post mentions "a new model architecture" and a
  "parallel sampler" but describes neither (Sec. 1); the FAQ says only that Jev is "neither small
  nor an LLM" (Sec. 11). Nothing says whether per-question outputs are separate heads.
- **Training: RLCD** ("Reinforcement Learning for Calibrated Decisions"). Stated contract: no
  text, decisions plus probabilities, higher probability should mean higher chance of being correct;
  probabilities "optimized against outcomes" (Sec. 13, Sec. 14). No objective, reward, data or scale
  is given; all training data is made in-house, none from customers (Sec. 11; Sec. 15).
- **Confidence** is a fixed statistic of the returned distribution, not a separate output: for a
  Choice with n options, (p_max - 1/n)/(1 - 1/n); for a Score, 1 minus the probability-weighted
  distance from the modal level over the uniform distribution's mean absolute deviation, floored
  at 0; a Noul has none (Sec. 16).
- **Workflow evals** (Sec. 5, Sec. 21): tasks are decomposed in code into narrow Noul/Choice/Score
  questions; reference labels are the average answer of GPT-6 Astra and Claude Fable 5.1 at high
  thinking, so "accuracy" is agreement with those two models, not with ground truth. Four
  workflows (security incidents, agent-trace observability, invoice processing, customer service);
  each plotted point averages one model's accuracy, cost and time over the four with equal weight;
  other models at their provider's default reasoning setting. Each LLM also runs the same policy
  as a single prompt ("prompt" route).
- **Game demos in the post.** Doom (Sec. 8): a bot driven by about 10 Jev queries per second, on
  game state given as a text data structure, not images. Wikiracing (Sec. 9): choose the next
  Wikipedia link from hundreds to thousands per step, scored in two stages above 255 options.
  Both are videos; the reader saw only the text around them.
- **Claim labels used below:** "asserted" = stated without evidence shown; "vendor-measured" =
  TypeSafe's own numbers, not reproduced by anyone else; "by construction" = follows from the API
  contract; "video (not viewed)" = the post embeds a recording of this, which the reader did not
  watch, so only the recording's existence is backed and every detail comes from the post's text.

Reference map (headings of the launch post numbered in page order, h1 to h4; most have a
"Nuance" box): Sec. 1 "Introducing System One Models & Jev" (title and intro text); Sec. 2
"Frontiers, Old and New" (comparison table); Sec. 3 "Evidence / Technical Results"; Sec. 4
"Side-by-side demonstration"; Sec. 5 "Workflow evals"; Sec. 6 "Hallucination and Type-safety";
Sec. 7 "Fun Demos"; Sec. 8 "Doom"; Sec. 9 "Wikiracing"; Sec. 10 "What's next"; Sec. 11 "We Give A
FAQ" (answers collapsed on the page, present in its embedded JSON, saved in launch.txt). All on
https://typesafe.ai/blog/introducing-system-one-models-and-jev.
Other pages, numbered on from the post (saved file in <main>/papers/almeida2026jev/, .html + .txt):
Sec. 12 docs Introduction, docs.typesafe.ai/introduction (docs_introduction); Sec. 13 System One,
/concepts/system-one (docs_concepts_system-one); Sec. 14 AI primer,
/introduction/machine-learning-primer (docs_introduction_machine-learning-primer); Sec. 15 Models,
/models (docs_models); Sec. 16 Confidence, /confidence (docs_confidence); Sec. 17 Primitives,
/primitives (docs_primitives); Sec. 18 Choice, /primitives/choice (docs_primitives_choice);
Sec. 19 Model jaggedness, /model-jaggedness (docs_model-jaggedness); Sec. 20 Model changelog,
/changelog (docs_changelog); Sec. 21 evals.typesafe.ai overview plot (evals); Sec. 22 home page
typesafe.ai (home). The launch post is launch.html/.txt.

## Claims

- C1: Asserted: Jev reaches intelligence similar to existing LLMs on "System One tasks" while being two orders of magnitude faster and more efficient (Sec. 1)
- C2: Asserted: end-to-end response time is 70-500 ms for Jev versus 3-329 s for frontier LLMs, 40x-200x faster at the same intelligence for "System One shaped queries"; their published evals run from laptops on the US West Coast, where the service is based (Sec. 2, Sec. 3)
- C3: Price: $0.042 per million input tokens ($42 per billion), output tokens free; the post says it cannot prove the price isn't subsidized (Sec. 2, Sec. 3, Sec. 15)
- C4: By construction: outputs are type-safe structured values from answer sets declared in advance, so schema violations cannot occur; the post's 0% hallucination figure for Jev is "not empirical" but follows from guaranteed schema matching, while the LLM rates beside it come from OpenRouter (Sec. 2, Sec. 3, Sec. 6)
- C5: Asserted: every answer carries calibrated probabilities (higher confidence means higher accuracy) and similar inputs get similar answers; the docs add that calibration holds over groups of predictions and does not guarantee any single answer (Sec. 2, Sec. 13, Sec. 14)
- C6: Asserted: all outputs of a query are produced in parallel in one pass rather than token by token; all questions in a request are evaluated in parallel and in isolation against one state, so adding questions "barely changes" response time and causes no context rot (Sec. 2, Sec. 4, Sec. 12, Sec. 17)
- C7: Vendor-measured: batching 13 questions over one article into one call was 12.2x cheaper and 11.8x faster than 13 separate calls, with no change in answers (Sec. 17)
- C8: Vendor-measured: averaged over the four workflows, Jev scores 67.8% agreement with the Astra/Fable reference at $0.0004 and 0.4 s per case; the most accurate entries are "sol" 74.1% ($0.0836, 23.3 s) and "opus 5" 73.1% ($0.1761, 37.8 s), and "terra" 67.9% ($0.0304, 10.1 s) (Sec. 21)
- C9: Vendor-measured: averaged over the four workflows, every model is more accurate, cheaper and faster run as a workflow than with the same policy as one prompt (Sec. 21, Sec. 5)
- C10: Vendor-measured: the home page's "193.6x faster, 444.6x cheaper" comes from these workflow evals; the authors expect it to be at the high end of real-world gains, note the workflows were written by their own model-capabilities team (possible bias) and are outside the training distribution, and that the two-model reference favours OpenAI and Anthropic models (Sec. 5, Sec. 22)
- C11: Video (not viewed) of Jev driving a Doom bot; asserted in the text: the state is a data structure with text, not images, and the bot makes about 10 queries per second costing about $7 per hour; the post concedes a non-AI bot could play better and that the point is reacting to different state representations and following instructions (Sec. 8)
- C12: Video (not viewed) of Jev playing Wikiracing against LLMs; asserted in the text: each step picks among hundreds to thousands of links; a Choice takes at most 255 options, so larger sets are scored independently then chosen in a second stage, causing occasional slowdowns; speedups are smaller than elsewhere because the LLMs ran without reasoning (Astra at its lowest setting), which the post says is also why Jev tended to finish in fewer steps, read by the authors as a sign of greater intelligence (Sec. 9, Sec. 18)
- C13: Video (not viewed) of a side-by-side run of Jev and an LLM on one query; asserted in its nuance box: the query was simplified and the state a short paragraph, which favours Jev; in the recorded run Jev disagreed with GPT-5.6 Terra (default reasoning) only on "Churn likelihood level" (Sec. 4)
- C14: Known failure modes the vendor lists: literal reading, counting and numeric precision, date comparison, multi-hop indirection, accuracy falling as the state fills with irrelevant detail, adversarial content in the state, contradictory instructions, in some cases a lean toward the first Choice option, and text generation (Sec. 19)
- C15: Jev publishes no results on public benchmarks, by policy (Sec. 11)
- C16: Context limits: 64k tokens per request and 32k per question (state plus one question) by default, or 262k for both with a long-context header, which is slower and less accurate; inputs are text and images, not audio or video (Sec. 15, Sec. 20)

## Implementation notes

- Model version: the launch post (2026-09-15) precedes Jev 1.14 (2026-10-14); the docs and evals
  describe 1.13/1.14 as of access, so C7, C8, C16 may not describe the launch model (source: Sec. 15, Sec. 20)
- Eval model names "sol", "luna", "terra", "opus 5", "sonnet 5", "haiku 4.5", "DS v4 flash/pro"
  are as printed; their providers are only given as the legend groups TypeSafe, OpenAI, Anthropic,
  Fireworks (source: Sec. 21)
- The 3-329 s LLM latency range links to llm-benchmarks.diegoromero.es; not checked (source: Sec. 2)
- A Python adapter is linked (github.com/typesafe-ai/system-one-adapter-python), presumably the
  "System One LLM" wrapper; not read (source: Sec. 5; the identification is an assumption)
- Architecture, parameter count, the parallel sampler and the RLCD objective are undisclosed; for a
  game policy, a stand-in is a transformer encoder over the state with one masked softmax per
  typed question, trained on outcome reward with a proper scoring rule (source: assumption)
- Calibration of a stand-in could be checked with expected calibration error on a held-out set
  of decisions with known outcomes (source: assumption)

## Limitations / open questions

- Vendor marketing, not a paper: no method, data, model size, or training details; every
  number is TypeSafe's own and none is independently reproduced (all sections).
- The workflow-eval "accuracy" is agreement with two other LLMs' averaged answers, which the
  authors say biases toward OpenAI and Anthropic models (Sec. 5, Sec. 21).
- On their own overview plot Jev is the cheapest and fastest but not the most accurate (C8).
- Calibration is asserted but no reliability diagram or calibration metric is published in the
  pages read (Sec. 2, Sec. 13, Sec. 16).
- The Doom demo uses text state, not pixels; the vendor says images are not used there "yet" (Sec. 8).
- Self-listed weaknesses: numbers, counting, dates, indirection, large states, option-order bias (C14).
- "Can't hallucinate" (Sec. 1) equates hallucination with schema violation (C4): a schema-valid
  but wrong choice is still possible; the docs say calibration does not guarantee that any one
  answer is correct (Sec. 13) and list failure modes (Sec. 19).
- Survey contrast (reader's): Jev reads text state; the pixel-based game agents in the queue
  (Gato, reed2022generalist; VPT, baker2022video; notes not merged yet) learn from raw frames.
- Open question: the brief's Minecraft (24 ms decisions), Pokémon Red and Pong demos appear on
  no typesafe.ai page fetched (launch post, home, blog index, docs index, evals); they are likely
  on social media or in videos. Not fetched, so not claimed here.
- Open question: is "parallel" one forward pass per request with a decoder per question, or
  batched independent passes sharing a cached state encoding? The docs say only that the state
  is ingested once (Sec. 15).
- Open question: what does RLCD optimize? A proper scoring rule (log loss, Brier) against
  outcomes would explain "calibrated", but the vendor doesn't say.
- Open question: for real-time games, 70-500 ms per call (C2) and about 10 Hz (C11) is slower
  than frame rate; how the latency splits between network and model is not reported.

## Related

- related [[schulman2017proximal]]: RL policy optimization; RLCD is undisclosed RL post-training, so PPO-style training of a typed-action policy is the obvious stand-in
- related [[vaswani2017attention]]: likely backbone family for a state-in, typed-decisions-out model, though Jev's architecture is undisclosed
- related [[reed2022generalist]]: Gato's tokenised actions are one answer to a single policy over mixed action spaces; Jev instead declares typed answers (Choice, Score, yes/no) per request and returns calibrated probabilities.
- related [[baker2022video]]: VPT is a game policy with a native keyboard-and-mouse action space in a "joint hierarchical" head (one categorical over key combinations plus a camera head), a contrast to Jev's typed heads over a text state.
- related [[serrano2024crossdomain]]: Jev's text state and per-request action sets are one way to cross domains without a shared state or action space

## Experiments

<!-- kb:begin experiments -->
- none
<!-- kb:end experiments -->
