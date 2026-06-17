# Codenames LLM Benchmark Design

## Goal

Build a reproducible benchmark where LLM models play many Codenames-style games against each other in homogeneous four-agent teams. Each model receives an Elo-style rating and diagnostic scores for semantic association, clue quality, deduction, collaboration, calibrated risk, and strategic use of public opponent information.

## Canonical launcher

The active OpenRouter runner is `scripts/run_openrouter_tournament.py`. It replaces the older one-off tournament and smoke scripts, which now live in `ARCHIVE/scripts/`.

## Why Codenames is a good LLM benchmark

Codenames tests several capabilities that are difficult to isolate with ordinary question-answer benchmarks. A spymaster must map a set of target words into a compact clue while avoiding misleading associations. Guessers must infer intended targets from a clue, reason probabilistically about ambiguity, stop before overreaching, and integrate earlier clues into later decisions. Because the opponent’s clues are public, the game also tests whether a model can extract useful information from another team’s behavior without confusing that information with its own objective.

This makes the benchmark especially valuable for measuring semantic compression, pragmatic reasoning, multi-agent coordination, latent theory-of-mind behavior, and robustness under partial observability.

## Game format

Each team is composed of four agents using the same model. In each game, one agent is assigned as spymaster and three agents are assigned as guessers. The spymaster sees the full key: friendly words, opponent words, neutral words, and assassin. Guessers see only public board state, prior clues, prior guesses, revealed colors, and all public discussion permitted by the benchmark protocol.

Roles should rotate across games. If a model has four agent instances, each instance should serve as spymaster for a balanced number of games. For API-backed models without persistent identity, this can be simulated by rotating role prompts and agent IDs.

A round-robin tournament pairs every model team against every other model team. Each matchup should include hundreds of games, split evenly between first-player advantage and second-player position. Every game uses a deterministic seed so boards can be replayed. For fairness, mirrored boards should be used where possible: model A as red against model B as blue, then model B as red against model A as blue on the same board seed.

## Public and private information

The benchmark should model information boundaries carefully. Spymasters receive private board identities and must not reveal illegal information. Guessers receive public state only. Both teams can observe all public clues and revealed guesses from both sides. This matters because the opposing team’s clue often leaks information about their hidden word set. A strong team may use this to avoid opponent words or infer dangerous associations.

Recommended information channels are: private spymaster prompt, private guesser prompt, public event log, team deliberation channel, and final action channel. The simplest version can make team deliberation visible only within a team, while all clues and guesses are public. A later variant can test fully public deliberation, but that changes the game substantially and should be reported separately.

## Agent protocol

A spymaster turn should require structured output with a clue word, an intended count, and optional private rationale for logging. The clue and count become public. The private rationale should never be shown to guessers during play.

A guesser phase should allow the three guesser agents to independently propose ranked guesses, confidence scores, and stop recommendations. A team aggregator then chooses the final guess sequence. The aggregator can be either a fourth reasoning call using the same model, a deterministic voting rule, or a separate protocol where guessers debate before final action. The cleanest initial benchmark uses deterministic aggregation so the measured model behavior comes from the spymaster and guessers rather than an extra hidden arbiter.

All model outputs should be JSON-constrained where possible. Invalid outputs should trigger a limited repair prompt. Persistent invalidity should count as a protocol failure and be scored.

## Clue legality

The benchmark must define and enforce clue legality. At minimum, clues may not be one of the board words, a morphological variant of a board word, a multiword phrase if the rule set forbids it, or an explicit spelling/position clue. Exact Codenames clue legality has subjective edges, so the benchmark should support multiple rule profiles.

The recommended baseline is a strict automated legality profile: lowercase normalization, stemming or lemmatization, edit-distance checks, substring checks, and a fixed allowed-word dictionary. This reduces human judgment and makes results reproducible. A secondary permissive profile can be added later.

## Board generation

Use a stable word list and deterministic seed generation. Each board should contain 25 words: 9 starting-team words, 8 second-team words, 7 neutral words, and 1 assassin. The first-player advantage must be balanced across matchups. Board difficulty should be stratified so ratings are not dominated by unusually easy or pathological boards.

A strong approach is to maintain several board sets: canonical random boards, semantically dense boards with many related words, adversarial boards with near-neighbor traps, and held-out evaluation boards. The main Elo rating should use the canonical random board set, while diagnostic reports can break down performance by board type.

## Scoring

The primary outcome is game win or loss. Use this for Elo, Glicko, or TrueSkill ratings. Elo is simple and legible, but TrueSkill may be better for uncertainty estimates if the model pool changes often.

Secondary metrics should include clue efficiency, average targets captured per clue, accidental opponent hits, neutral hits, assassin hits, illegal clue rate, invalid JSON rate, stop-decision calibration, comeback rate, first-player-adjusted win rate, and exploitation of opponent clues. Opponent-clue exploitation can be approximated by measuring whether teams avoid words semantically close to the opponent’s public clues or correctly infer likely opponent-owned unrevealed words.

Do not reduce the benchmark to Elo alone. Elo tells you who wins; diagnostics tell you why.

## Tournament design

For N models, run every ordered matchup with both color assignments. A practical baseline is 200 mirrored board pairs per matchup, producing 400 games per unordered pair. For early development, use 20 mirrored pairs per matchup to keep costs controlled.

Use fixed seeds and store every event. A complete game record should include model name, provider, model version if available, temperature, prompts, board seed, board words, hidden key, public event log, private spymaster outputs, guesser outputs, final actions, legality decisions, token counts, latency, cost estimate, and terminal result.

Temperature should be standardized. A good default is temperature 0.7 for clue generation and 0.2 to 0.4 for structured deduction, but the benchmark should report the exact settings. If the goal is model comparison rather than product optimization, keep settings fixed across models unless a provider requires different sampling semantics.

## Baselines

The benchmark needs non-LLM and weak-agent baselines. Include random legal clue agents, embedding-similarity spymasters, embedding-similarity guessers, and simple stop rules. These baselines help identify whether LLMs are genuinely coordinating or merely benefiting from obvious word similarity.

An especially useful baseline is an oracle-clue ablation where the guessers receive handcrafted or algorithmically high-quality clues. This separates clue-generation quality from clue-interpretation quality.

## Reproducibility and fairness

Pin prompts, rule profile, word list, model IDs, provider versions where possible, seeds, and sampling parameters. Cache raw model responses. Never silently retry a game with a new seed after a model failure; record failures and apply a defined penalty. Use the same boards across all matchups when feasible.

Provider-specific outages and rate limits should be tracked separately from model behavior. If a call fails due to infrastructure, retry according to a documented retry policy. If the model produces invalid or illegal game actions after repair attempts, score that as model behavior.

## Risks and mitigations

The largest design risk is ambiguity in clue legality. Mitigate this by starting with an automated strict rule profile and publishing the legality code.

The second risk is cost. Mitigate with a deterministic simulator, mock agents, small smoke tournaments, response caching, and staged expansion.

The third risk is prompt leakage between private and public channels. Mitigate with typed observation objects, explicit role prompts, tests that assert spymaster-only fields never appear in guesser prompts, and full event-log auditing.

The fourth risk is collusion through hidden reasoning if using models that preserve conversation state across games. Mitigate by making every game stateless unless persistent memory is an explicit experimental condition.

## Minimal viable benchmark

The minimal viable benchmark should include a deterministic game engine, a board generator, strict clue legality checks, a JSON agent interface, mock agents, one LLM adapter, a local tournament runner, and an Elo report. It should run a tiny tournament between mock agents in seconds and a small tournament between two real models with cached outputs.

Only after that should the project expand to many providers, elaborate strategy analysis, and large round-robin runs.
