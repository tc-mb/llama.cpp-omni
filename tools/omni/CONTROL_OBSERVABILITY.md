# Observing interaction-control decisions

Text perplexity alone does not describe when a duplex model listens, speaks, or
ends a turn. A difference between two runs can also come from `listen_prob_scale`,
length penalties, a forbidden token, or the sampler rather than the weights.
The optional `omni_context::decode_observer` exposes these boundaries for the
LLM sampling helper used by duplex decoding. It is disabled by default.

## Contract

Install or remove the observer only while the context is idle, before starting
decode workers. It runs synchronously on the decoding thread. Do not mutate the
context, re-enter inference, or throw from the callback. Copy any data you retain;
the event and logits pointers are borrowed for the callback only. Capture objects
whose lifetimes cover every callback, including worker shutdown.

Each sampled step emits, in order:

1. `raw_logits`: full-vocabulary logits before Omni's control adjustments.
2. `adjusted_logits`: after LISTEN bias, the TTS-pad mask, and the applicable
   EOS length penalty, **before** the common sampler chain. These are not the
   final temperature/top-k/top-p/repetition-penalized sampling probabilities.
3. `sampled`: the actual accepted token, before evaluating that token. Its logits
   pointer is null and `n_vocab` is zero. A sample does not establish successful
   evaluation, transmission, or audible playback.

If the backend returns no logits, only `sampled` is emitted. `n_past` records the
position before sampling; sliding windows can reuse positions, so it is not an
event ID. Assign monotonically increasing step IDs in the recorder and associate
them with the caller's session, frame, input, and prefix metadata.

This hook covers calls to `sample_with_hidden_and_token`. Forced startup LISTEN
decisions bypass sampling and are not observed. It does not instrument TTS
sampling, legacy `sample` helpers, or expose hidden states. It is not a complete
interaction trace or a teacher-forcing implementation.

For example, an embedding application can copy observations for later grouping:

```cpp
struct observed_event {
    omni_decode_phase phase;
    int position = 0;
    std::vector<float> logits;
    llama_token token = -1;
};
std::vector<observed_event> events; // must outlive all decoding callbacks
ctx->decode_observer = [&events](const omni_decode_observation & e) {
    observed_event copy{e.phase, e.n_past, {}, e.token};
    if (e.logits && e.n_vocab > 0) {
        copy.logits.assign(e.logits, e.logits + e.n_vocab);
    }
    events.push_back(std::move(copy));
};
// Start and join the decoding workers before destroying `events` or clearing ctx.
```

The consumer should detect missing phases and discard incomplete records rather
than attaching a token to an earlier step. Copying every vocabulary vector has
substantial storage and latency cost. Use bounded storage or compute aggregates
inside the callback, and benchmark latency separately with the observer disabled.

## Minimum experiment before a control-aware quantization claim

The proposed hypothesis is that preserving interaction decisions can be a more
useful calibration objective than preserving average text loss alone. This hook
does not establish that hypothesis or implement a quantization method.

- Freeze model/converter revisions, tokenizer and control-token IDs, input
  audio/video frames, system prompt, prefix tokens, context/window settings,
  control biases, sampler configuration, and seeds. Record their hashes.
- Compare an unquantized reference with the candidate on the **same input and
  prefix**. After independent generation diverges, differences at matching step
  numbers are not same-state quantization errors. A replay/teacher-forcing harness
  is required for this comparison and is not supplied by the hook.
- Compute a stable full-vocabulary softmax at the raw and adjusted phases. Keep
  probabilities for LISTEN, SPEAK, and TURN_EOS plus the sum of all OTHER tokens.
  Do not renormalize only control tokens: that hides loss of mass to other tokens.
  Validate finite logits while allowing deliberate negative-infinity masks.
- Report control-decision disagreement and grouped JS divergence, stratified by
  speech onset, overlap, silence, and turn ending; include sample counts and
  uncertainty. Inspect rare but consequential changes separately from averages.
- Keep a standard quantization baseline at matched memory/size, and compare any
  proposed calibration scheme on held-out speakers and acoustic conditions.
- Separately test free interaction: interruption success, premature ending,
  response onset, playback gaps, and audio quality. Text quality and TTS
  conditioning need their own controls; this observer exposes no hidden states.

Stop or revise the hypothesis if differences vanish under matched prefixes, if
runtime bias/sampling explains them, or if decision improvements do not improve
held-out interaction outcomes at comparable memory and latency. No real-model
F16/Q4 comparison is claimed by the model-free contract test below.

## Model-free contract check

```bash
python3 -m unittest discover -s tools/omni/test -p 'test_decode_observer.py' -v
```

The test compiles the actual production helper and observation declaration with
deterministic backend stubs under ASan/UBSan. It checks callback ordering,
pre-adjustment copies, duplex/simplex adjustments, disabled special tokens, null
logits, evaluation failure, EOG output, and observer-on/off state equivalence.
It does not run a GGUF model, a real sampler chain, or concurrent inference.
