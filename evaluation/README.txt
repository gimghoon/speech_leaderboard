Korean Streaming EOT Pilot
=========================

Run the approved Korean-only, local evaluation from this folder:

  python -m eot_stream.cli \
    --source-root /absolute/path/to/minds-ko-pseudo-source \
    --output /absolute/path/to/minds-ko-pilot-stream-v1 \
    --cache-dir /absolute/path/to/stream-trace-cache \
    --dev-cache-dir /absolute/path/to/pinned-development-cache \
    --model-cache-dir /absolute/path/to/pinned-model-cache \
    --vad-path /absolute/path/to/pinned-silero.onnx

The default models are timer, smart-turn and livekit-mini. --models accepts an
explicit subset. Use the prepared Python3.12 environment and existing pinned
local weights. Set HF_HUB_OFFLINE=1 and HF_DATASETS_OFFLINE=1 for offline runs.
No training or external audio API is used.

The source and existing results are preserved. The output must be fresh or have
a matching run manifest. Repeating the same command resumes verified trace
caches. Changes to source/data/reference/protocol/model hashes fail closed.
A failed or interrupted run is never a complete result; inspect manifest.json.

All377 Korean LiveKit development turns are scored before policies for the5%
and10% premature budgets are selected, saved in frozen_policies.json, and then
applied unchanged to all160 Korean MInDS pilot turns. Both budgets also require
a development miss rate <=5%; infeasible fallback is reported explicitly.
Development/pilot ID and encoded source audio overlap is rejected.

All models, including the silence timer, use the same causal Silero VAD. The
timer never reads annotated pauses. Endpoint models receive only observed PCM
prefixes.16kHz mono frames are512samples; no incomplete final frame is padded
with future samples.8kHz pilot audio uses a causal FIR conversion; the1.25ms
filter delay is retained and reported. Each observation ends at reference
EOT+4seconds. Any required digital silence after the original end is appended
only in the inference buffer and its duration is disclosed in data.json.

Interpretation matters: public source data and pseudo annotations are not gold
accuracy. The reference construction and runtime VAD use the SAME Silero
weights, so this is circular and not an independent reference validation.
Latency is the first actual stop on the AUDIO timeline, excluding computation;
it is not wall-clock end-to-end latency. Successful-turn mean delay is
conditional; always read it with whole-denominator premature/missed rates and
the4000ms-per-error penalized mean delay. Model computation statistics are
separate, exclude loading/VAD/I/O and include all diagnostic trace calls even
after a policy's first declaration.

Outputs:
  report.txt           Korean plain-text report and limitations
  summary.csv/json     both budgets, development and pilot metrics for all models
  frozen_policies.json development policy, feasibility, settings, trace hashes
  outcomes/*.csv       first declaration and outcome per turn/model/split/budget
  traces/*/index.json  copied pilot score traces with IDs and SHA256 hashes
  data.json            per-turn references and provenance, no audio arrays
  data/                byte-identical source WAV snapshot, original annotations,
                       all342selection records, source/hash manifest
  manifest.json        running/failed/complete, counts, IDs, protocol, command,
                       source/data/model/VAD/result hashes

Use python -m pytest -q from this folder for the test suite. Integration tests
use tiny normalized fixtures and mocked local model/cache boundaries; they do
not download or invoke real endpoint models.
