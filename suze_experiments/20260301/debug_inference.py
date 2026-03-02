# aime_experinents(run_experiment) --> submitit_utils(launch_experiment) --> submitit_utils(run_single_experiment) --> base_experiment(run) --> base_experiment(eval, from inspect_ai)
# inspect eval --help

"""
(emergent_doordash) suzeva@sphinx2:/nlp/scr/suzeva/projects/emergent-doordash$ inspect eval --help
Usage: inspect eval [OPTIONS] [TASKS]...

  Evaluate tasks.

Options:
  --model TEXT                    Model used to evaluate tasks.
  --model-base-url TEXT           Base URL for for model API
  -M TEXT                         One or more native model arguments (e.g. -M
                                  arg=value)
  --model-config TEXT             YAML or JSON config file with model
                                  arguments.
  --model-role TEXT               Named model role with model name or
                                  YAML/JSON config, e.g. --model-role
                                  critic=openai/gpt-4o or --model-role
                                  grader="{model: mockllm/model, temperature:
                                  0.5}"
  -T TEXT                         One or more task arguments (e.g. -T
                                  arg=value)
  --task-config TEXT              YAML or JSON config file with task
                                  arguments.
  --solver TEXT                   Solver to execute (overrides task default
                                  solver)
  -S TEXT                         One or more solver arguments (e.g. -S
                                  arg=value)
  --solver-config TEXT            YAML or JSON config file with solver
                                  arguments.
  --tags TEXT                     Tags to associate with this evaluation run.
  --metadata TEXT                 Metadata to associate with this evaluation
                                  run (more than one --metadata argument can
                                  be specified).
  --approval TEXT                 Config file for tool call approval.
  --sandbox TEXT                  Sandbox environment type (with optional
                                  config file). e.g. 'docker' or
                                  'docker:compose.yml'
  --no-sandbox-cleanup            Do not cleanup sandbox environments after
                                  task completes
  --limit TEXT                    Limit samples to evaluate e.g. 10 or 10-20
  --sample-id TEXT                Evaluate specific sample(s) (comma separated
                                  list of ids)
  --sample-shuffle TEXT           Shuffle order of samples (pass a seed to
                                  make the order deterministic)
  --epochs INTEGER                Number of times to repeat dataset (defaults
                                  to 1)
  --epochs-reducer TEXT           Method for reducing per-epoch sample scores
                                  into a single score. Built in reducers
                                  include 'mean', 'median', 'mode', 'max', and
                                  'at_least_{n}'.
  --no-epochs-reducer             Do not reduce per-epoch sample scores.
  --max-connections INTEGER       Maximum number of concurrent connections to
                                  Model API (defaults to 10)
  --max-retries INTEGER           Maximum number of times to retry model API
                                  requests (defaults to unlimited)
  --timeout INTEGER               Model API request timeout in seconds
                                  (defaults to no timeout)
  --attempt-timeout INTEGER       Timeout (in seconds) for any given attempt
                                  (if exceeded, will abandon attempt and retry
                                  according to max_retries).
  --max-samples INTEGER           Maximum number of samples to run in parallel
                                  (default is running all samples in parallel)
  --max-tasks INTEGER             Maximum number of tasks to run in parallel
                                  (default is 1 for eval and 4 for eval-set)
  --max-subprocesses INTEGER      Maximum number of subprocesses to run in
                                  parallel (default is os.cpu_count())
  --max-sandboxes INTEGER         Maximum number of sandboxes (per-provider)
                                  to run in parallel.
  --message-limit INTEGER         Limit on total messages used for each
                                  sample.
  --token-limit INTEGER           Limit on total tokens used for each sample.
  --time-limit INTEGER            Limit on total running time for each sample.
  --working-limit INTEGER         Limit on total working time (e.g. model
                                  generation, tool calls, etc.) for each
                                  sample.
  --fail-on-error FLOAT           Threshold of sample errors to tolerage (by
                                  default, evals fail when any error occurs).
                                  Value between 0 to 1 to set a proportion;
                                  value greater than 1 to set a count.
  --no-fail-on-error              Do not fail the eval if errors occur within
                                  samples (instead, continue running other
                                  samples)
  --continue-on-fail              Do not immediately fail the eval if the
                                  error threshold is exceeded (instead,
                                  continue running other samples until the
                                  eval completes, and then possibly fail the
                                  eval).
  --retry-on-error TEXT           Retry samples if they encounter errors (by
                                  default, no retries occur). Specify --retry-
                                  on-error to retry a single time, or specify
                                  e.g. `--retry-on-error=3` to retry multiple
                                  times.
  --no-log-samples                Do not include samples in the log file.
  --no-log-realtime               Do not log events in realtime (affects live
                                  viewing of samples in inspect view)
  --log-images / --no-log-images  Include base64 encoded versions of filename
                                  or URL based images in the log file.
  --log-buffer INTEGER            Number of samples to buffer before writing
                                  log file. If not specified, an appropriate
                                  default for the format and filesystem is
                                  chosen (10 for most all cases, 100 for JSON
                                  logs on remote filesystems).
  --log-shared TEXT               Sync sample events to log directory so that
                                  users on other systems can see log updates
                                  in realtime (defaults to no syncing). If
                                  enabled will sync every 10 seconds (or pass
                                  a value to sync every `n` seconds).
  --no-score                      Do not score model output (use the inspect
                                  score command to score output later)
  --no-score-display              Do not score model output (use the inspect
                                  score command to score output later)
  --max-tokens INTEGER            The maximum number of tokens that can be
                                  generated in the completion (default is
                                  model specific)
  --system-message TEXT           Override the default system message.
  --best-of INTEGER               Generates best_of completions server-side
                                  and returns the 'best' (the one with the
                                  highest log probability per token). OpenAI
                                  only.
  --frequency-penalty FLOAT       Number between -2.0 and 2.0. Positive values
                                  penalize new tokens based on their existing
                                  frequency in the text so far, decreasing the
                                  model's likelihood to repeat the same line
                                  verbatim. OpenAI, Google, Grok, Groq, llama-
                                  cpp-python and vLLM only.
  --presence-penalty FLOAT        Number between -2.0 and 2.0. Positive values
                                  penalize new tokens based on whether they
                                  appear in the text so far, increasing the
                                  model's likelihood to talk about new topics.
                                  OpenAI, Google, Grok, Groq, llama-cpp-python
                                  and vLLM only.
  --logit-bias TEXT               Map token Ids to an associated bias value
                                  from -100 to 100 (e.g. "42=10,43=-10").
                                  OpenAI, Grok, and Grok only.
  --seed INTEGER                  Random seed. OpenAI, Google, Groq, Mistral,
                                  HuggingFace, and vLLM only.
  --stop-seqs TEXT                Sequences where the API will stop generating
                                  further tokens. The returned text will not
                                  contain the stop sequence.
  --temperature FLOAT             What sampling temperature to use, between 0
                                  and 2. Higher values like 0.8 will make the
                                  output more random, while lower values like
                                  0.2 will make it more focused and
                                  deterministic.
  --top-p FLOAT                   An alternative to sampling with temperature,
                                  called nucleus sampling, where the model
                                  considers the results of the tokens with
                                  top_p probability mass.
  --top-k INTEGER                 Randomly sample the next word from the top_k
                                  most likely next words. Anthropic, Google,
                                  HuggingFace, and vLLM only.
  --num-choices INTEGER           How many chat completion choices to generate
                                  for each input message. OpenAI, Grok,
                                  Google, TogetherAI, and vLLM only.
  --logprobs                      Return log probabilities of the output
                                  tokens. OpenAI, Google, TogetherAI,
                                  Huggingface, llama-cpp-python, and vLLM
                                  only.
  --top-logprobs INTEGER          Number of most likely tokens (0-20) to
                                  return at each token position, each with an
                                  associated log probability. OpenAI, Google,
                                  TogetherAI, Huggingface, and vLLM only.
  --parallel-tool-calls / --no-parallel-tool-calls
                                  Whether to enable parallel function calling
                                  during tool use (defaults to True) OpenAI
                                  and Groq only.
  --internal-tools / --no-internal-tools
                                  Whether to automatically map tools to model
                                  internal implementations (e.g. 'computer'
                                  for anthropic).
  --max-tool-output INTEGER       Maximum size of tool output (in bytes).
                                  Defaults to 16 * 1024.
  --cache-prompt [auto|true|false]
                                  Cache prompt prefix (Anthropic only).
                                  Defaults to "auto", which will enable
                                  caching for requests with tools.
  --verbosity [low|medium|high]   Constrains the verbosity of the model's
                                  response. Lower values will result in more
                                  concise responses, while higher values will
                                  result in more verbose responses. GPT 5.x
                                  models only (defaults to "medium" for OpenAI
                                  models)
  --effort [low|medium|high]      Control how many tokens are used for a
                                  response, trading off between response
                                  thoroughness and token efficiency. Anthropic
                                  Claude 4.5 Opus only.
  --reasoning-effort [none|minimal|low|medium|high|xhigh]
                                  Constrains effort on reasoning. Defaults
                                  vary by provider and model and not all
                                  models support all values (please consult
                                  provider documentation for details).
  --reasoning-tokens INTEGER      Maximum number of tokens to use for
                                  reasoning. Anthropic Claude models only.
  --reasoning-summary [none|concise|detailed|auto]
                                  Provide summary of reasoning steps (OpenAI
                                  reasoning models only). Use 'auto' to access
                                  the most detailed summarizer available for
                                  the current model (defaults to 'auto' if
                                  your organization is verified by OpenAI).
  --reasoning-history [none|all|last|auto]
                                  Include reasoning in chat message history
                                  sent to generate (defaults to "auto", which
                                  uses the recommended default for each
                                  provider)
  --response-schema TEXT          JSON schema for desired response format
                                  (output should still be validated). OpenAI,
                                  Google, and Mistral only.
  --cache TEXT                    Policy for caching of model generations.
                                  Specify --cache to cache with 7 day
                                  expiration (7D). Specify an explicit
                                  duration (e.g. (e.g. 1h, 3d, 6M) to set the
                                  expiration explicitly (durations can be
                                  expressed as s, m, h, D, W, M, or Y).
                                  Alternatively, pass the file path to a YAML
                                  or JSON config file with a full
                                  `CachePolicy` configuration.
  --batch TEXT                    Batch requests together to reduce API calls
                                  when using a model that supports batching
                                  (by default, no batching). Specify --batch
                                  to batch with default configuration,
                                  specify a batch size e.g. `--batch=1000` to
                                  configure batches of 1000 requests, or pass
                                  the file path to a YAML or JSON config file
                                  with batch configuration.
  --log-format [eval|json]        Format for writing log files.
  --log-level-transcript [debug|trace|http|info|warning|error|critical|notset]
                                  Set the log level of the transcript
                                  (defaults to 'info')
  --log-level [debug|trace|http|info|warning|error|critical|notset]
                                  Set the log level (defaults to 'warning')
  --log-dir TEXT                  Directory for log files.
  --display [full|conversation|rich|plain|log|none]
                                  Set the display type (defaults to 'full')
  --traceback-locals              Include values of local variables in
                                  tracebacks (note that this can leak private
                                  data e.g. API keys so should typically only
                                  be enabled for targeted debugging).
  --env TEXT                      Define an environment variable e.g. --env
                                  NAME=value (--env can be specified multiple
                                  times)
  --debug                         Wait to attach debugger
  --debug-port INTEGER            Port number for debugger
  --debug-errors                  Raise task errors (rather than logging them)
                                  so they can be debugged.
  --help                          Show this message and exit.
"""

# important config changes
# max_connections
# bath size
# checkpointed or not
# timeout
# ...


# evaluation setup:
"""
eval() orchestrator
  └─ for each (sample, epoch):
       1. instructions()  → mutate prompt text
       2. intext()        → append masked hint to prompt
       3. generate()      → HTTP POST to vLLM at localhost:{port}/v1
            └─ GenerateConfig(max_tokens=8192, temperature=0.6, top_p=0.95, ...)
       4. aime_scorer()   → grade the response via sympy exact match
lots of state to keep track of!
"""

"""
Next steps
1. set up basic eval framework
2. explore using different configs (see above)
2. collect questions separately and do inference like luke
"""



