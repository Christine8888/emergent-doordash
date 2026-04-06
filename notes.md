**April 2nd**
I have two types of inference: 
- generating hints wih Claude
- using hints, generate answers for many other models
I need to seperate this inference

Some TODO's:
- experiment with ether truncating hints, or try to make some better hints (for AIME 2025, 2026 for example)
- set up nice infra to do small scale experiments of generating answers for models and making scaling laws
    - use this to make mini scaling law for truncated hints
    - get together API key
    - use EleutherAI framework?
- prep meeting doc


**Proposed Eval Setup: hinted**
For small-scale experiments:
- Evaluate on AIME 2025+2026 (60 problems), 5 rollouts, only masked inference (?), 10 hint levels = 60 * 5 * 10 = 3000 inference requests per model
    - enough problems/rollouts?

3000 * 20k = 60 million tokens. 10k tokens/second --> 


- Evaluate on a set of models (which ones? want a wide variery of ECIs)
    - Kimi, Qwen, GPT, Claude, Llama, Gemma, etc.
    - not all models served on Together (eg Llama 3.1 8B)

Currently used:                             Available on together? (under 'serverless pricing')
    "Qwen/Qwen3-0.6B"                       No
    "Qwen/Qwen3-1.7B"                       No
    "Qwen/Qwen3-4B"                         No
    "Qwen/Qwen3-8B"                         No
    "Qwen/Qwen3-14B"                        No
    "Qwen/Qwen3-32B"                        No
    "Qwen/Qwen2.5-1.5B-Instruct"            No
    "Qwen/Qwen2.5-3B-Instruct"              No
    "Qwen/Qwen2.5-7B-Instruct"              No
    "Qwen/Qwen2.5-14B-Instruct"             No
    "Qwen/Qwen2.5-32B-Instruct"             No
    "meta-llama/Llama-3.1-8B-Instruct"      No
    "meta-llama/Llama-3.1-70B-Instruct"     No
    "google/gemma-3-4b-it"                  No
    "google/gemma-3-12b-it"                 No
    "google/gemma-3-27b-it"                 No

Some newer ones (which are better, which I'm not sure we want)


I seem to be able to deploy models on Together? Even for the ones that are not available under serverless. 
Not sure how much better that is than just running it on the cluster then

Am I looking in the wrong place then? Maybe we use the batch API instead

What is the best way to use Together?

I could do some experimentation on truncated hints, maybe just locally?




**April 3rd**
I need a consistent and modularized pipeline for hinting.
We have the following stages
1. hint generation (with claude)
- hint_type: what type of hint?
    - ideas: truncated, masked, bag of hints
- rollout_id: eg if we generate 10 hints per problem, they should all have a different id
- problem_id
- model: generated with which model
- problem
- answer
- full_hint
- benchmark_name
- time_created
These should all be written to one file and named by the hint_types used

2. Hint checking
- make sure that hint does not leak the answer
- measure behavior at different hint fractions
- I would like a streamlit viewer for this; potentially one viewer that looks at all the hints of one hint_type and another that looks at all types of hints per problem_id

3. Hinted inference (with inspect_ai; either local or through API)
- model
- hint_fraction
- [all info saved in hint generation]
- time_created
- model_output
- per extractor_grader:
    - extractor_grader_type
    - extracted_answer
    - is_correct
This inference can (andmaybe should?) be in a completely different file than the other inference, though they can use the same helper functions

4. Plotting/analysis
[todo]
to make this easier, we should ideally save one file per model+hint_type+hint_fraction s.t. the file is not too big and easy to use with a data viewer.

General
- we should save all the data in subfolders in a data/ folder


[22:35:45] [hint_generation] dry_run missing_rollouts problem_id=aime2025_2026_0045 missing_count=10 rollout_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]