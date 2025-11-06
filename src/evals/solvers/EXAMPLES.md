# Solver Composition Examples

## How Solvers Work

Each solver transforms the `TaskState` sequentially. They execute in order.

## Concrete Example: Task State Flow

**Setup:**
```python
from evals.solvers import instructions, fewshot, prefill, generate
from evals.fewshot import FewShotConfig
from evals.prefill import PrefillConfig

fewshot_config = FewShotConfig(
    path="examples.jsonl",
    num_examples=2,
    prefix="Examples:",
    suffix="Your turn:"
)
prefill_config = PrefillConfig(path="hints.jsonl", fraction=0.5)
```

**Solver chain:**
```python
solver = [
    instructions("Answer the question."),
    fewshot(fewshot_config),
    prefill(prefill_config),
    generate(timeout=600)
]
```

### State Transformations

**Initial state (from dataset):**
```
state.user_prompt.text = "What is the capital of France?\nA) Paris\nB) London"
state.messages = []
```

**After `instructions("Answer the question.")`:**
```
state.user_prompt.text = "Answer the question.\n\nWhat is the capital of France?\nA) Paris\nB) London"
state.messages = []
```

**After `fewshot(fewshot_config)`:**
```
state.user_prompt.text = "Answer the question.\n\nWhat is the capital of France?\nA) Paris\nB) London\n\nExamples:\n\nQ: What is 1+1?\nA: 2\n\nQ: What is 2+2?\nA: 4\n\nYour turn:"
state.messages = []
```

**After `prefill(prefill_config)`:**
```
state.user_prompt.text = [unchanged]
state.messages = [
    ChatMessageUser(content="Answer the question.\n\nWhat is the capital of France?...[full prompt]"),
    ChatMessageAssistant(content="Let me think step by step. France is a country in")  # 50% of hint
]
```

**After `generate(timeout=600)`:**
```
state.output.completion = " Europe with Paris as its capital. ANSWER: A"
```

## Common Patterns

### Basic Evaluation (no hints)
```python
solver = [
    instructions(DEFAULT_INSTRUCTIONS),
    generate()
]
```

### With Prefill Hints
```python
solver = [
    instructions(DEFAULT_INSTRUCTIONS),
    prefill(PrefillConfig(path="hints.jsonl", fraction=0.8)),
    generate()
]
```

### With Few-shot + Prefill
```python
solver = [
    instructions(DEFAULT_INSTRUCTIONS),
    fewshot(FewShotConfig(path="examples.jsonl", num_examples=3)),
    prefill(PrefillConfig(path="hints.jsonl", fraction=0.5)),
    generate()
]
```

### Custom Ordering (advanced)
```python
# Put examples before problem (unusual but possible)
solver = [
    fewshot(FewShotConfig(path="examples.jsonl", suffix="Now solve:")),
    instructions(DEFAULT_INSTRUCTIONS),  # This prepends, so ends up first
    prefill(config),
    generate()
]
# Result: [Instructions][Examples + suffix][Problem][Prefill]
```

## Key Points

1. **`instructions()`** - Prepends to `state.user_prompt.text`
2. **`fewshot()`** - Appends to `state.user_prompt.text`
3. **`prefill()`** - Adds assistant message to `state.messages`
4. **`generate()`** - Generates completion, auto-detects continuation if last message is assistant

**Order matters!** Think about execution order to get the prompt structure you want.
