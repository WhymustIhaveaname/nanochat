
## 环境配置

`.venv` 下有一个虚拟环境，使用 `source .venv/bin/activate` 激活。这个虚拟环境是用 `uv sync --extra gpu --extra dev --package light-core` 创建的。

## Code Style

### For Loops Must Not Nest More Than 3 Levels, Function Calls Must Not Recurse More Than 2 Levels

### No Defensive Programming
- Avoid unnecessary safety checks and validations
- Let code fail fast when preconditions are not met. Let exceptions bubble up naturally when appropriate
- Use try-catch blocks sparingly, only when absolutely necessary
- **NEVER** use `dict.get()` with default values - use direct dictionary access: `config['key']`

### Self-Documenting Code Over Comments
- **Good code doesn't need comments** - if control flow is clear and variables are well-named, the code should be self-explanatory
- Reading the code should be easier than reading comments
- However, if your code is complex or unclear, comments are better than no documentation
- Prioritize: Clean Code > Commented Code > Uncommented Complex Code

### Minimize Code - Write Only What's Necessary to Complete the Task, Nothing More

### Import Organization
- **All imports must be placed at the beginning of the file** - imports are not allowed in the middle of code

### Ask User Before Execute
- AI-generated code is often low-quality and cannot be executed directly, so please ask the user before execution and only execute after the user agrees
- When you become confused or uncertain while coding, ask the user for guidance or which approach is better - don't just keep writing blindly

### Do Not Modify User's Code Without Permission
If you notice the latest code differs from the version in context (e.g., reverted or modified), it means the user has made changes or was unsatisfied with your code and undid it. Do NOT revert their changes or restore your previous code. Focus on the current task instead of arguing with the user.

### Reuse Before Reinvent
- Before implementing new functionality, always search the codebase for existing code that can be reused or adapted
- Avoid duplicating logic that already exists elsewhere in the repository

### Minimize Token Usage
- For large copy-paste operations, use the search/replace tool instead of generating the entire content
- For file renaming, use `mv` command instead of generating a new file
- Maximize efficiency to save tokens and user's time

### When Uncertain, Ask First
- If you encounter something unclear or are unsure how to implement it, ask the user immediately - clarify everything before writing code

### No Fake Unit Tests
- Fake UTs are meaningless unit tests, typically falling into two categories: paranoid assertions that can never fail, and tautological assertions
- To identify tautological assertions: ask yourself, if I intentionally break the code, would this assertion catch it?

### Unit Tests Must Test Real Code, Not Self-Defined Rules
- Unit tests should test actual functions/classes in the codebase as black boxes
- **WRONG**: Creating tests that define your own rules and then test those rules (self-entertainment)
- **CORRECT**: End-to-end tests that exercise the real function with real inputs and expected outputs
- Example of bad practice:
  ```python
  # BAD: Testing your own interpretation of rules, not the actual code
  class TestRule1_ShortNameMatch:
      def test_exact_match(self):
          # This tests YOUR understanding of rule 1, not the actual is_equal() function
          assert "mod11c3" in "mod11c3 modis terra"  # self-defined logic

  # GOOD: Test the actual function end-to-end
  class TestIsEqual:
      def test_positive_cases(self):
          result = {"short_name": "MOD11C3", "title": "MODIS Terra LST"}
          assert is_equal(result, "MOD11C3 MODIS/Terra LST")  # actual function
  ```

### Define Variables Close to Their Usage
- Variable definitions should be placed immediately before where they are used
- Avoid inserting unrelated code between a variable's definition and its usage

### Inline Single-Use Variables
- Variables used only once should be inlined, unless the variable name adds semantic clarity
- If the expression is complex or non-obvious, a descriptive variable name serves as self-documentation

## Code Style Examples

```python
# non-informative comment
# Import the correct class and functions
from vla.data_pipeline.preprocess.tokenizer_youtube import YouTubeTokenizer
from vla.data_pipeline.preprocess.sequence_tokenize import tokenize, tokenize_ray

# non-informative comment
# Compute metrics
metrics = {
    "l1_error": torch.mean(l1_error).item(),
    "l2_error": torch.mean(l2_error).item(),
    "l1_error_weighted": torch.mean(weighted_l1).item(),
    "l2_error_weighted": torch.mean(weighted_l2).item(),
    "num_frames": len(l1_error),
}

# non-informative comment
def clear_conversation(self):
    """Clear conversation history."""
    self.conversations = []

# import inside code
class LightArgumentParser(transformers.HfArgumentParser):
    def parse_hydra_config(self, cfg):
        """Parse Hydra config to dataclass instances"""
        from omegaconf import DictConfig
        model_args = ModelArguments(**cfg.model)
        data_args = DataArguments(**cfg.data)
        training_args = TrainingArguments(**cfg.training)
        return model_args, data_args, training_args

# wrong: too many checks
[arg for arg in sys.argv[2:] if "=" in arg]
# correct: simple and pop out error
sys.argv[2:]

# totally wrong: defense programming using wrong parameters and user doees not even know
if torch.distributed.is_initialized():
    world_size = torch.distributed.get_world_size()
else:
    world_size = 1

# wrong: too many checks and hide the error
# correct way: this if is necessary, del it
if arg.startswith("--") and i + 1 < len(sys.argv):
    key = arg[2:]  # 去掉 '--'
    value = sys.argv[i + 1]

# wrong: useless if, complex control flow
if i == j:
    distance_matrix[i, j] = 0.0
else:
    lev_distance = distance.Levenshtein.distance(texts[i], texts[j])
    max_len = max(len(texts[i]), len(texts[j]))
    similarity = 1.0 - (lev_distance / max_len) if max_len > 0 else 1.0
    distance_matrix[i, j] = similarity
    distance_matrix[j, i] = similarity

# wrong: low readbility + comment
tracks_queue = list(self.tracks)  # copy to pop from
# good: reading codes is easier than reading commnet
tracks_queue = self.tracks.copy()
```
