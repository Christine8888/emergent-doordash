from __future__ import annotations

import unittest

from src.token_budget import count_prompt_tokens_with_tokenizer, extract_context_limit_from_error


class _TemplateTokenizer:
    def __init__(self, tokenized: object) -> None:
        self.tokenized = tokenized

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> object:
        assert messages
        assert tokenize is True
        assert add_generation_prompt is True
        return self.tokenized

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        raise AssertionError("chat-template path should not fall back to encode")


class _ObjectWithInputIds:
    input_ids = [1, 2, 3, 4]


class _Encoding:
    ids = [1, 2, 3, 4, 5]


class _EncodeOnlyTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert text == "hello"
        assert add_special_tokens is True
        return [1, 2]


def _count(tokenized: object) -> int:
    return count_prompt_tokens_with_tokenizer(
        _TemplateTokenizer(tokenized),
        messages=[{"role": "user", "content": "hello"}],
        prompt_text="hello",
    )


class TokenBudgetTest(unittest.TestCase):
    def test_count_prompt_tokens_reads_mapping_input_ids(self) -> None:
        tokenized = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        self.assertEqual(_count(tokenized), 3)

    def test_count_prompt_tokens_reads_batched_mapping_input_ids(self) -> None:
        tokenized = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
        self.assertEqual(_count(tokenized), 3)

    def test_count_prompt_tokens_reads_object_input_ids(self) -> None:
        self.assertEqual(_count(_ObjectWithInputIds()), 4)

    def test_count_prompt_tokens_reads_encoding_ids(self) -> None:
        self.assertEqual(_count(_Encoding()), 5)

    def test_count_prompt_tokens_falls_back_to_encode_without_chat_template(self) -> None:
        self.assertEqual(
            count_prompt_tokens_with_tokenizer(
                _EncodeOnlyTokenizer(),
                prompt_text="hello",
            ),
            2,
        )

    def test_extract_context_limit_from_vllm_context_error(self) -> None:
        error_text = (
            "This model's maximum context length is 32768 tokens. "
            "However, you requested 31486 output tokens and your prompt contains "
            "at least 1283 input tokens."
        )
        self.assertEqual(extract_context_limit_from_error(error_text), 32768)


if __name__ == "__main__":
    unittest.main()
