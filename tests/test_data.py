import json

from slime.utils.data import Dataset


class _Tokenizer:
    def apply_chat_template(self, prompt, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "rendered<|im_start|>assistant\n<think>\n"

    def __call__(self, prompts, add_special_tokens=False):
        return {"input_ids": [[1] for _ in prompts]}


def test_disabled_thinking_removes_ignored_template_marker(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps({"query": "solve", "answer": "label"}) + "\n", encoding="utf-8")

    dataset = Dataset(
        str(path),
        _Tokenizer(),
        None,
        max_length=None,
        prompt_key="query",
        label_key="answer",
        apply_chat_template=True,
        apply_chat_template_kwargs={"enable_thinking": False},
    )

    assert dataset.samples[0].prompt == "rendered<|im_start|>assistant\n"
