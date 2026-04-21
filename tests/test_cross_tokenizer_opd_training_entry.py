from __future__ import annotations

import torch

from slime.utils import opd_utils


class FakeTokenizer:
    def __init__(self, token_map):
        self.token_map = token_map
        self.inverse_map = {text: token_id for token_id, text in token_map.items()}

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "PAB":
            return [1, 2, 3] if self.inverse_map.get("P") == 1 else [10, 11]
        if text == "P":
            return [1] if self.inverse_map.get("P") == 1 else [10]
        if text == "AB":
            return [2, 3] if self.inverse_map.get("A") == 2 else [11]
        raise KeyError(text)

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(self.token_map[token_id] for token_id in token_ids)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens
        input_ids = self.encode(text)
        result = {"input_ids": input_ids}
        if return_offsets_mapping:
            if text == "PAB":
                result["offset_mapping"] = [(0, 1), (1, 2), (2, 3)] if len(input_ids) == 3 else [(0, 1), (1, 3)]
            elif text == "AB":
                result["offset_mapping"] = [(0, 1), (1, 2)] if len(input_ids) == 2 else [(0, 2)]
            elif text == "P":
                result["offset_mapping"] = [(0, 1)]
        return result


def test_prepare_byte_chunk_training_entry_exists_and_validates_one_sample():
    assert hasattr(opd_utils, "prepare_byte_chunk_training_entry")

    student_tokenizer = FakeTokenizer({1: "P", 2: "A", 3: "B"})
    teacher_tokenizer = FakeTokenizer({10: "P", 11: "AB"})

    result = opd_utils.prepare_byte_chunk_training_entry(
        full_text="PAB",
        prompt_text="P",
        student_token_ids=[1, 2, 3],
        response_token_count=2,
        student_log_probs=torch.tensor([-0.2, -0.3]),
        teacher_log_probs=torch.tensor([-0.4]),
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        recorded_student_response_bytes=[65, 66],
        recorded_student_token_byte_spans=[[0, 1], [1, 2]],
    )

    assert result["status"] == "ok"
    assert result["prompt_token_count"] == 1
    assert result["response_token_count"] == 2
    assert torch.allclose(result["student_chunk_log_probs"], torch.tensor([-0.25, -0.25]))
    assert torch.allclose(result["teacher_chunk_log_probs"], torch.tensor([-0.2, -0.2]))
    assert torch.allclose(result["reverse_kl"], torch.tensor([-0.05, -0.05]))
