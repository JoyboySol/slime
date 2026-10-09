import json
import os
from pathlib import Path

from sglang import Engine
from slime.rollout.rm_hub.math_dapo_utils import compute_score
from transformers import AutoTokenizer


ROOT = Path("/mnt/yulan/lvzhihao/PostTrain/data/hf_data/math_rl_mid_high_20k_20260924")
MODEL = Path("/mnt/yulan/lvzhihao/PostTrain/outputs/yulan_moe_math_dapo_full_cp8_rebuild_20260914/hf/rollout_67")
INPUT = ROOT / os.environ.get("MATH_EVAL_INPUT", "eval_100.jsonl")
RUN_TAG = os.environ.get("MATH_EVAL_RUN_TAG", "eval")
SAMPLES_PER_QUERY = int(os.environ.get("MATH_EVAL_SAMPLES", "2"))


def main():
    os.environ.setdefault("SGLANG_EXTERNAL_MODEL_PACKAGE", "sglang_qwen3_next_plugin")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    rows = [json.loads(line) for line in INPUT.open()]
    worker_id = int(os.environ.get("MATH_EVAL_WORKER_ID", "0"))
    worker_count = int(os.environ.get("MATH_EVAL_WORKER_COUNT", "1"))
    rows = [row for row in rows if row["eval_id"] % worker_count == worker_id]
    prompts = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": row["query"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.extend([prompt] * SAMPLES_PER_QUERY)

    engine = Engine(
        model_path=str(MODEL),
        trust_remote_code=True,
        tp_size=2,
        ep_size=2,
        dtype="bfloat16",
        mem_fraction_static=0.7,
        json_model_override_args='{"moe_router_sqrt_gate":true}',
        max_running_requests=32,
        log_level="warning",
    )
    try:
        outputs = engine.generate(
            prompts,
            sampling_params={"temperature": 0.7, "top_p": 1.0, "max_new_tokens": 8192},
        )
    finally:
        engine.shutdown()

    results = []
    for i, row in enumerate(rows):
        generations = []
        for attempt in range(SAMPLES_PER_QUERY):
            response = outputs[SAMPLES_PER_QUERY * i + attempt]["text"]
            score = compute_score(response, row["answer"], prompt=row["query"])
            generations.append({"response": response, **score})
        results.append({**row, "generations": generations})
    output = ROOT / f"{RUN_TAG}_worker_{worker_id}.jsonl"
    with output.open("w") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    total_correct = sum(g["acc"] for row in results for g in row["generations"])
    avg_accuracy = total_correct / (SAMPLES_PER_QUERY * len(results))
    pass_at_n = sum(any(g["acc"] for g in row["generations"]) for row in results) / len(results)
    valid_group_ratio = sum(
        any(g["acc"] for g in row["generations"]) and not all(g["acc"] for g in row["generations"])
        for row in results
    ) / len(results)
    summary = {
        "model": str(MODEL),
        "num_questions": len(results),
        "samples_per_question": SAMPLES_PER_QUERY,
        "total_responses": SAMPLES_PER_QUERY * len(results),
        "correct_responses": int(total_correct),
        "avg_accuracy": avg_accuracy,
        f"pass_at_{SAMPLES_PER_QUERY}": pass_at_n,
        "valid_group_count": int(round(valid_group_ratio * len(results))),
        "valid_group_ratio": valid_group_ratio,
        "result_file": str(output),
    }
    (ROOT / f"{RUN_TAG}_worker_{worker_id}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
