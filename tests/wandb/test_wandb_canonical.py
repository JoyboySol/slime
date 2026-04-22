from slime.utils.wandb_canonical import RunHistorySegment, RuntimeStepAligner, drop_first_history_rows
from slime.utils.wandb_canonical import filter_loggable_history_row
from slime.utils.wandb_canonical import merge_run_histories, repair_low_step_rows, shift_step_rows


def test_merge_run_histories_auto_stitches_step_keys():
    merged = merge_run_histories(
        [
            RunHistorySegment(
                run_path="entity/project/bvx4hc7w",
                rows=[
                    {"train/step": 0, "train/loss": 1.0},
                    {"train/step": 1, "train/loss": 0.8},
                ],
            ),
            RunHistorySegment(
                run_path="entity/project/ctq3vfmo",
                rows=[
                    {"train/step": 0, "train/loss": 0.7},
                    {"train/step": 1, "train/loss": 0.6},
                ],
            ),
        ]
    )

    assert [row["train/step"] for row in merged.rows] == [0, 1, 2.0, 3.0]
    assert merged.per_run_offsets["bvx4hc7w"] == {}
    assert merged.per_run_offsets["ctq3vfmo"] == {"train/step": 2.0}
    assert merged.next_step_targets["train/step"] == 4.0


def test_merge_run_histories_applies_manual_offsets_before_next_target_updates():
    merged = merge_run_histories(
        [
            RunHistorySegment(
                run_path="entity/project/bvx4hc7w",
                rows=[{"rollout/step": 10, "reward": 1.0}],
            ),
            RunHistorySegment(
                run_path="entity/project/ctq3vfmo",
                rows=[{"rollout/step": 10, "reward": 1.1}],
            ),
        ],
        manual_offsets_by_run={"ctq3vfmo": {"rollout/step": 5.0}},
    )

    assert [row["rollout/step"] for row in merged.rows] == [10, 15.0]
    assert merged.next_step_targets["rollout/step"] == 16.0


def test_runtime_step_aligner_lazily_matches_first_seen_step_to_target():
    aligner = RuntimeStepAligner({"train/step": 35, "rollout/step": 41})

    first = aligner.apply({"train/step": 30, "rollout/step": 36, "train/loss": 0.5})
    second = aligner.apply({"train/step": 31, "rollout/step": 37, "train/loss": 0.4})

    assert first["train/step"] == 35.0
    assert first["rollout/step"] == 41.0
    assert second["train/step"] == 36.0
    assert second["rollout/step"] == 42.0


def test_merge_run_histories_respects_checkpoint_safe_max_step_boundary():
    merged = merge_run_histories(
        [
            RunHistorySegment(
                run_path="entity/project/bvx4hc7w",
                rows=[
                    {"train/step": 0, "train/loss": 1.0},
                    {"train/step": 4, "train/loss": 0.9},
                ],
            ),
            RunHistorySegment(
                run_path="entity/project/ctq3vfmo",
                rows=[
                    {"train/step": 5, "train/loss": 0.8},
                    {"train/step": 29, "train/loss": 0.4},
                    {"train/step": 30, "train/loss": 0.3},
                ],
            ),
        ],
        max_step_targets={"train/step": 29},
    )

    assert [row["train/step"] for row in merged.rows] == [0, 4, 5, 29]
    assert merged.next_step_targets["train/step"] == 30.0


def test_drop_first_history_rows_skips_prefix_without_touching_remaining_rows():
    rows = [{"x": 1}, {"x": 2}, {"x": 3}]
    assert drop_first_history_rows(rows, 2) == [{"x": 3}]


def test_filter_loggable_history_row_drops_null_and_private_fields():
    filtered = filter_loggable_history_row(
        {
            "_runtime": 12.3,
            "train/step": 34,
            "train/loss": 0.42,
            "train/grad_norm": None,
            "rollout/step": None,
        }
    )

    assert filtered == {
        "train/step": 34,
        "train/loss": 0.42,
    }


def test_repair_low_step_rows_uses_latest_anchor_step_for_sparse_eval_points():
    repaired = repair_low_step_rows(
        [
            {"rollout/step": 30, "train/step": 30},
            {"rollout/step": 31, "train/step": 31},
            {"eval/step": 5, "eval/aime": 0.0},
        ],
        step_key="eval/step",
        min_expected_step=30,
        anchor_step_keys=("train/step", "rollout/step"),
    )

    assert repaired[-1]["eval/step"] == 31.0


def test_shift_step_rows_only_shifts_steps_at_or_above_threshold():
    shifted = shift_step_rows(
        [
            {"eval/step": 4.0, "eval/math500": 0.354},
            {"eval/step": 5.0, "eval/math500": 0.442},
            {"eval/step": 10.0, "eval/math500": 0.498},
        ],
        step_key="eval/step",
        min_step_inclusive=5.0,
        delta=4.0,
    )

    assert [row["eval/step"] for row in shifted] == [4.0, 9.0, 14.0]
