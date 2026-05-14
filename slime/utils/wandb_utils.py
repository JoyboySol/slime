import logging
import os
import json
from copy import deepcopy

import wandb

from .wandb_canonical import RuntimeStepAligner, normalize_step_targets
from .wandb_canonical import build_uniform_step_targets, filter_loggable_history_row, row_within_max_steps

logger = logging.getLogger(__name__)

_WANDB_STATE_FILENAME = "slime_wandb_state.json"


def _is_offline_mode(args) -> bool:
    """Detect whether W&B should run in offline mode.

    Priority order:
    1) args.wandb_mode if provided
    2) WANDB_MODE environment variable
    """
    if args.wandb_mode:
        return args.wandb_mode == "offline"
    return os.environ.get("WANDB_MODE") == "offline"


def _get_wandb_state_path(args) -> str | None:
    if not getattr(args, "wandb_dir", None):
        return None
    return os.path.join(args.wandb_dir, _WANDB_STATE_FILENAME)


def _load_persisted_run_identity(args) -> dict | None:
    if getattr(args, "wandb_start_fresh", False):
        return None
    state_path = _get_wandb_state_path(args)
    if state_path is None or not os.path.exists(state_path):
        return None

    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to load persisted W&B state from %s", state_path)
        return None

    if not isinstance(state, dict):
        logger.warning("Ignoring non-dict W&B state in %s", state_path)
        return None
    return state


def _load_step_targets_from_state(args, persisted_state: dict | None) -> dict[str, float]:
    current_targets = getattr(args, "wandb_step_targets", None)
    if getattr(args, "wandb_start_fresh", False):
        return normalize_step_targets(current_targets)
    if persisted_state is None:
        return normalize_step_targets(current_targets)
    return normalize_step_targets(persisted_state.get("step_targets") or current_targets)


def _persist_run_identity(args, *, group: str | None, run_name: str | None) -> None:
    state_path = _get_wandb_state_path(args)
    if state_path is None or wandb.run is None:
        return

    state = {
        "run_id": wandb.run.id,
        "group": group,
        "run_name": run_name,
        "project": getattr(args, "wandb_project", None),
        "entity": getattr(args, "wandb_team", None),
        "step_targets": normalize_step_targets(getattr(args, "wandb_step_targets", None)),
    }
    tmp_path = f"{state_path}.tmp"
    try:
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=True, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, state_path)
    except OSError:
        logger.exception("Failed to persist W&B state to %s", state_path)
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            logger.exception("Failed to remove temporary W&B state file %s", tmp_path)


def _compute_wandb_identity(args) -> tuple[str | None, str | None, str | None, str | None]:
    if args.wandb_random_suffix:
        group = args.wandb_group + "_" + wandb.util.generate_id()
        run_name = f"{group}-RANK_{args.rank}"
    else:
        group = args.wandb_group
        run_name = args.wandb_group

    explicit_run_id = getattr(args, "wandb_run_id", None)
    persisted_state = _load_persisted_run_identity(args)

    if explicit_run_id:
        if persisted_state and persisted_state.get("run_id") == explicit_run_id:
            return (
                explicit_run_id,
                "must",
                persisted_state.get("group") or group,
                persisted_state.get("run_name") or run_name,
            )
        return explicit_run_id, "must", group, run_name

    if persisted_state and persisted_state.get("run_id"):
        return (
            persisted_state["run_id"],
            "must",
            persisted_state.get("group") or group,
            persisted_state.get("run_name") or run_name,
        )

    return None, None, group, run_name


def _validate_wandb_resume_from_step(args, run_id: str | None) -> None:
    resume_from_step = getattr(args, "wandb_resume_from_step", None)
    if resume_from_step is None:
        return
    if run_id is None and not getattr(args, "wandb_start_fresh", False):
        raise ValueError("--wandb-resume-from-step requires --wandb-run-id or a persisted W&B run id.")
    if resume_from_step < 0:
        raise ValueError("--wandb-resume-from-step must be non-negative.")


def _resolve_wandb_run_path(args, run_id: str) -> str | None:
    project = getattr(args, "wandb_project", None)
    if not project:
        return None
    entity = getattr(args, "wandb_team", None)
    return f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"


def _load_wandb_backfill_rows(args, source_run_id: str, max_step: int | float) -> list[dict]:
    project = getattr(args, "wandb_project", None)
    if not project:
        logger.warning("Skipping W&B history backfill because --wandb-project is not set.")
        return []

    max_step_targets = build_uniform_step_targets(("train/step", "rollout/step", "eval/step"), float(max_step))
    try:
        api = wandb.Api(timeout=int(os.environ.get("WANDB_PUBLIC_API_TIMEOUT", "60")))
        run_path = _resolve_wandb_run_path(args, source_run_id)
        if run_path is None:
            entity = api.default_entity
            run_path = f"{entity}/{project}/{source_run_id}" if entity else f"{project}/{source_run_id}"
        rows = []
        for row in api.run(run_path).scan_history():
            if not row_within_max_steps(row, max_step_targets):
                continue
            filtered = filter_loggable_history_row(row)
            if filtered:
                rows.append(filtered)
        return rows
    except Exception:
        logger.exception("Failed to load W&B history backfill rows from %s", run_path)
        return []


def _backfill_wandb_history_if_needed(args, source_run_id: str | None) -> None:
    if not getattr(args, "wandb_start_fresh", False):
        return
    resume_from_step = getattr(args, "wandb_resume_from_step", None)
    if resume_from_step is None or not source_run_id:
        return

    rows = _load_wandb_backfill_rows(args, source_run_id, resume_from_step)
    if not rows:
        logger.warning(
            "No W&B history rows were backfilled from run %s up to step %s.",
            source_run_id,
            resume_from_step,
        )
        return

    logger.info(
        "Backfilling %d W&B history rows from run %s up to step %s into fresh run %s.",
        len(rows),
        source_run_id,
        resume_from_step,
        wandb.run.id if wandb.run is not None else "<unknown>",
    )
    for row in rows:
        wandb.log(row)


def init_wandb_primary(args):
    if not args.use_wandb:
        args.wandb_run_id = None
        return

    # Set W&B mode if specified (overrides WANDB_MODE env var)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_mode == "offline":
            logger.info("W&B offline mode enabled. Data will be saved locally.")
        elif args.wandb_mode == "disabled":
            logger.info("W&B disabled mode enabled. No data will be logged.")
        elif args.wandb_mode == "online":
            logger.info("W&B online mode enabled. Data will be uploaded to cloud.")

    offline = _is_offline_mode(args)

    # Only perform explicit login when NOT offline
    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    persisted_state = _load_persisted_run_identity(args)
    explicit_run_id = getattr(args, "wandb_run_id", None)
    backfill_source_run_id = explicit_run_id
    if getattr(args, "wandb_start_fresh", False):
        if explicit_run_id is not None:
            logger.info("Using explicit --wandb-run-id as W&B history backfill source because --wandb-start-fresh was requested.")
        explicit_run_id = None
        args.wandb_run_id = None
    args.wandb_step_targets = _load_step_targets_from_state(args, persisted_state)
    if explicit_run_id and persisted_state and persisted_state.get("run_id") != explicit_run_id:
        args.wandb_step_targets = {}
    run_id, resume_mode, group, run_name = _compute_wandb_identity(args)
    _validate_wandb_resume_from_step(args, run_id)

    # Prepare wandb init parameters
    init_kwargs = {
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "group": group,
        "name": run_name,
        "config": _compute_config_for_logging(args),
    }
    if run_id is not None:
        init_kwargs["id"] = run_id
    if resume_mode is not None:
        init_kwargs["resume"] = resume_mode

    # Configure settings based on offline/online mode
    if offline:
        init_kwargs["settings"] = wandb.Settings(mode="offline")
    else:
        init_kwargs["settings"] = wandb.Settings(mode="shared", x_primary=True)

    # Add custom directory if specified
    if args.wandb_dir:
        # Ensure directory exists to avoid backend crashes
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir
        logger.info(f"W&B logs will be stored in: {args.wandb_dir}")

    wandb.init(**init_kwargs)

    _init_wandb_common()

    # Set wandb_run_id in args for easy access throughout the training process
    args.wandb_run_id = wandb.run.id
    _backfill_wandb_history_if_needed(args, backfill_source_run_id)
    _persist_run_identity(args, group=group, run_name=run_name)


def reinit_wandb_primary_with_open_metrics(args, router_addr):
    """Re-initialize the primary W&B run with open metrics endpoints.

    The primary wandb init happens before rollout servers start (to obtain
    ``wandb_run_id`` for secondary processes).  This function is called
    *after* servers are up so the router address is available for scraping
    SGLang Prometheus metrics via the primary process's stats monitor.
    """
    if not args.use_wandb or _is_offline_mode(args):
        return
    if getattr(args, "wandb_mode", None) == "disabled":
        return
    if router_addr is None:
        return
    wandb_run_id = getattr(args, "wandb_run_id", None)
    if wandb_run_id is None:
        return

    import sglang_router

    if "slime" not in sglang_router.__version__:
        logger.warning(
            "Only customized sglang_router from https://github.com/zhuzilin/sgl-router supports uploading metrics."
        )
        return

    logger.info(f"Re-initializing primary W&B with SGLang metrics at {router_addr}.")

    wandb.finish()

    init_kwargs = {
        "id": wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "resume": "allow",
        "reinit": True,
        "settings": wandb.Settings(
            mode="shared",
            x_primary=True,
            x_stats_open_metrics_endpoints={
                "sgl_engine": f"{router_addr}/engine_metrics",
            },
            x_stats_open_metrics_filters={
                "sgl_engine.*": {},
            },
        ),
    }

    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    wandb.init(**init_kwargs)
    _init_wandb_common()
    _persist_run_identity(args, group=None, run_name=None)


def _compute_config_for_logging(args):
    output = deepcopy(args.__dict__)

    whitelist_env_vars = [
        "SLURM_JOB_ID",
        # We may insert more default values here, and may also allow users to configure a whitelist
    ]
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}

    return output


# https://docs.wandb.ai/guides/track/log/distributed-training/#track-all-processes-to-a-single-run
def init_wandb_secondary(args):
    wandb_run_id = getattr(args, "wandb_run_id", None)
    if wandb_run_id is None:
        return
    args.wandb_step_targets = _load_step_targets_from_state(args, _load_persisted_run_identity(args))

    # Set W&B mode if specified (same as primary)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    offline = _is_offline_mode(args)

    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Configure settings based on offline/online mode
    if offline:
        settings_kwargs = dict(mode="offline")
    else:
        settings_kwargs = dict(
            mode="shared",
            x_primary=False,
            x_update_finish_state=False,
        )

    init_kwargs = {
        "id": wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "config": args.__dict__,
        "resume": "allow",
        "reinit": True,
        "settings": wandb.Settings(**settings_kwargs),
    }

    # Add custom directory if specified
    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    wandb.init(**init_kwargs)

    _init_wandb_common()


def prepare_metrics_for_wandb(args, metrics: dict):
    step_targets = normalize_step_targets(getattr(args, "wandb_step_targets", None))
    if not step_targets:
        return metrics

    aligner = getattr(args, "_wandb_runtime_step_aligner", None)
    if aligner is None:
        aligner = RuntimeStepAligner(step_targets)
        setattr(args, "_wandb_runtime_step_aligner", aligner)
    return aligner.apply(metrics)


def _init_wandb_common():
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("rollout/step")
    wandb.define_metric("rollout/*", step_metric="rollout/step")
    wandb.define_metric("multi_turn/*", step_metric="rollout/step")
    wandb.define_metric("passrate/*", step_metric="rollout/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    wandb.define_metric("perf/*", step_metric="rollout/step")
