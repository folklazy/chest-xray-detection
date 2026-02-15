"""
View training metrics from TensorBoard event files.

Examples:
  # list runs
  py -3.11 scripts/view_metrics.py --list --base_logdir "logs/chexpert_medpre"

  # summarize one run
  py -3.11 scripts/view_metrics.py --logdir "logs/chexpert_medpre\\xrv_densenet121_res224_all_xrv_img224_bs32_lr0.0002_e8"

  # summarize all runs under base_logdir (rank by best val_auc)
  py -3.11 scripts/view_metrics.py --all --base_logdir "logs/chexpert_medpre"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import argparse
from typing import Dict, List, Tuple, Optional

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("❌ Please install tensorboard: pip install tensorboard")
    sys.exit(1)


# -----------------------------
# Utilities
# -----------------------------
def _has_event_files(run_dir: str) -> bool:
    if not os.path.isdir(run_dir):
        return False
    for f in os.listdir(run_dir):
        if f.startswith("events.out.tfevents"):
            return True
    # sometimes events are in subdirs (rare)
    for root, _, files in os.walk(run_dir):
        for f in files:
            if f.startswith("events.out.tfevents"):
                return True
    return False


def _find_event_dirs(base_dir: str) -> List[str]:
    """Return run directories that contain TensorBoard event files."""
    runs = []
    if not os.path.isdir(base_dir):
        return runs

    # Common lightning structure: base_dir/<run>/events...
    for name in sorted(os.listdir(base_dir)):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path) and _has_event_files(path):
            runs.append(path)
    return runs


def _load_scalars(logdir: str) -> Dict[str, List[Tuple[int, float]]]:
    """
    Load all scalar tags from a TensorBoard run directory.
    Returns dict[tag] = [(step, value), ...] sorted by step.
    """
    ea = event_accumulator.EventAccumulator(
        logdir,
        size_guidance={
            event_accumulator.SCALARS: 0,  # load all
        },
    )
    ea.Reload()

    scalars: Dict[str, List[Tuple[int, float]]] = {}
    tags = ea.Tags().get("scalars", [])
    for tag in tags:
        events = ea.Scalars(tag)
        scalars[tag] = [(int(e.step), float(e.value)) for e in events]
        scalars[tag].sort(key=lambda x: x[0])
    return scalars


def _best_by_metric(
    scalars: Dict[str, List[Tuple[int, float]]],
    metric: str,
    mode: str = "max",
) -> Tuple[Optional[int], Optional[float]]:
    """
    Returns (best_step, best_value) by metric tag.
    """
    if metric not in scalars or not scalars[metric]:
        return None, None

    seq = scalars[metric]
    if mode == "max":
        best_step, best_val = max(seq, key=lambda x: x[1])
    else:
        best_step, best_val = min(seq, key=lambda x: x[1])
    return best_step, best_val


def _final_value(
    scalars: Dict[str, List[Tuple[int, float]]],
    tag: str
) -> Tuple[Optional[int], Optional[float]]:
    if tag not in scalars or not scalars[tag]:
        return None, None
    step, val = scalars[tag][-1]
    return step, val


def _nearest_step_value(
    scalars: Dict[str, List[Tuple[int, float]]],
    tag: str,
    target_step: int
) -> Optional[float]:
    """
    Get scalar value at the closest step to target_step.
    Avoids missing when step doesn't match exactly.
    """
    if tag not in scalars or not scalars[tag]:
        return None
    seq = scalars[tag]
    # binary search-ish
    lo, hi = 0, len(seq) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if seq[mid][0] < target_step:
            lo = mid + 1
        else:
            hi = mid
    # lo is first >= target_step
    candidates = []
    candidates.append(seq[lo])
    if lo - 1 >= 0:
        candidates.append(seq[lo - 1])
    # choose closest
    best = min(candidates, key=lambda x: abs(x[0] - target_step))
    return float(best[1])


def _sanitize_key(name: str) -> str:
    """Make logging keys stable (consistent with src/model.py)."""
    return (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("__", "_")
    )


def _bar(v: float, width: int = 20) -> str:
    if v is None:
        return ""
    v = max(0.0, min(1.0, float(v)))
    return "█" * int(v * width)


def _fmt(v: Optional[float], nd: int = 4) -> str:
    if v is None:
        return "N/A"
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return "N/A"


def _fmt_sci(v: Optional[float]) -> str:
    if v is None:
        return "N/A"
    return f"{float(v):.2e}"


# -----------------------------
# Summary
# -----------------------------
DEFAULT_CLASSES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]


def summarize_run(logdir: str, class_names: List[str]) -> Dict[str, object]:
    scalars = _load_scalars(logdir)

    # prefer val_auc_comp or val_auc_mean
    auc_tag = "val_auc_comp" if "val_auc_comp" in scalars else ("val_auc_mean" if "val_auc_mean" in scalars else "val_auc")
    
    best_step, best_auc = _best_by_metric(scalars, auc_tag, mode="max")

    # prefer epoch-aggregated losses if present
    train_loss_tag = "train_loss_epoch" if "train_loss_epoch" in scalars else "train_loss"
    val_loss_tag = "val_loss_epoch" if "val_loss_epoch" in scalars else ("val_loss" if "val_loss" in scalars else None)

    final_train_step, final_train_loss = _final_value(scalars, train_loss_tag) if train_loss_tag else (None, None)
    final_val_step, final_val_loss = _final_value(scalars, val_loss_tag) if val_loss_tag else (None, None)
    final_auc_step, final_auc = _final_value(scalars, auc_tag)

    # lr tag can vary; Lightning LRMonitor commonly uses like "lr-AdamW"
    lr_tag = None
    for cand in ["lr-AdamW", "lr", "learning_rate"]:
        if cand in scalars:
            lr_tag = cand
            break
    final_lr_step, final_lr = _final_value(scalars, lr_tag) if lr_tag else (None, None)

    # optional nan rate
    nan_rate_best = _nearest_step_value(scalars, "val_nan_rate", best_step) if best_step is not None else None
    nan_rate_final = _final_value(scalars, "val_nan_rate")[1] if "val_nan_rate" in scalars else None

    # per-class at best step (nearest)
    per_class = {}
    if best_step is not None:
        for name in class_names:
            tag = f"val_auc_{_sanitize_key(name)}"
            per_class[name] = _nearest_step_value(scalars, tag, best_step)

    return {
        "logdir": logdir,
        "scalars": scalars,
        "best_step": best_step,
        "best_val_auc": best_auc,
        "final_val_auc": final_auc,
        "final_val_auc_step": final_auc_step,
        "final_train_loss": final_train_loss,
        "final_train_loss_step": final_train_step,
        "final_val_loss": final_val_loss,
        "final_val_loss_step": final_val_step,
        "final_lr": final_lr,
        "final_lr_step": final_lr_step,
        "nan_rate_best": nan_rate_best,
        "nan_rate_final": nan_rate_final,
        "per_class_auc_at_best": per_class,
        "train_loss_tag_used": train_loss_tag,
        "val_loss_tag_used": val_loss_tag,
        "lr_tag_used": lr_tag,
    }


def print_run_summary(summary: Dict[str, object], class_names: List[str], show_tags: bool = False):
    logdir = summary["logdir"]
    best_step = summary["best_step"]
    best_auc = summary["best_val_auc"]
    final_auc = summary["final_val_auc"]
    final_train_loss = summary["final_train_loss"]
    final_val_loss = summary["final_val_loss"]
    final_lr = summary["final_lr"]
    nan_best = summary["nan_rate_best"]
    nan_final = summary["nan_rate_final"]
    per_class = summary["per_class_auc_at_best"]

    print("\n" + "=" * 70)
    print(f"📂 RUN: {logdir}")
    print("=" * 70)

    if best_step is not None and best_auc is not None:
        print(f"🏆 Best val_auc: {_fmt(best_auc)}  (step {best_step})")
    else:
        print("🏆 Best val_auc: N/A")

    print(f"📌 Final val_auc: {_fmt(final_auc)}")
    print(f"📉 Final train loss: {_fmt(final_train_loss)}")
    print(f"📉 Final val loss: {_fmt(final_val_loss)}")
    print(f"📐 Final LR: {_fmt_sci(final_lr)}")

    if nan_best is not None or nan_final is not None:
        print(f"🧪 val_nan_rate(best/final): {_fmt(nan_best, 6)} / {_fmt(nan_final, 6)}")

    print("\n📈 AUC per class (at best step):")
    print("-" * 70)
    for name in class_names:
        v = per_class.get(name, None) if isinstance(per_class, dict) else None
        if v is None:
            print(f"  {name:<20}: N/A")
        else:
            print(f"  {name:<20}: {_fmt(v)} {_bar(v)}")

    if show_tags:
        print("\n🔎 Tags used:")
        print(f"  train_loss tag: {summary.get('train_loss_tag_used')}")
        print(f"  val_loss tag  : {summary.get('val_loss_tag_used')}")
        print(f"  lr tag        : {summary.get('lr_tag_used')}")


def print_ranked_table(summaries: List[Dict[str, object]]):
    # rank by best_val_auc desc, fallback -inf
    def key_fn(s):
        v = s.get("best_val_auc", None)
        return float(v) if v is not None else -1e9

    ranked = sorted(summaries, key=key_fn, reverse=True)

    print("\n" + "=" * 90)
    print("🏁 RANKING (by best val_auc)")
    print("=" * 90)
    print(f"{'Rank':<6}{'best_auc':<10}{'final_auc':<10}{'final_train':<12}{'final_val':<12}{'run':<1}")
    print("-" * 90)

    for i, s in enumerate(ranked, 1):
        best_auc = s.get("best_val_auc", None)
        final_auc = s.get("final_val_auc", None)
        ft = s.get("final_train_loss", None)
        fv = s.get("final_val_loss", None)
        run_name = os.path.basename(s["logdir"])
        print(f"{i:<6}{_fmt(best_auc):<10}{_fmt(final_auc):<10}{_fmt(ft):<12}{_fmt(fv):<12}{run_name}")

    print("=" * 90)


def list_runs(base_logdir: str):
    runs = _find_event_dirs(base_logdir)
    print(f"\n📁 Available runs in: {base_logdir}")
    print("-" * 70)
    if not runs:
        print("  (no runs found)")
        return
    for p in runs:
        print(f"  ✅ {os.path.basename(p)}")


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="View training metrics from TensorBoard logs")
    parser.add_argument("--base_logdir", default="logs/chexpert", help="Base log directory for --list/--all")
    parser.add_argument("--list", action="store_true", help="List runs in base_logdir")
    parser.add_argument("--all", action="store_true", help="Summarize all runs in base_logdir and rank them")
    parser.add_argument("--logdir", default=None, help="Path to a specific run directory")
    parser.add_argument("--show_tags", action="store_true", help="Show which tags were used for loss/lr")
    args = parser.parse_args()

    class_names = DEFAULT_CLASSES

    if args.list:
        list_runs(args.base_logdir)
        return

    if args.all:
        runs = _find_event_dirs(args.base_logdir)
        if not runs:
            print(f"❌ No runs found in: {args.base_logdir}")
            return

        summaries = []
        for run in runs:
            try:
                summaries.append(summarize_run(run, class_names))
            except Exception as e:
                print(f"⚠️ Skipped {run} (error: {e})")

        if not summaries:
            print("❌ Could not read any runs.")
            return

        print_ranked_table(summaries)

        # print top 3 details for convenience
        ranked = sorted(
            summaries,
            key=lambda s: float(s.get("best_val_auc", -1e9) or -1e9),
            reverse=True
        )
        topk = ranked[:3]
        for s in topk:
            print_run_summary(s, class_names, show_tags=args.show_tags)
        return

    if args.logdir is None:
        print("💡 Use one of:")
        print("  --list --base_logdir <dir>")
        print("  --all  --base_logdir <dir>")
        print("  --logdir <run_dir>")
        return

    if not os.path.exists(args.logdir):
        print(f"❌ Directory not found: {args.logdir}")
        return

    print(f"\n📂 Loading: {args.logdir}")
    summary = summarize_run(args.logdir, class_names)
    print_run_summary(summary, class_names, show_tags=args.show_tags)


if __name__ == "__main__":
    main()