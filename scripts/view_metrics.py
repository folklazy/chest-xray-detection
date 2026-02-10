"""
View training metrics from TensorBoard event files.

Usage:
    py -3.11 scripts/view_metrics.py --logdir logs/chexpert/convnext_tiny_img384_bs12_lr0.0003_e8
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
import argparse
from collections import defaultdict

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    print("❌ Please install tensorboard: pip install tensorboard")
    sys.exit(1)


def load_events(logdir: str) -> dict:
    """Load all scalar events from a TensorBoard log directory."""
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    
    scalars = {}
    for tag in ea.Tags()["scalars"]:
        events = ea.Scalars(tag)
        scalars[tag] = [(e.step, e.value) for e in events]
    
    return scalars


def find_best_epoch(scalars: dict, metric: str = "val_auc", mode: str = "max") -> tuple:
    """Find the best epoch based on a metric."""
    if metric not in scalars:
        return None, None
    
    values = scalars[metric]
    if mode == "max":
        best_idx = max(range(len(values)), key=lambda i: values[i][1])
    else:
        best_idx = min(range(len(values)), key=lambda i: values[i][1])
    
    return values[best_idx]  # (step, value)


def print_summary(scalars: dict, class_names: list[str]):
    """Print a summary of training metrics."""
    
    print("\n" + "=" * 60)
    print("📊 TRAINING METRICS SUMMARY")
    print("=" * 60)
    
    # Best epoch
    best_epoch, best_auc = find_best_epoch(scalars, "val_auc", "max")
    if best_epoch is not None:
        print(f"\n🏆 Best Epoch: {best_epoch}")
        print(f"   Mean val_auc: {best_auc:.4f}")
    
    # AUC per class at best epoch
    print("\n📈 AUC per Disease (at best epoch):")
    print("-" * 40)
    
    auc_per_class = {}
    for name in class_names:
        tag = f"val_auc_{name}"
        if tag in scalars:
            # Get value at best epoch
            for step, value in scalars[tag]:
                if step == best_epoch:
                    auc_per_class[name] = value
                    break
    
    for name in class_names:
        if name in auc_per_class:
            bar = "█" * int(auc_per_class[name] * 20)
            print(f"  {name:<20}: {auc_per_class[name]:.4f} {bar}")
        else:
            print(f"  {name:<20}: N/A")
    
    # Final training loss
    if "train_loss_epoch" in scalars:
        final_loss = scalars["train_loss_epoch"][-1][1]
        print(f"\n📉 Final Training Loss: {final_loss:.4f}")
    elif "train_loss" in scalars:
        final_loss = scalars["train_loss"][-1][1]
        print(f"\n📉 Final Training Loss: {final_loss:.4f}")
    
    # Learning rate
    if "lr-AdamW" in scalars:
        final_lr = scalars["lr-AdamW"][-1][1]
        print(f"📐 Final Learning Rate: {final_lr:.2e}")
    
    print("\n" + "=" * 60)


def list_runs(logdir: str):
    """List all runs in the log directory."""
    print(f"\n📁 Available runs in {logdir}:")
    print("-" * 40)
    
    for item in sorted(os.listdir(logdir)):
        path = os.path.join(logdir, item)
        if os.path.isdir(path):
            # Check if it has events
            has_events = any(f.startswith("events.out") for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))
            marker = "✅" if has_events else "📂"
            print(f"  {marker} {item}")


def main():
    parser = argparse.ArgumentParser(description="View training metrics from TensorBoard logs")
    parser.add_argument("--logdir", default=None, help="Path to specific run directory")
    parser.add_argument("--list", action="store_true", help="List available runs")
    args = parser.parse_args()
    
    CLASS_NAMES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]
    
    base_logdir = "logs/chexpert"
    
    if args.list or args.logdir is None:
        list_runs(base_logdir)
        if args.logdir is None:
            print("\n💡 Usage: py -3.11 scripts/view_metrics.py --logdir <run_path>")
            return
    
    if args.logdir:
        if not os.path.exists(args.logdir):
            print(f"❌ Directory not found: {args.logdir}")
            return
        
        print(f"\n📂 Loading: {args.logdir}")
        scalars = load_events(args.logdir)
        
        if not scalars:
            print("❌ No scalar events found. Is this a valid TensorBoard log directory?")
            return
        
        print_summary(scalars, CLASS_NAMES)


if __name__ == "__main__":
    main()
