# src/utils.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.constants import DEFAULT_POLICY_NAME, POLICY_MIXED_V1, TARGET_COLS

PolicySpec = Union[str, Dict[str, str], None]


# -------------------------
# Basic helpers
# -------------------------
def ensure_targets(df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    """
    Ensure all target columns exist and are numeric.
    Missing columns are created with 0.
    NaN -> 0 (CheXpert convention for blanks).
    """
    df = df.copy()
    for c in target_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[target_cols] = df[target_cols].fillna(0)
    return df


def extract_patient_id(path_value: str) -> str:
    """
    Match dataset.py behavior:
    - Prefer 'patientXXXX' pattern
    - Fallback to path segments
    """
    s = str(path_value)
    m = re.search(r"(patient\d+)", s)
    if m:
        return m.group(1)

    parts = re.split(r"[\\/]+", s)
    if len(parts) >= 3:
        return parts[2]
    return parts[0] if parts else "unknown"


# -------------------------
# Policy helpers
# -------------------------
def resolve_policy(policy: PolicySpec, target_cols: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Returns per-class policy dict mapping label -> {"ignore","zeros","ones"}.

    Notes:
    - If policy is a dict, values are normalized to lowercase.
    - If a class is missing in dict, caller should treat it as default "zeros".
    """
    if target_cols is None:
        target_cols = list(TARGET_COLS)

    if policy is None:
        policy = DEFAULT_POLICY_NAME

    if isinstance(policy, dict):
        return {k: str(v).lower().strip() for k, v in policy.items()}

    name = str(policy).lower().strip()

    if name in {"mixed_v1", "mixed-v1"}:
        return {k: str(v).lower().strip() for k, v in POLICY_MIXED_V1.items()}

    if name in {"u-zeros", "zeros"}:
        return {c: "zeros" for c in target_cols}

    if name in {"u-ones", "ones"}:
        return {c: "ones" for c in target_cols}

    if name in {"paperish", "paper"}:
        d = {c: "zeros" for c in target_cols}
        for c in ["Atelectasis", "Edema"]:
            if c in d:
                d[c] = "ones"
        return d

    if name == "custom":
        # legacy behavior you used earlier
        d = {c: "zeros" for c in target_cols}
        for c in ["Atelectasis", "Edema", "Pleural Effusion"]:
            if c in d:
                d[c] = "ones"
        if "Consolidation" in d:
            d["Consolidation"] = "ignore"
        return d

    raise ValueError(f"Unknown policy: {policy}")


def apply_uncertainty_policy(
    df: pd.DataFrame,
    policy: PolicySpec,
    target_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Apply per-class uncertainty policy:
      - ignore: keep -1
      - zeros : -1 -> 0
      - ones  : -1 -> 1
    """
    if target_cols is None:
        target_cols = list(TARGET_COLS)

    df = ensure_targets(df, target_cols)
    p = resolve_policy(policy, target_cols=target_cols)

    df = df.copy()
    for c in target_cols:
        mode = p.get(c, "zeros")
        if mode == "ignore":
            continue
        if mode == "zeros":
            df[c] = df[c].replace(-1, 0)
        elif mode == "ones":
            df[c] = df[c].replace(-1, 1)
        else:
            raise ValueError(f"Invalid policy for {c}: {mode} (use ignore/zeros/ones)")
    return df


# -------------------------
# pos_weight computation
# -------------------------
def compute_pos_weight_from_train_split(
    csv_path: str,
    target_cols: Optional[List[str]] = None,
    seed: int = 42,
    frontal_only: bool = True,
    policy: PolicySpec = DEFAULT_POLICY_NAME,
    test_size: float = 0.1,
    min_pw: float = 1.0,
    max_pw: float = 100.0,
    eps: float = 1e-6,
    return_stats: bool = False,
) -> Union[List[float], Tuple[List[float], Dict[str, Dict[str, Any]]]]:
    """
    Compute pos_weight per class from the TRAIN split only (patient-level split),
    using the SAME uncertainty policy as dataset.

    - "ignore": values -1 are excluded (masked).
    - Clip pos_weight to [min_pw, max_pw] for stability.
    """
    if target_cols is None:
        target_cols = list(TARGET_COLS)

    df = pd.read_csv(csv_path)

    if frontal_only and "Frontal/Lateral" in df.columns:
        df = df[df["Frontal/Lateral"] == "Frontal"].copy()

    if "Path" not in df.columns:
        raise ValueError("CSV must contain a 'Path' column")

    # apply policy (and ensure columns)
    df = apply_uncertainty_policy(df, policy=policy, target_cols=target_cols)

    # patient-level split (match dataset.py)
    df["Patient"] = df["Path"].apply(extract_patient_id)
    patients = df["Patient"].unique()

    train_p, _ = train_test_split(patients, test_size=test_size, random_state=seed, shuffle=True)
    train_df = df[df["Patient"].isin(train_p)].copy()

    pos_weight: List[float] = []
    stats: Dict[str, Dict[str, Any]] = {}

    for col in target_cols:
        y = train_df[col].to_numpy(dtype=np.float32)

        valid = y >= 0  # ignore(-1)
        yv = y[valid]

        pos = float(np.sum(yv == 1))
        neg = float(np.sum(yv == 0))

        pw = 1.0 if pos < 1 else (neg / (pos + eps))
        pw = float(np.clip(pw, min_pw, max_pw))

        pos_weight.append(pw)
        stats[col] = {
            "pos": pos,
            "neg": neg,
            "pw": pw,
            "valid": float(valid.sum()),
            "total": float(y.shape[0]),
        }

    if return_stats:
        return pos_weight, stats
    return pos_weight