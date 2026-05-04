from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Suppress cosmetic sklearn / LightGBM warnings that fire dozens of times per
# walk-forward fold but carry no signal:
# - feature-name mismatch between fit and predict (Optuna trial loops fit on
#   numpy arrays then predict on numpy arrays; the warning compares against
#   prior training-set names, not output correctness),
# - duplicate `eval_at` argument warning that LightGBM emits when LGBMRanker
#   receives `eval_at` both in constructor params and via fit() default.
# None of these affect numeric results.
warnings.filterwarnings(
    "ignore",
    message=r"X has feature names, but LogisticRegression was fitted without",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but LGBM(Classifier|Ranker) was fitted with",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Found 'eval_at' in params",
    category=UserWarning,
)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the YAML config file. Default: configs/base.yaml",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="DEBUG, INFO, WARNING...",
    )
    return parser
