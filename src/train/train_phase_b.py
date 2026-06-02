"""Canonical Phase B training and evaluation entrypoint."""
from __future__ import annotations

import sys

from src.train.train_phase_b_m0_corner import DependencyError, main, parse_args


if __name__ == "__main__":
    try:
        raise SystemExit(main(parse_args()))
    except DependencyError as exc:
        print(f"[phase_b] 런타임 의존성 오류: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except (RuntimeError, ValueError) as exc:
        print(f"[phase_b] 설정 오류: {exc}", file=sys.stderr)
        raise SystemExit(2)
