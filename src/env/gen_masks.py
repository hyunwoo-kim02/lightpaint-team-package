"""Generate binary letter mask PNG files for LightPaint references."""
import sys
from pathlib import Path

# Resolve package root for `python gen_masks.py` direct invocation.
_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from src.env.letter_masks import generate_all_masks


def main() -> None:
    """Generate 64x64 PNG masks for letters R, L, T, I into data/letters/."""
    data_dir = _PKG_ROOT / "data" / "letters"
    letters = ["R", "L", "T", "I"]
    print(f"Generating masks for {letters} into {data_dir}", flush=True)
    generate_all_masks(data_dir, letters, size=64)
    print("gen_masks: DONE", flush=True)


if __name__ == "__main__":
    main()
