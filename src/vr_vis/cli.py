from __future__ import annotations
import argparse
import sys
from .core import run


def main() -> int:
    p = argparse.ArgumentParser(
        prog="vrvis", description="Vietoris–Rips filtration progression visualizer"
    )
    p.add_argument(
        "--embeddings",
        required=True,
        help="Path to a pickle file containing a dict: { mouse: {'embedding': np.ndarray} }",
    )
    p.add_argument("--mouse", required=True, help="Mouse/session key to visualize")
    p.add_argument("--n-samples", type=int, default=1000)
    p.add_argument("--min-d", type=float, default=0.05)
    p.add_argument("--max-d", type=float, default=10.0)
    p.add_argument("--n-steps", type=int, default=10)
    p.add_argument(
        "--sampling-method",
        choices=["random", "uniform", "first"],
        default="uniform",
    )

    args = p.parse_args()

    try:
        run(
            args.embeddings,
            mouse_name=args.mouse,
            n_samples=args.n_samples,
            diameter_range=(args.min_d, args.max_d),
            n_filtration_steps=args.n_steps,
            sampling_method=args.sampling_method,
        )
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
