import argparse
import sys

from typing import Any


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a video compression network on a video dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parent_parser = argparse.ArgumentParser(add_help=False)

    
    subparsers = parser.add_subparsers(help="model source", dest="source")
    subparsers.required = True

    # Options for pretrained models
    pretrained_parser = subparsers.add_parser("pretrained", parents=[parent_parser])
    pretrained_parser.add_argument(
        "-q",
        "--quality",
        dest="qualities",
        nargs="+",
        type=int,
        default=(1,),
    )
    checkpoint_parser = subparsers.add_parser("checkpoint", parents=[parent_parser])
    checkpoint_parser.add_argument(
        "-p",
        "--path",
        dest="paths",
        type=str,
        nargs="*",
        required=True,
        help="checkpoint path",
    )
    return parser



def main(args: Any = None) -> None:
    if args is None:
        args = sys.argv[1:]
    parser = create_parser()
    args = parser.parse_args(args)

    if not args.source:
        print("Error: missing 'checkpoint' or 'pretrained' source.", file=sys.stderr)
        parser.print_help()
        raise SystemExit(1)
    description = (
        "entropy-estimation" if args.entropy_estimation else args.entropy_coder
    )




if __name__ == "__main__":
    main(sys.argv[1:])