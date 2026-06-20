"""Command-line interface: the `hh` command.

    hh eval     run a detector over a dataset and report metrics
    hh detect   check a single answer against a context, ad hoc
    hh list     show available detectors and model backends

Built on argparse from the standard library — no third-party CLI deps, so `hh`
works the moment the package is installed.
"""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .dataset import load_dataset
from .detectors.base import Detector, available_detectors, load_detector
from .eval import EvalRun, run_eval
from .metrics import evaluate
from .models.base import Model, available_backends, load_model
from .report import to_json, to_markdown
from .types import Example


def _build_detector(args) -> Detector:
    kwargs: dict[str, object] = {}
    if getattr(args, "verifier", None):
        kwargs["verifier"] = args.verifier
    model = None
    if getattr(args, "verifier", None) == "llm" or args.detector == "self_consistency":
        model = (
            load_model(args.model, model_id=args.model_id)
            if args.model_id
            else load_model(args.model)
        )
        kwargs["model"] = model
    return load_detector(args.detector, **kwargs)


def _generator(args) -> Model | None:
    if not getattr(args, "generate", False):
        return None
    return (
        load_model(args.model, model_id=args.model_id) if args.model_id else load_model(args.model)
    )


def cmd_eval(args) -> int:
    examples = load_dataset(args.dataset)
    detector = _build_detector(args)
    run = run_eval(examples, detector, generator=_generator(args))
    _emit(run, args)
    return 0


def cmd_detect(args) -> int:
    detector = _build_detector(args)
    ex = Example(id="adhoc", prompt=args.prompt or "", context=args.context, answer=args.answer)
    det = detector.detect(ex)
    run = EvalRun(result=evaluate([det]), detections=[det])
    _emit(run, args)
    return 0


def cmd_list(_args) -> int:
    print("Detectors:", ", ".join(available_detectors()))
    print("Model backends:", ", ".join(available_backends()))
    return 0


def _emit(run: EvalRun, args) -> None:
    text = to_json(run) if args.format == "json" else to_markdown(run)
    if getattr(args, "out", None):
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"wrote report to {args.out}", file=sys.stderr)
    else:
        print(text)


def _add_detector_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--detector",
        default="groundedness",
        choices=available_detectors(),
        help="detection strategy (default: groundedness)",
    )
    p.add_argument(
        "--verifier",
        default="lexical",
        choices=["lexical", "llm"],
        help="how claims are verified (default: lexical, no API key)",
    )
    p.add_argument(
        "--model",
        default="mock",
        choices=available_backends(),
        help="model backend for the llm verifier / generation",
    )
    p.add_argument("--model-id", default=None, help="override the backend's default model id")
    p.add_argument("--format", default="markdown", choices=["markdown", "json"])
    p.add_argument("--out", default=None, help="write the report to a file")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hh",
        description="hallucination-hunter — detect and evaluate LLM hallucinations.",
    )
    parser.add_argument(
        "--version", action="version", version=f"hallucination-hunter {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pe = sub.add_parser("eval", help="evaluate a detector over a dataset")
    pe.add_argument("--dataset", required=True, help="path to a JSONL dataset")
    pe.add_argument(
        "--generate",
        action="store_true",
        help="generate answers from prompts using --model before detecting",
    )
    _add_detector_flags(pe)
    pe.set_defaults(func=cmd_eval)

    pd = sub.add_parser("detect", help="check a single answer against a context")
    pd.add_argument("--answer", required=True)
    pd.add_argument("--context", default="")
    pd.add_argument("--prompt", default="")
    _add_detector_flags(pd)
    pd.set_defaults(func=cmd_detect)

    pl = sub.add_parser("list", help="list detectors and model backends")
    pl.set_defaults(func=cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
