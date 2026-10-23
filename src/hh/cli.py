"""Command-line interface: the `hh` command.

    hh eval          run a detector over a dataset and report metrics
    hh detect        check a single answer against a context, ad hoc
    hh investigate   check an answer claim-by-claim with a tool-using agent
    hh list          show available detectors and model backends

Built on argparse from the standard library — no third-party CLI deps, so `hh`
works the moment the package is installed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from . import __version__
from .dataset import load_dataset
from .detectors.base import Detector, available_detectors, load_detector
from .eval import EvalRun, run_eval
from .metrics import evaluate
from .models.base import Model, available_backends, load_model
from .report import to_json, to_markdown
from .types import Detection, Example, Support


def _load_model(args) -> Model:
    if args.model_id:
        return load_model(args.model, model_id=args.model_id)
    return load_model(args.model)


def _build_detector(args) -> Detector:
    kwargs: dict[str, object] = {}
    if getattr(args, "verifier", None):
        kwargs["verifier"] = args.verifier
    if args.detector == "self_consistency" or getattr(args, "verifier", None) == "llm":
        kwargs["model"] = _load_model(args)
    if args.detector == "investigate":
        kwargs["model"] = _load_model(args)
        if getattr(args, "docs", None):
            kwargs["docs"] = args.docs
        if getattr(args, "sentences", False):
            kwargs["atomic"] = False
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


def cmd_investigate(args) -> int:
    if args.answers and args.answer:
        raise SystemExit("hh investigate: use --answer or --answers, not both")
    if args.answers:
        examples = _load_answers(args.answers)
        if not examples:
            raise SystemExit(f"hh investigate: no answers found in {args.answers}")
    elif args.answer:
        examples = [
            Example(id="adhoc", prompt=args.prompt or "", context=args.context, answer=args.answer)
        ]
    else:
        raise SystemExit("hh investigate: provide --answer <text> or --answers <jsonl file>")
    kwargs: dict[str, object] = {
        "model": _load_model(args),
        "max_steps": args.max_steps,
        "samples": args.samples,
    }
    if args.docs:
        kwargs["docs"] = args.docs
    detector = load_detector("investigate", **kwargs)
    detections, failures = _run_batch(detector, examples)
    run = EvalRun(result=evaluate(detections), detections=detections)
    # For a batch of answers the verdicts ARE the report; default to showing
    # every example rather than the first five.
    show = args.show if args.show is not None else len(detections)
    _emit(run, args, show_examples=max(show, 1))
    return 1 if failures else 0


def _run_batch(
    detector: Detector, examples: list[Example]
) -> tuple[list[Detection], list[tuple[str, str]]]:
    """Investigate answer by answer with progress on stderr.

    One failing answer (an API error, a bad row) must not sink the batch:
    the failure is reported and skipped, and every other answer's verdicts
    survive. Progress goes to stderr so stdout stays a clean report.
    """
    detections: list[Detection] = []
    failures: list[tuple[str, str]] = []
    total = len(examples)
    started = time.monotonic()
    for i, ex in enumerate(examples, 1):
        t0 = time.monotonic()
        print(f"[{ex.id}] {i}/{total}: investigating…", file=sys.stderr, flush=True)
        try:
            det = detector.detect(ex)
        except Exception as e:  # noqa: BLE001 - a bad answer never sinks the batch
            msg = _cli_error(e)
            failures.append((ex.id, msg))
            print(
                f"[{ex.id}] {i}/{total}: FAILED after {time.monotonic() - t0:.1f}s — {msg}",
                file=sys.stderr,
                flush=True,
            )
            continue
        supported = sum(1 for v in det.verdicts if v.support is Support.SUPPORTED)
        flagged = sum(1 for v in det.verdicts if v.is_hallucination)
        print(
            f"[{ex.id}] {i}/{total}: {len(det.verdicts)} claims — "
            f"{supported} supported, {flagged} flagged ({time.monotonic() - t0:.1f}s)",
            file=sys.stderr,
            flush=True,
        )
        detections.append(det)
    print(
        f"batch: {len(detections)}/{total} answers investigated, {len(failures)} failed "
        f"({time.monotonic() - started:.1f}s)",
        file=sys.stderr,
        flush=True,
    )
    return detections, failures


def _cli_error(e: Exception) -> str:
    """A one-line, length-capped error for stderr — never a traceback."""
    msg = " ".join(str(e).split())
    return msg if len(msg) <= 200 else msg[:199] + "…"


def _load_answers(path: str) -> list[Example]:
    """Read a JSONL file of answers: one object per line, `answer` required.

    `question` is accepted as an alias for `prompt` (the investigate fixture
    format); `context` and `labels` are honored when present. Rows with no
    answer are skipped — there is nothing to investigate.
    """
    examples: list[Example] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError as e:
                raise SystemExit(f"{path}:{i}: not valid JSON ({e})") from e
            if not isinstance(d, dict):
                raise SystemExit(f"{path}:{i}: expected a JSON object per line")
            if not d.get("answer"):
                continue
            if "prompt" not in d and "question" in d:
                d["prompt"] = d["question"]
            if not d.get("id"):
                d["id"] = f"row{i}"
            examples.append(Example.from_dict(d))
    return examples


def cmd_list(_args) -> int:
    print("Detectors:", ", ".join(available_detectors()))
    print("Model backends:", ", ".join(available_backends()))
    return 0


def _emit(run: EvalRun, args, show_examples: int = 5) -> None:
    text = to_json(run) if args.format == "json" else to_markdown(run, show_examples=show_examples)
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
    pe.add_argument(
        "--docs",
        default=None,
        help="folder of source documents (for the investigate detector)",
    )
    pe.add_argument(
        "--sentences",
        action="store_true",
        help="with --detector investigate: split answers into sentences, not atomic facts",
    )
    _add_detector_flags(pe)
    pe.set_defaults(func=cmd_eval)

    pd = sub.add_parser("detect", help="check a single answer against a context")
    pd.add_argument("--answer", required=True)
    pd.add_argument("--context", default="")
    pd.add_argument("--prompt", default="")
    _add_detector_flags(pd)
    pd.set_defaults(func=cmd_detect)

    pi = sub.add_parser(
        "investigate",
        help="check an answer claim-by-claim with a tool-using agent",
    )
    pi.add_argument("--answer", default=None, help="a single answer to check")
    pi.add_argument(
        "--answers",
        default=None,
        help="a JSONL file of answers to check in one go (one object per line, "
        "'answer' required; 'question' or 'prompt' and 'context' optional)",
    )
    pi.add_argument("--context", default="", help="context to check against, if you have one")
    pi.add_argument("--prompt", default="", help="the question the answer responds to")
    pi.add_argument("--docs", default=None, help="folder of source documents the agent may search")
    pi.add_argument(
        "--max-steps", type=int, default=6, help="max tool rounds per claim (default: 6)"
    )
    pi.add_argument(
        "--samples",
        type=int,
        default=3,
        help="samples per self_consistency_check call (default: 3)",
    )
    pi.add_argument(
        "--sentences",
        action="store_true",
        help="split the answer into sentences instead of atomic facts "
        "(atomic gives each fact its own verdict but costs an extra model call)",
    )
    pi.add_argument(
        "--show",
        type=int,
        default=None,
        help="with --answers: how many examples to include in the markdown "
        "report (default: all of them)",
    )
    pi.add_argument(
        "--model",
        default="mock",
        choices=available_backends(),
        help="model backend driving the agent (default: mock; use --model openai-compat "
        "or another tools-capable backend for real investigations)",
    )
    pi.add_argument("--model-id", default=None, help="override the backend's default model id")
    pi.add_argument("--format", default="markdown", choices=["markdown", "json"])
    pi.add_argument("--out", default=None, help="write the report to a file")
    pi.set_defaults(func=cmd_investigate)

    pl = sub.add_parser("list", help="list detectors and model backends")
    pl.set_defaults(func=cmd_list)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
