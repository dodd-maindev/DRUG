#!/usr/bin/env python3
"""CLI entrypoint for Stage 0 sanity, mask-leakage, shortcut, and strata checks."""

import argparse
import importlib
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


class StageZeroCliApplication:
    """Builds argument parsers and lazily dispatches Stage 0 diagnostic analyzers."""

    RUNNER_MAP = {
        "tokens": ("diagnostics.token_alignment_analyzer", "TokenAlignmentAnalyzer"),
        "leak": ("diagnostics.mask_leakage_analyzer", "MaskLeakageAnalyzer"),
        "shortcut": ("diagnostics.shortcut_analyzer", "ShortcutAnalyzer"),
        "strata": ("diagnostics.strata_analyzer", "StrataAnalyzer"),
    }

    @staticmethod
    def _register_common_arguments(parser: argparse.ArgumentParser) -> None:
        """Register dataset, directory, trimming, and output options."""
        parser.add_argument("--protein-dir")
        parser.add_argument("--drug-dir")
        parser.add_argument("--struct-dir")
        parser.add_argument("--train-csv")
        parser.add_argument("--val-csv")
        parser.add_argument("--test-csv")
        parser.add_argument("--col-drug")
        parser.add_argument("--col-prot")
        parser.add_argument("--col-label")
        parser.add_argument(
            "--trim",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Drop final token row like MyDataset (pass --no-trim for KIBA).",
        )
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--out", default="stage0_out")

    @staticmethod
    def _register_model_arguments(parser: argparse.ArgumentParser, checkpoint_required: bool) -> None:
        """Register repository, configuration, model, checkpoint, and batch options."""
        parser.add_argument("--repo", default="3DICE")
        parser.add_argument("--cfg-spec", default="config.cfg:get_cfg_defaults")
        parser.add_argument("--model-spec", default="model:Model")
        parser.add_argument("--ckpt", required=checkpoint_required)
        parser.add_argument("--batch", type=int, default=32)

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        """Construct the top-level CLI parser with tokens, leak, shortcut, and strata."""
        parser = argparse.ArgumentParser(description="Stage 0 diagnostic checks for 3DICE.")
        subparsers = parser.add_subparsers(dest="cmd", required=True)

        tokens_parser = subparsers.add_parser("tokens")
        cls._register_common_arguments(tokens_parser)
        tokens_parser.add_argument("--n-drugs", type=int, default=300)
        tokens_parser.add_argument("--n-proteins", type=int, default=100)
        tokens_parser.add_argument("--chain-map")
        tokens_parser.add_argument("--seq-csv")
        tokens_parser.add_argument("--rerun", type=int, default=0)

        leak_parser = subparsers.add_parser("leak")
        cls._register_common_arguments(leak_parser)
        cls._register_model_arguments(leak_parser, checkpoint_required=True)
        leak_parser.add_argument("--n-pairs", type=int, default=256)

        shortcut_parser = subparsers.add_parser("shortcut")
        cls._register_common_arguments(shortcut_parser)
        shortcut_parser.add_argument("--seeds", type=int, default=3)

        strata_parser = subparsers.add_parser("strata")
        cls._register_common_arguments(strata_parser)
        cls._register_model_arguments(strata_parser, checkpoint_required=False)
        strata_parser.add_argument("--pred-csv")
        strata_parser.add_argument("--source-map")
        return parser

    @classmethod
    def run(cls) -> None:
        """Parse CLI arguments and invoke the selected diagnostic analyzer."""
        arguments = cls.build_parser().parse_args()
        if arguments.cmd == "strata" and not arguments.pred_csv and not arguments.ckpt:
            sys.exit("[!] strata needs --ckpt or --pred-csv")
        module_name, class_name = cls.RUNNER_MAP[arguments.cmd]
        runner_class = getattr(importlib.import_module(module_name), class_name)
        runner_class(arguments).run()


if __name__ == "__main__":
    StageZeroCliApplication.run()
