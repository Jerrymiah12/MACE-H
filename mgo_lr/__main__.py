import argparse
import importlib
import os
import sys

from . import __version__
from .config import load_config, save_resolved

# stage name -> (module, function). Functions: stage(cfg, workspace, args) -> int
STAGES = {
    "init-reference": ("mgo_lr.reference", "init_reference_stage"),
    "collect-reference": ("mgo_lr.reference", "collect_reference_stage"),
    "init-dfpt": ("mgo_lr.dfpt", "init_dfpt_stage"),
    "collect-dfpt": ("mgo_lr.dfpt", "collect_dfpt_stage"),
    "gen-structures": ("mgo_lr.displacements", "gen_structures_stage"),
    "collect-dft": ("mgo_lr.convert", "collect_dft_stage"),
    "lr-process": ("mgo_lr.lr", "lr_process_stage"),
    "validate": ("mgo_lr.validate", "validate_stage"),
    "locality-report": ("mgo_lr.locality", "locality_report_stage"),
    "organize": ("mgo_lr.organize", "organize_stage"),
    "export-target": ("mgo_lr.export", "export_target_stage"),
    "status": ("mgo_lr.snapshot", "status_stage"),
}

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "configs", "mgo.yaml")


def main(argv):
    p = argparse.ArgumentParser(
        prog="python -m mgo_lr",
        description=f"MgO MACE-H-LR dataset pipeline v{__version__}. "
                    f"Stages: {', '.join(STAGES)}")
    p.add_argument("stage", choices=sorted(STAGES))
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--workspace", required=True)
    p.add_argument("--set", dest="set_name",
                   choices=["pilot", "main", "large"], default=None)
    p.add_argument("--target", choices=["full", "lr", "sr"], default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    os.makedirs(args.workspace, exist_ok=True)
    save_resolved(cfg, args.workspace, args.stage)
    module, func = STAGES[args.stage]
    fn = getattr(importlib.import_module(module), func)
    return fn(cfg, args.workspace, args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
