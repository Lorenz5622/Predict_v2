#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys


def _force_stage(stage: int) -> None:
    argv = sys.argv
    if "--stage" in argv:
        idx = argv.index("--stage")
        if idx + 1 >= len(argv):
            argv.append(str(stage))
        else:
            argv[idx + 1] = str(stage)
    else:
        argv.extend(["--stage", str(stage)])


def main() -> None:
    sys.path.insert(0, os.getcwd())
    _force_stage(1)

    import finetune_qwen_dynamic_moe

    finetune_qwen_dynamic_moe.main()


if __name__ == "__main__":
    main()
