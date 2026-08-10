#!/usr/bin/env python3
"""Simple CLI to view and edit backend/llm_config.json safely.

Usage:
  python scripts/llm_config_cli.py show
  python scripts/llm_config_cli.py set llm_provider openai_compatible
  python scripts/llm_config_cli.py set llm_model gpt-4o-mini
  python scripts/llm_config_cli.py unset llm_api_key

This edits `backend/llm_config.json` in place. If the file doesn't exist it will
be created from the example when possible.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "llm_config.json"
EXAMPLE = ROOT / "llm_config.json.example"


def load_config():
    if CFG.exists():
        try:
            return json.loads(CFG.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
    if EXAMPLE.exists():
        try:
            return json.loads(EXAMPLE.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
    return {}


def save_config(cfg):
    CFG.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=("show", "set", "unset"))
    p.add_argument("key", nargs="?")
    p.add_argument("value", nargs="?")
    args = p.parse_args()

    cfg = load_config()

    if args.command == "show":
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        return

    if args.command == "set":
        if not args.key or args.value is None:
            p.error("set requires KEY VALUE")
        cfg[args.key] = args.value
        save_config(cfg)
        print(f"Wrote {args.key} to {CFG}")
        return

    if args.command == "unset":
        if not args.key:
            p.error("unset requires KEY")
        cfg.pop(args.key, None)
        save_config(cfg)
        print(f"Removed {args.key} from {CFG}")


if __name__ == "__main__":
    main()
