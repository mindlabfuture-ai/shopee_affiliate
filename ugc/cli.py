"""python -m ugc <command>: plan | qa | serve"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import config, planner, qa
from .spec import ProductSpec, listing_dir, listing_images, load_spec, product_dir, read_json, write_json


def cmd_plan(a) -> int:
    spec = load_spec(a.product)
    result = planner.make_plan(spec, listing_images(a.product))
    write_json(product_dir(a.product) / "plan.json", result)
    errors = [i for i in result["lint"] if i["level"] == "error"]
    for c in result["clips"]:
        print(f"clip {c['index']} ({c['duration_s']}s, {c['shots']}):\n  {c['prompt']}\n")
    print("hooks:", result["hook_captions"])
    print("lint:", result["lint"] or "clean")
    return 1 if errors else 0


def cmd_qa(a) -> int:
    spec = load_spec(a.product)
    listing = [Path(p) for p in a.listing] if a.listing else listing_images(a.product)
    target = Path(a.video)
    if target.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
        report = qa.run_image_qa(spec, listing, target, use_vision=not a.no_vision)
    else:
        report = qa.run_video_qa(spec, listing, target, use_vision=not a.no_vision, caption=a.caption)
    print(report.summary())
    out = Path(a.out) if a.out else product_dir(a.product) / "qa" / f"{target.stem}.json"
    write_json(out, report.to_dict())
    print(f"report: {out}")
    return {"pass": 0, "warn": 1 if a.strict else 0, "fail": 1}[report.overall]


def cmd_serve(a) -> int:
    import uvicorn
    uvicorn.run("app.server:app", host=a.host, port=a.port, reload=False)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="ugc")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("plan", help="write prompts for a product")
    s.add_argument("product")
    s.set_defaults(fn=cmd_plan)
    s = sub.add_parser("qa", help="check a video or avatar image against the listing")
    s.add_argument("product")
    s.add_argument("video", help="video file, or an image for avatar QA")
    s.add_argument("--listing", nargs="*", help="override listing photos")
    s.add_argument("--caption", help="lint a caption too")
    s.add_argument("--no-vision", action="store_true", help="skip the Claude vision review")
    s.add_argument("--strict", action="store_true", help="exit 1 on warnings")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_qa)
    s = sub.add_parser("serve", help="run the local web app")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(fn=cmd_serve)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
