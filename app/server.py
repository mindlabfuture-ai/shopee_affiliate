"""Local web app (FastAPI). Binds to 127.0.0.1; set APP_PASSWORD to require a login."""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import shutil
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import Body, Cookie, Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse

from ugc import config, costs, fal_gen, planner, qa
from ugc.imaging import IMG_EXT
from ugc.spec import (Feature, ProductSpec, list_slugs, listing_dir, listing_images, load_spec, product_dir,
                      read_json, save_spec, slugify, write_json)

app = FastAPI(title="UGC pipeline")
STATIC = Path(__file__).parent / "static"
pool = ThreadPoolExecutor(max_workers=2)
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
MAX_UPLOAD = 25 * 1024 * 1024


# ---------- auth ----------
def _token() -> str:
    return hmac.new(os.environ["APP_PASSWORD"].encode(), b"ugc-session", hashlib.sha256).hexdigest()


def require_auth(ugc_session: str | None = Cookie(default=None)) -> None:
    if os.environ.get("APP_PASSWORD") and not (ugc_session and hmac.compare_digest(ugc_session, _token())):
        raise HTTPException(401, "login required")


@app.post("/api/login")
def login(response: Response, body: dict = Body(...)):
    pw = os.environ.get("APP_PASSWORD")
    if pw and not hmac.compare_digest(str(body.get("password", "")), pw):
        raise HTTPException(401, "wrong password")
    if pw:
        response.set_cookie("ugc_session", _token(), httponly=True, samesite="strict")
    return {"ok": True}


# ---------- jobs ----------
def start_job(label: str, fn) -> str:
    jid = uuid.uuid4().hex[:10]
    jobs[jid] = {"id": jid, "label": label, "status": "running", "result": None, "error": None}

    def run():
        try:
            res = fn()
            with jobs_lock:
                jobs[jid].update(status="done", result=res)
        except Exception as e:  # surfaced to the UI
            traceback.print_exc()
            with jobs_lock:
                jobs[jid].update(status="error", error=f"{type(e).__name__}: {e}")

    pool.submit(run)
    return jid


@app.get("/api/jobs/{jid}", dependencies=[Depends(require_auth)])
def get_job(jid: str):
    if jid not in jobs:
        raise HTTPException(404)
    return jobs[jid]


# ---------- products ----------
def _parse_features(text: str) -> list[Feature]:
    out = []
    for line in text.splitlines():
        if line.strip():
            name, _, desc = line.partition(":")
            out.append(Feature(name=name.strip(), description=(desc or name).strip()))
    return out


def _save_uploads(files: list[UploadFile], dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in IMG_EXT:
            raise HTTPException(400, f"unsupported image type: {f.filename}")
        data = f.file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, f"{f.filename} is over 25 MB")
        (dest / f"{uuid.uuid4().hex[:8]}{ext}").write_bytes(data)
        n += 1
    return n


def _state(slug: str) -> dict:
    spec = load_spec(slug)
    d = product_dir(slug)
    rel = lambda p: f"/files/{slug}/{p.relative_to(d).as_posix()}"  # noqa: E731
    return {
        "spec": spec.model_dump(),
        "listing": [rel(p) for p in listing_images(slug)],
        "plan": read_json(d / "plan.json"),
        "avatars": [{"url": rel(p), "name": p.name, "qa": read_json(d / "qa" / f"{p.stem}.json")}
                    for p in sorted((d / "avatars").glob("avatar_*.png"))] if (d / "avatars").exists() else [],
        "approved_avatar": (read_json(d / "approved.json") or {}).get("avatar"),
        "clips": [rel(p) for p in sorted((d / "clips").glob("clip_*.mp4"))] if (d / "clips").exists() else [],
        "final": rel(d / "final.mp4") if (d / "final.mp4").exists() else None,
        "video_qa": read_json(d / "qa" / "final.json"),
    }


@app.get("/api/state", dependencies=[Depends(require_auth)])
def state():
    return {"products": list_slugs(), "spent_today": costs.today_total(), "cap": config.DAILY_CAP_USD,
            "config": {"image": config.IMAGE_ENDPOINT, "video": config.VIDEO_ENDPOINT, "claude": config.CLAUDE_MODEL,
                       "resolution": config.VIDEO_RESOLUTION},
            "keys": {"anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")), "fal": bool(os.environ.get("FAL_KEY"))}}


@app.post("/api/products", dependencies=[Depends(require_auth)])
def create_product(name: str = Form(...), colorway: str = Form(...), color_hex: str = Form(""),
                   length: str = Form(""), fabric: str = Form(""), features: str = Form(""),
                   required_shots: str = Form("front,back,full_length"), avoid: str = Form(""),
                   category: str = Form("dress"), photos: list[UploadFile] = File(default=[])):
    slug = slugify(name)
    hex_ = color_hex.strip() or None
    if hex_ and not (len(hex_.lstrip("#")) == 6 and all(c in "0123456789abcdefABCDEF" for c in hex_.lstrip("#"))):
        raise HTTPException(400, "color_hex must look like #b32d2f")
    spec = ProductSpec(slug=slug, name=name, category=category, colorway=colorway, color_hex=hex_, length=length,
                       fabric=fabric, features=_parse_features(features),
                       required_shots=[s.strip() for s in required_shots.split(",") if s.strip()],
                       avoid=[s.strip() for s in avoid.split(",") if s.strip()])
    save_spec(spec)
    _save_uploads(photos, listing_dir(slug))
    return {"slug": slug}


@app.get("/api/products/{slug}", dependencies=[Depends(require_auth)])
def get_product(slug: str):
    try:
        return _state(slug)
    except FileNotFoundError:
        raise HTTPException(404)


@app.post("/api/products/{slug}/photos", dependencies=[Depends(require_auth)])
def add_photos(slug: str, photos: list[UploadFile] = File(...)):
    load_spec(slug)
    return {"added": _save_uploads(photos, listing_dir(slug))}


@app.post("/api/products/{slug}/plan", dependencies=[Depends(require_auth)])
def plan(slug: str):
    spec, listing = load_spec(slug), listing_images(slug)
    if not listing:
        raise HTTPException(400, "upload listing photos first")

    def run():
        result = planner.make_plan(spec, listing)
        costs.record("claude_plan", 0.07, slug)  # rough estimate, recorded only on success
        write_json(product_dir(slug) / "plan.json", result)
        return result

    return {"job": start_job(f"plan {slug}", run)}


# ---------- avatar images ----------
@app.post("/api/products/{slug}/avatar/estimate", dependencies=[Depends(require_auth)])
def avatar_estimate(slug: str, body: dict = Body(default={})):
    return {"usd": costs.estimate_images(int(body.get("n", 2)))}


@app.post("/api/products/{slug}/avatar", dependencies=[Depends(require_auth)])
def avatar(slug: str, body: dict = Body(...)):
    n = max(1, min(int(body.get("n", 2)), 4))
    est = costs.estimate_images(n)
    if float(body.get("confirm_usd", 0)) < est:
        raise HTTPException(400, f"confirm the estimated cost (${est:.2f})")
    d = product_dir(slug)
    p = read_json(d / "plan.json")
    if not p:
        raise HTTPException(400, "generate a plan first")
    sb_idx = _storyboard(p, body.get("storyboard", 0))["index"]
    spec, listing = load_spec(slug), listing_images(slug)
    avatar_ref = config.DATA_DIR / "avatar_reference.png"
    refs = listing[:3] + ([avatar_ref] if avatar_ref.exists() else [])
    prompt = p["storyboards"][sb_idx]["avatar_image"]["prompt"]
    if avatar_ref.exists():
        prompt += " The last reference image is the avatar: keep her face, hair and body the same."

    def run():
        costs.guard(est)
        imgs = fal_gen.generate_avatar_images(prompt, refs, n, d / "avatars", prefix=f"avatar_s{sb_idx}")
        for img in imgs:  # cheap gate: check each still before any video money is spent
            try:
                rep = qa.run_image_qa(spec, listing, img, use_vision=bool(os.environ.get("ANTHROPIC_API_KEY")))
                write_json(d / "qa" / f"{img.stem}.json", rep.to_dict())
            except Exception as e:
                write_json(d / "qa" / f"{img.stem}.json", {"overall": "error", "error": str(e), "checks": []})
        return [i.name for i in imgs]

    return {"job": start_job(f"avatar {slug}", run)}


@app.post("/api/avatar-reference", dependencies=[Depends(require_auth)])
def avatar_reference(photo: UploadFile = File(...)):
    """Saved avatar used for identity consistency across products."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with (config.DATA_DIR / "avatar_reference.png").open("wb") as f:
        shutil.copyfileobj(photo.file, f)
    return {"ok": True}


@app.post("/api/products/{slug}/approve", dependencies=[Depends(require_auth)])
def approve(slug: str, body: dict = Body(...)):
    name = Path(str(body.get("avatar", ""))).name
    if not (product_dir(slug) / "avatars" / name).exists():
        raise HTTPException(404, "no such avatar")
    write_json(product_dir(slug) / "approved.json", {"avatar": name})
    return {"approved": name}


# ---------- video ----------
def _storyboard(plan: dict, idx) -> dict:
    sbs = plan.get("storyboards") or []
    try:
        return sbs[int(idx)]
    except (IndexError, ValueError, TypeError):
        raise HTTPException(400, "unknown storyboard; regenerate the plan")


def _approved_storyboard(d: Path, plan: dict) -> dict:
    """The storyboard the approved avatar was made for (from its file name), else the first."""
    name = (read_json(d / "approved.json") or {}).get("avatar", "")
    m = re.match(r"avatar_s(\d+)_", name)
    return _storyboard(plan, m.group(1) if m else 0)


@app.post("/api/products/{slug}/video/estimate", dependencies=[Depends(require_auth)])
def video_estimate(slug: str):
    d = product_dir(slug)
    p = read_json(d / "plan.json")
    if not p:
        raise HTTPException(400, "generate a plan first")
    seconds = sum(c["duration_s"] for c in _approved_storyboard(d, p)["clips"])
    return {"usd": costs.estimate_video(seconds), "seconds": seconds}


@app.post("/api/products/{slug}/video", dependencies=[Depends(require_auth)])
def video(slug: str, body: dict = Body(...)):
    d = product_dir(slug)
    p, appr = read_json(d / "plan.json"), read_json(d / "approved.json")
    if not p or not appr:
        raise HTTPException(400, "need a plan and an approved avatar image")
    sb = _approved_storyboard(d, p)
    est = costs.estimate_video(sum(c["duration_s"] for c in sb["clips"]))
    if float(body.get("confirm_usd", 0)) < est:
        raise HTTPException(400, f"confirm the estimated cost (${est:.2f})")
    spec, listing = load_spec(slug), listing_images(slug)
    image = d / "avatars" / appr["avatar"]

    def run():
        costs.guard(est)
        shutil.rmtree(d / "clips", ignore_errors=True)  # clips from a previous storyboard must not linger
        clips = [fal_gen.generate_clip(c["prompt"], c["negative_prompt"], image, c["duration_s"],
                                       d / "clips" / f"clip_{c['index']:02d}.mp4") for c in sb["clips"]]
        final = fal_gen.join_clips(clips, d / "final.mp4")
        rep = qa.run_video_qa(spec, listing, final, use_vision=bool(os.environ.get("ANTHROPIC_API_KEY")),
                              caption=p["tiktok_caption"])
        write_json(d / "qa" / "final.json", rep.to_dict())
        return rep.to_dict()

    return {"job": start_job(f"video {slug}", run)}


@app.post("/api/products/{slug}/qa", dependencies=[Depends(require_auth)])
def qa_upload(slug: str, video: UploadFile = File(...), caption: str = Form("")):
    """QA a video made elsewhere (e.g. from Flow)."""
    d = product_dir(slug)
    spec, listing = load_spec(slug), listing_images(slug)
    dest = d / "uploads" / f"{uuid.uuid4().hex[:8]}{Path(video.filename or 'v.mp4').suffix.lower() or '.mp4'}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    def run():
        rep = qa.run_video_qa(spec, listing, dest, use_vision=bool(os.environ.get("ANTHROPIC_API_KEY")),
                              caption=caption or None)
        write_json(d / "qa" / "final.json", rep.to_dict())
        return rep.to_dict()

    return {"job": start_job(f"qa {slug}", run)}


# ---------- files + UI ----------
@app.get("/files/{slug}/{path:path}", dependencies=[Depends(require_auth)])
def files(slug: str, path: str):
    root = product_dir(slug).resolve()
    f = (root / path).resolve()
    if root not in f.parents or not f.is_file():
        raise HTTPException(404)
    return FileResponse(f)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
