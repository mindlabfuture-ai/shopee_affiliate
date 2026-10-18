"""Local web app (FastAPI). Binds to 127.0.0.1; set APP_PASSWORD to require a login."""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from fastapi import Body, Cookie, Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse

from ugc import avatar as avatar_sheet
from ugc import config, costs, fal_gen, llm, planner, providers, qa
from ugc.imaging import IMG_EXT
from ugc.spec import (Feature, ProductSpec, list_slugs, listing_dir, listing_images, load_spec, product_dir,
                      read_json, save_spec, slugify, write_json)

app = FastAPI(title="UGC pipeline")
STATIC = Path(__file__).parent / "static"
pool = ThreadPoolExecutor(max_workers=2)
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()
MAX_UPLOAD = 25 * 1024 * 1024
MAX_VIDEO = 300 * 1024 * 1024
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".webm"}


# ---------- auth ----------
LOGIN_WINDOW_S, LOGIN_MAX_FAILURES = 600, 8
_failures: list[float] = []


def _password() -> str | None:
    """APP_PASSWORD, or None for open local use. In production (UGC_REQUIRE_PASSWORD=1) a missing password
    locks the app instead of leaving it open."""
    pw = os.environ.get("APP_PASSWORD")
    if not pw and os.environ.get("UGC_REQUIRE_PASSWORD") == "1":
        raise HTTPException(503, "APP_PASSWORD is not set on the server; the app stays locked until it is")
    return pw


def _token(pw: str) -> str:
    return hmac.new(pw.encode(), b"ugc-session", hashlib.sha256).hexdigest()


def require_auth(ugc_session: str | None = Cookie(default=None)) -> None:
    pw = _password()
    if pw and not (ugc_session and hmac.compare_digest(ugc_session, _token(pw))):
        raise HTTPException(401, "login required")


@app.get("/health", include_in_schema=False)
def health():
    return {"ok": True}  # no data; used by Railway's healthcheck


@app.post("/api/login")
def login(response: Response, body: dict = Body(...)):
    pw = _password()
    if pw:
        now = time.time()
        _failures[:] = [t for t in _failures if now - t < LOGIN_WINDOW_S]
        if len(_failures) >= LOGIN_MAX_FAILURES:  # single-user app: one global lockout is enough
            wait = max(1, int(LOGIN_WINDOW_S - (now - _failures[0])))
            raise HTTPException(429, f"Too many wrong passwords. Try again in {-(-wait // 60)} min.",
                                headers={"Retry-After": str(wait)})
        if not hmac.compare_digest(str(body.get("password", "")), pw):
            _failures.append(now)
            left = LOGIN_MAX_FAILURES - len(_failures)
            raise HTTPException(401, f"Wrong password. {left} attempt{'s' if left != 1 else ''} left before a "
                                     f"{LOGIN_WINDOW_S // 60}-minute lockout." if left else
                                     f"Wrong password. Locked for {LOGIN_WINDOW_S // 60} minutes.")
        _failures.clear()
        response.set_cookie("ugc_session", _token(pw), httponly=True, samesite="strict", max_age=30 * 86400,
                            secure=os.environ.get("UGC_SECURE_COOKIE") == "1")
    return {"ok": True}


# ---------- jobs ----------
def start_job(label: str, fn, slug: str | None = None) -> str:
    """Run fn in the background. The last failure per product is saved to disk, so the page can still show
    it after a reload or a server restart (jobs themselves live in memory only)."""
    jid = uuid.uuid4().hex[:10]
    jobs[jid] = {"id": jid, "label": label, "status": "running", "result": None, "error": None}
    err_file = product_dir(slug) / "last_error.json" if slug else None

    def run():
        try:
            res = fn()
            with jobs_lock:
                jobs[jid].update(status="done", result=res)
            if err_file:
                err_file.unlink(missing_ok=True)
        except Exception as e:  # surfaced to the UI
            traceback.print_exc()
            msg = f"{type(e).__name__}: {e}"
            with jobs_lock:
                jobs[jid].update(status="error", error=msg)
            if err_file:
                write_json(err_file, {"label": label, "error": msg, "at": time.strftime("%Y-%m-%d %H:%M:%S")})

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
        "clips": [rel(p) for p in sorted((d / "clips").glob("clip_*.*"))] if (d / "clips").exists() else [],
        "final": rel(d / "final.mp4") if (d / "final.mp4").exists() else None,
        "video_qa": read_json(d / "qa" / "final.json"),
        "last_error": read_json(d / "last_error.json"),
    }


@app.get("/api/state", dependencies=[Depends(require_auth)])
def state():
    prov = providers.get_provider()
    missing = ([] if os.environ.get("ANTHROPIC_API_KEY") else ["ANTHROPIC_API_KEY"]) + prov.missing_env()
    return {"products": list_slugs(), "spent_today": costs.today_total(), "cap": config.DAILY_CAP_USD,
            "config": {**prov.describe(), "claude": config.CLAUDE_MODEL},
            "missing": missing, "generation_ready": not prov.missing_env()}


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

    return {"job": start_job(f"plan {slug}", run, slug=slug)}


# ---------- avatar images ----------
def _qa_avatar(d: Path, spec: ProductSpec, listing: list[Path], img: Path) -> None:
    try:
        rep = qa.run_image_qa(spec, listing, img, use_vision=bool(os.environ.get("ANTHROPIC_API_KEY")))
        write_json(d / "qa" / f"{img.stem}.json", rep.to_dict())
    except Exception as e:
        write_json(d / "qa" / f"{img.stem}.json", {"overall": "error", "error": str(e), "checks": []})



@app.post("/api/products/{slug}/avatar/estimate", dependencies=[Depends(require_auth)])
def avatar_estimate(slug: str, body: dict = Body(default={})):
    return {"usd": providers.get_provider().estimate_images(int(body.get("n", 2)))}


@app.post("/api/products/{slug}/avatar", dependencies=[Depends(require_auth)])
def avatar(slug: str, body: dict = Body(...)):
    n = max(1, min(int(body.get("n", 2)), 4))
    prov = providers.get_provider()
    est = prov.estimate_images(n)
    if float(body.get("confirm_usd", 0)) < est:
        raise HTTPException(400, f"confirm the estimated cost (${est:.2f})")
    d = product_dir(slug)
    p = read_json(d / "plan.json")
    if not p:
        raise HTTPException(400, "generate a plan first")
    sb_idx = _storyboard(p, body.get("storyboard", 0))["index"]
    spec, listing = load_spec(slug), listing_images(slug)
    avatar_refs = avatar_sheet.reference_files()
    prompt = p["storyboards"][sb_idx]["avatar_image"]["prompt"]
    if avatar_refs:
        prompt += (" The last reference images show the avatar: keep her face, hair and body the same, but do not "
                   "copy the clothing she wears there. She wears only the garment described above, shown in the "
                   "first reference photos.")

    def run():
        costs.guard(est)
        imgs = prov.generate_avatar_images(
            prompt, listing[:3], avatar_refs, n, d / "avatars", f"avatar_s{sb_idx}",
            on_image=lambda img: _qa_avatar(d, spec, listing, img))  # cheap gate before any video money is spent
        return [i.name for i in imgs]

    return {"job": start_job(f"avatar {slug}", run, slug=slug)}


@app.post("/api/products/{slug}/avatar/upload", dependencies=[Depends(require_auth)])
def avatar_upload(slug: str, storyboard: int = Form(0), photos: list[UploadFile] = File(...)):
    """Manual mode: avatar stills made elsewhere (e.g. ChatGPT). Checked against the listing like generated ones."""
    d, spec, listing = product_dir(slug), load_spec(slug), listing_images(slug)
    plan_ = read_json(d / "plan.json")
    if plan_:
        _storyboard(plan_, storyboard)
    decoded = []
    for f in photos:
        if Path(f.filename or "").suffix.lower() not in IMG_EXT:
            raise HTTPException(400, f"unsupported image type: {f.filename}")
        data = f.file.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, f"{f.filename} is over 25 MB")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, f"cannot read image: {f.filename}")
        decoded.append(img)
    adir = d / "avatars"
    adir.mkdir(parents=True, exist_ok=True)
    saved = []
    for img in decoded:
        n = len(list(adir.glob(f"avatar_s{storyboard}_*.png"))) + 1
        path = adir / f"avatar_s{storyboard}_{n:02d}.png"
        cv2.imwrite(str(path), img)
        saved.append(path)

    def run():
        for img in saved:
            _qa_avatar(d, spec, listing, img)
        return [i.name for i in saved]

    return {"job": start_job(f"avatar upload {slug}", run, slug=slug)}


@app.post("/api/products/{slug}/clips/upload", dependencies=[Depends(require_auth)])
def clips_upload(slug: str, videos: list[UploadFile] = File(...), caption: str = Form("")):
    """Manual mode: clips made elsewhere (e.g. Flow), in playing order. Joined, audio stripped, then QA'd."""
    d, spec, listing = product_dir(slug), load_spec(slug), listing_images(slug)
    blobs = []
    for f in videos:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in VIDEO_EXT:
            raise HTTPException(400, f"unsupported video type: {f.filename}")
        data = f.file.read(MAX_VIDEO + 1)
        if len(data) > MAX_VIDEO:
            raise HTTPException(413, f"{f.filename} is over 300 MB")
        blobs.append((ext, data))
    shutil.rmtree(d / "clips", ignore_errors=True)  # replaces any earlier clips
    (d / "clips").mkdir(parents=True)
    paths = []
    for i, (ext, data) in enumerate(blobs, 1):
        paths.append(d / "clips" / f"clip_{i:02d}{ext}")
        paths[-1].write_bytes(data)
    plan_ = read_json(d / "plan.json") or {}

    def run():
        final = fal_gen.join_clips(paths, d / "final.mp4")
        rep = qa.run_video_qa(spec, listing, final, use_vision=bool(os.environ.get("ANTHROPIC_API_KEY")),
                              caption=caption or plan_.get("tiktok_caption"))
        write_json(d / "qa" / "final.json", rep.to_dict())
        return rep.to_dict()

    return {"job": start_job(f"clips upload {slug}", run, slug=slug)}


# ---------- permanent avatar (four-view identity sheet) ----------
def _current_hairstyle() -> str:
    """The saved choice; else the suggested style for a brand-new avatar, but empty ("keep her hair as in the
    references") once any view exists, so a missing view matches the ones you already have."""
    saved = read_json(avatar_sheet.avatar_dir() / "settings.json")
    if saved and "hairstyle" in saved:
        return saved["hairstyle"]
    any_view = any(avatar_sheet.view_path(v).exists() for v in avatar_sheet.VIEWS)
    return "" if any_view else avatar_sheet.DEFAULT_HAIRSTYLE


def _avatar_state() -> dict:
    sd = avatar_sheet.sheet_dir()
    url = lambda p: f"/avatar-files/{p.relative_to(avatar_sheet.avatar_dir()).as_posix()}"  # noqa: E731
    views = {}
    for v in avatar_sheet.DISPLAY_ORDER:
        pth = avatar_sheet.view_path(v)
        views[v] = {"url": url(pth) + f"?t={int(pth.stat().st_mtime)}", "check": read_json(sd / f"{v}.json")} if pth.exists() else None
    base = avatar_sheet.base_path()
    return {"base": url(base) + f"?t={int(base.stat().st_mtime)}" if base.exists() else None, "views": views,
            "hairstyle": _current_hairstyle(),
            "default_hairstyle": avatar_sheet.DEFAULT_HAIRSTYLE,
            "prompts": {v: avatar_sheet.view_prompt(v, _current_hairstyle()) for v in avatar_sheet.DISPLAY_ORDER}}


@app.get("/api/avatar", dependencies=[Depends(require_auth)])
def avatar_get():
    return _avatar_state()


@app.post("/api/avatar/base", dependencies=[Depends(require_auth)])
def avatar_base(photo: UploadFile = File(...)):
    """The one source photo every view is made from. Replacing it does not delete existing views."""
    if Path(photo.filename or "").suffix.lower() not in IMG_EXT:
        raise HTTPException(400, "unsupported image type")
    data = photo.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "image is over 25 MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "cannot read that image")
    avatar_sheet.save_base(img)
    return _avatar_state()


def _read_image_upload(photo: UploadFile) -> np.ndarray:
    if Path(photo.filename or "").suffix.lower() not in IMG_EXT:
        raise HTTPException(400, "unsupported image type")
    data = photo.file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "image is over 25 MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "cannot read that image")
    return img


@app.post("/api/avatar/view/{view}", dependencies=[Depends(require_auth)])
def avatar_view_upload(view: str, photo: UploadFile = File(...)):
    """Use an image you made elsewhere for one slot (closeup, front, left, right). Replaces what is there."""
    if view not in avatar_sheet.VIEWS:
        raise HTTPException(400, f"unknown view {view!r}")
    avatar_sheet.save_view(view, _read_image_upload(photo))
    return _avatar_state()


@app.delete("/api/avatar/view/{view}", dependencies=[Depends(require_auth)])
def avatar_view_remove(view: str):
    if view not in avatar_sheet.VIEWS:
        raise HTTPException(400, f"unknown view {view!r}")
    avatar_sheet.remove_view(view)
    return _avatar_state()


@app.post("/api/avatar/clear", dependencies=[Depends(require_auth)])
def avatar_clear(body: dict = Body(...)):
    """Delete the whole permanent avatar. Product shoot images are not touched."""
    if body.get("confirm") is not True:
        raise HTTPException(400, "send {\"confirm\": true} to delete the avatar")
    return {"deleted": avatar_sheet.clear_all(), **_avatar_state()}


@app.post("/api/avatar/sheet/estimate", dependencies=[Depends(require_auth)])
def avatar_sheet_estimate(body: dict = Body(default={})):
    n = len(body.get("views") or avatar_sheet.VIEW_ORDER)
    return {"usd": providers.get_provider().estimate_images(n), "images": n}


@app.post("/api/avatar/sheet", dependencies=[Depends(require_auth)])
def avatar_sheet_generate(body: dict = Body(...)):
    """Generate (or regenerate) views of the permanent avatar. hairstyle: '' keeps her hair as in the photo."""
    views = [v for v in avatar_sheet.VIEW_ORDER if v in (body.get("views") or avatar_sheet.VIEW_ORDER)]
    if not views:
        raise HTTPException(400, "no valid views requested")
    if not avatar_sheet.base_path().exists():
        raise HTTPException(400, "upload the base avatar photo first")
    prov = providers.get_provider()
    est = prov.estimate_images(len(views))
    if float(body.get("confirm_usd", 0)) < est:
        raise HTTPException(400, f"confirm the estimated cost (${est:.2f})")
    hairstyle = str(body.get("hairstyle", "")).strip()
    write_json(avatar_sheet.avatar_dir() / "settings.json", {"hairstyle": hairstyle})

    def run():
        costs.guard(est)
        client = llm.get_client() if os.environ.get("ANTHROPIC_API_KEY") else None
        return {v: avatar_sheet.generate_view(prov, v, hairstyle, client) for v in views}

    return {"job": start_job("avatar sheet", run)}


@app.get("/avatar-files/{path:path}", dependencies=[Depends(require_auth)])
def avatar_files(path: str):
    root = avatar_sheet.avatar_dir().resolve()
    f = (root / path).resolve()
    if root not in f.parents or not f.is_file() or f.suffix.lower() not in IMG_EXT:
        raise HTTPException(404)
    return FileResponse(f)


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
    return {"usd": providers.get_provider().estimate_video(seconds), "seconds": seconds}


@app.post("/api/products/{slug}/video", dependencies=[Depends(require_auth)])
def video(slug: str, body: dict = Body(...)):
    d = product_dir(slug)
    p, appr = read_json(d / "plan.json"), read_json(d / "approved.json")
    if not p or not appr:
        raise HTTPException(400, "need a plan and an approved avatar image")
    sb = _approved_storyboard(d, p)
    prov = providers.get_provider()
    est = prov.estimate_video(sum(c["duration_s"] for c in sb["clips"]))
    if float(body.get("confirm_usd", 0)) < est:
        raise HTTPException(400, f"confirm the estimated cost (${est:.2f})")
    spec, listing = load_spec(slug), listing_images(slug)
    image = d / "avatars" / appr["avatar"]

    def run():
        manifest = d / "clips" / "manifest.json"
        want = {"avatar": appr["avatar"], "storyboard": sb["index"]}
        if read_json(manifest) != want:  # clips from another avatar or storyboard must not linger
            shutil.rmtree(d / "clips", ignore_errors=True)
        (d / "clips").mkdir(parents=True, exist_ok=True)
        write_json(manifest, want)
        def done(c) -> bool:
            f = d / "clips" / f"clip_{c['index']:02d}.mp4"
            return f.exists() and f.stat().st_size > 0

        todo = [c for c in sb["clips"] if not done(c)]  # finished clips are reused, never paid for twice
        costs.guard(prov.estimate_video(sum(c["duration_s"] for c in todo)))
        clips = []
        for c in sb["clips"]:
            path = d / "clips" / f"clip_{c['index']:02d}.mp4"
            if c in todo:
                prov.generate_clip(c["prompt"], c["negative_prompt"], image, c["duration_s"], path)
            clips.append(path)
        final = fal_gen.join_clips(clips, d / "final.mp4")
        rep = qa.run_video_qa(spec, listing, final, use_vision=bool(os.environ.get("ANTHROPIC_API_KEY")),
                              caption=p["tiktok_caption"])
        write_json(d / "qa" / "final.json", rep.to_dict())
        return rep.to_dict()

    return {"job": start_job(f"video {slug}", run, slug=slug)}


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

    return {"job": start_job(f"qa {slug}", run, slug=slug)}


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
