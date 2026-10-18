# UGC Studio

Local tool for making garment-faithful AI UGC videos for Shopee and TikTok Shop (ladies' fashion).
Upload the real listing photos, let Claude plan 3-4 different outdoor fashion-shoot storyboards (each with its own location, avatar still and clips), generate silent clips on fal, and gate every
step with a QA check against the listing.

```
listing photos -> Claude plan -> avatar image (fal) -> QA gate -> video clips (fal) -> join, strip audio -> QA report
```

Lean stack (about $1 per finished video, estimates): `fal-ai/nano-banana/edit` for the avatar still,
`fal-ai/veo3.1/lite/image-to-video` for clips (silent, 720p), Claude Sonnet 5.5 for planning and vision QA.
fal says the Veo 3.1 Lite endpoint is deprecated on 2026-11-17. Swap models with `UGC_VIDEO_ENDPOINT`.

## Run it

```bash
pip install -e ".[dev]"            # needs ffmpeg on PATH
export ANTHROPIC_API_KEY=...       # planner + vision QA
export FAL_KEY=...                 # generation
export APP_PASSWORD=...            # optional login for the page
python -m ugc serve                # http://127.0.0.1:8000
```

Workflow in the page: add a product with its listing photos (and the exact color hex), generate the plan,
generate avatar images (each is checked before you spend on video), approve one, generate the video, read
the QA report. Every paid step shows an estimated cost and asks first; `UGC_DAILY_CAP_USD` (default 15) stops spend.
You can also check a video made elsewhere (for example in Flow) with "Check this video".

## Providers: fal or deAPI

`UGC_PROVIDER=fal` (default) or `UGC_PROVIDER=deapi`. deAPI gives $5 of free credit on signup (no card), which
is enough to try the pipeline.

```bash
export UGC_PROVIDER=deapi DEAPI_API_KEY=...
python -m ugc deapi-models     # lists image-edit and image-to-video models with their limits
```

The app picks a model automatically (an image-edit model that accepts several reference images, and an LTX
video model); pin them with `UGC_DEAPI_IMAGE_MODEL` / `UGC_DEAPI_VIDEO_MODEL`. Size comes from
`UGC_DEAPI_SIZE` (default 576x1024) and is clamped to each model's limits. Notes:

- Model limits are read live from deAPI. If a model caps clip length below the plan (for example about 5 s),
  the clip is shorter than planned and a message is printed; the join still works.
- Costs are quoted with deAPI's price endpoints and treated as USD; the daily cap still applies.
- Basic accounts have small limits (some models 3 requests/minute, status and price calls 200/day), so the app
  caches quotes and polls every 8 s (`UGC_DEAPI_POLL_S`). Any top-up moves the account to Premium limits.
- The deAPI dashboard shows five image-to-video models: LTX-2.5 22B Distilled INT8, Seedance 2.5, MiniMax H3 33B
  Turbo INT8, LTX-2.3 22B and LTX-Video 13B. Its banner says a $10 top-up unlocks "premium models", and I could not
  see which ones are locked on a free account. If a call returns 403, pin another model.
- LTX-Video 13B tops out at 768x768 and 1-4 s clips. The size is scaled down keeping 9:16, and for models like
  this set `UGC_MAX_CLIP_S=4` so the planner writes several 4 s clips per storyboard instead of 6-8 s ones.
- Check deAPI's terms for commercial use before posting results; I could not see them.

## Manual mode (no generation spend)

Make the images and clips in tools you already use and let the app do the planning and checking:

1. Add the product, click **Generate plan**, and click a prompt to copy it.
2. Make the avatar stills in ChatGPT with the storyboard's avatar prompt and the listing photo, then use
   **Upload and check** in step 3. Each still gets the same garment check as a generated one. Approve one.
3. Make the clips in Flow (or any tool) from that still. Pick them all in **Join and check** in step 4:
   they are joined in file-name order (clip1, clip2, ...), resized to the first clip, audio stripped, then QA'd.
4. Read the QA report in step 5. Only Claude calls cost money (about $0.10 per video), and manual mode works
   without `FAL_KEY`.

## CLI

```bash
python -m ugc plan <product>                       # writes data/products/<product>/plan.json
python -m ugc qa <product> video.mp4 [--no-vision] [--caption "..."] [--strict]
python -m ugc qa <product> avatar.png              # image gate
```

`examples/` has starter specs; copy one to `data/products/<slug>/product.json` and put photos in `listing/`.

## What QA checks

| Check | How |
|---|---|
| format, duration, audio track, black frames, scene cuts | ffprobe / ffmpeg |
| garment color vs listing, stability across frames, color jump across joins | CIELAB distance on frames |
| every listed feature, frame-level garment morphing | Claude vision, listing vs sampled frames |
| caption wording | risky-phrase lint |

Exit code is 1 on any failure. Color and stability thresholds in `ugc/config.py` are starting points;
set `color_hex` in the product spec for an exact target and calibrate on videos you have already accepted.
Color checks cannot see shape or print changes, which is what the vision review is for. Neither replaces
looking at the video yourself before posting.

## Layout

`ugc/` core (planner, QA, fal calls, costs) · `app/` FastAPI + single-page UI · `tests/` pytest ·
`data/` your products, outputs and spend ledger (git-ignored).

Rules this encodes: see `CLAUDE.md`.

## Deployed (Railway + Netlify)

- **API:** Railway project `ugc-studio`, service `ugc-api`, built from `Dockerfile` on branch `claude/eager-babbage-r72e68`,
  volume `ugc-data` mounted at `/data`, URL `https://ugc-api-production.up.railway.app`.
- **UI:** Netlify site `ugc-studio-mindlab` (`https://ugc-studio-mindlab.netlify.app`) serves `app/static` and proxies
  `/api/*` and `/files/*` to Railway (`netlify.toml`), so the browser sees one origin.
- The container sets `UGC_REQUIRE_PASSWORD=1`: until `APP_PASSWORD` is set on Railway the app answers 503 to everything but
  `/health`. Set these in Railway > ugc-api > Variables (never in the repo): `APP_PASSWORD`, `ANTHROPIC_API_KEY`, and the
  generation key for your provider (`DEAPI_API_KEY` with `UGC_PROVIDER=deapi`, or `FAL_KEY`). Optional: `UGC_MAX_CLIP_S=4`,
  `UGC_DAILY_CAP_USD`, `UGC_DEAPI_*` model pins.
- Login: 8 failed attempts lock logins for 10 minutes; the session cookie is HttpOnly, Secure, SameSite=Strict.
- UI changes: the Netlify site was deployed by upload, not Git. Redeploy with the command from the Netlify tool, or link the
  site to the GitHub repo in Netlify settings. API changes redeploy automatically when `ugc/`, `app/` or the Dockerfile change.
