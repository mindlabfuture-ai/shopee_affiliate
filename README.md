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
