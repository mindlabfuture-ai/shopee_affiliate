# Shopee / TikTok Shop affiliate: AI UGC video automation

Niche: ladies' fashion. Buyers are meticulous: the garment in the video must look
exactly like the listing. Rules below were researched 2026-10-18 from official
sources; re-verify periodically (policies changed several times in 2026).

Sources:
- Shopee Video AI-Generated Content Guidelines (eff. 1 Aug 2026): https://help.shopee.sg/portal/10/article/201777
- TikTok Shop AI-Generated Content Restrictions and Requirements: https://seller-us.tiktok.com/university/essay?knowledge_id=491489038501663&lang=en
- TikTok Community Guidelines: https://www.tiktok.com/community-guidelines

Caveat: the Shopee page is the SG help center and the TikTok page is US Shop. Check
the PH versions for local differences.

## Hard rules for any generated video

1. **Never alter the product.** No change to color, size, material, shape, weight,
   design, print/pattern, neckline, sleeve/hem length, buttons/zippers, bundle
   contents or accessories vs. the listing. No fake 3D/rotation of flat products, no
   glow/lighting effects the item doesn't have. Variant video = that colorway only.
2. **Product comes from real assets.** Condition generation on the seller's real
   product photos/clips (image-to-video / reference). AI changes scene, model, camera
   only, never the garment itself.
3. **Always disclose AI.** Shopee: "AI-generated Inform" toggle ON before publishing
   (missing it = 3 penalty points + removal). TikTok: AI-generated toggle, plus
   caption/sticker note. Never falsely label real footage as AI either.
4. **Realistic results only.** No exaggerated before/after or body-change claims. AI
   model fit/stretch/length must match the size chart.
5. **No invented stories or testimonials.** No fabricated crisis/sad stories, fake
   reviews, fake "experts", or real people's likeness/voice/trademarks without
   permission. No minors.
6. **Check brand terms.** Some brands ban AI affiliate content and revoke commissions.
7. **Keep real footage in the mix** (unboxing, fabric close-up, try-on) as a fallback.

## Pipeline requirements (automation)

- Add a QA gate before publishing: compare generated frames against the listing
  photos (color delta, print/pattern, neckline, length, hardware). Reject on any drift.
  Human review of every video before posting at least at first.
- Store per video: source product assets, model + prompt used, QA result, and the
  disclosure toggle state, so there is an audit trail.
- Keep a per-platform publish checklist; the AI toggle step must not be skippable.
- Unverified: third parties claim labeled AI videos on Shopee may lose standard
  affiliate commissions (still earning AMS commissions). Test one video before scaling.

## Penalties to avoid

- Shopee: 3+ pts video removed; 4-14 pts lose recommendation/discovery; 15+ pts
  permanent posting ban and loss of Shopee commission eligibility. Points reset
  1 Jan / 1 Apr / 1 Jul / 1 Oct.
- TikTok Shop: warnings, posting limits, product links removed, Creator Health Rating
  deductions, disabled commission withdrawal, loss of promotion privileges, ban.

## fal.ai

fal has an official MCP server for running models (generation): https://fal.ai/docs/documentation/setting-up/mcp
- URL: `https://mcp.fal.ai/mcp-relay` (Streamable HTTP), OAuth sign-in via browser.
- Tools: search_models, get_model_schema, get_pricing, recommend_model, run_model,
  submit_job, check_job, get_job_result, cancel_job, upload_file, search_docs.
- Video models: use submit_job / check_job / get_job_result (don't resubmit to poll,
  that creates a new billable job). Check get_pricing before generating.
- Docs-only server (no model runs): https://fal.ai/docs/mcp
- For the production pipeline use the fal API/SDK with an API key from env vars
  (never commit keys). MCP is for interactive exploration and prototyping.

## Code in this repo (UGC Studio)

Local app, built 2026-10-18. See README.md. Run: `python -m ugc serve`; tests: `python -m pytest`.

- `ugc/rules.py` encodes the compliance rules above; keep it in sync with this file.
- `ugc/planner.py`: Claude Sonnet plans 3-4 distinct outdoor storyboards (lint enforces >=3, different locations, all required shots); each has its own avatar still prompt + clips. Avatar files are named avatar_s<storyboard>_NN.png and the approved one decides which clips are generated. **Code**, not the model, prepends the garment-lock clause,
  forces `generate_audio: false`, 9:16, clip lengths of 4/6/8 s, and lints for alteration verbs, speech/audio and
  risky caption words (revises up to twice).
- `ugc/qa/`: technical (ffmpeg), color (CIELAB), vision (Claude). `python -m ugc qa <product> <video|image>`.
- Lean stack, configurable by env: `UGC_IMAGE_ENDPOINT` (fal-ai/nano-banana/edit), `UGC_VIDEO_ENDPOINT`
  (fal-ai/veo3.1/lite/image-to-video, **deprecated by fal on 2026-11-17**), `UGC_VIDEO_RESOLUTION` (720p).
- Cost rates in `ugc/config.py` are estimates from third-party pages, not fal's live prices. Daily cap `UGC_DAILY_CAP_USD`.
- Avatar image is QA'd before any video is generated (cheap gate). Videos are silent; music is added in TikTok.
- Secrets only via env vars (ANTHROPIC_API_KEY, FAL_KEY, APP_PASSWORD). Never commit them or `data/`.
- Not yet verified live: real fal generation and real Claude calls (no keys in the build session). Tests mock both.
- Possible later: deploy API on Railway and UI on Netlify (see earlier cost estimate); not needed for local use.
- Manual mode: `/avatar/upload` (stills from ChatGPT, QA'd like generated ones) and `/clips/upload` (clips from Flow: joined in name order, scaled to the first clip, audio stripped, QA'd). Works without FAL_KEY.
- Providers: `ugc/providers/` (`fal`, `deapi`), chosen by `UGC_PROVIDER`. deAPI = API v2 (docs.deapi.ai): multipart POST ->
  `data.request_id` -> poll `/api/v2/jobs/{id}` -> `result_url`; auth `Authorization: Bearer $DEAPI_API_KEY`; limits and model
  slugs come live from `/api/v2/models`; `python -m ugc deapi-models` lists them. Built against the docs and a fake of the API in
  `tests/test_deapi.py`; not yet run against the real service. Never send the key to the `result_url` host (tested).
- Deployed 2026-10-18 (accounts: Railway user mindlabfuture-ai, Netlify mindlabfutureai@gmail.com): Railway project
  bfb3b1cb-877b-4741-b4fb-18e5dba3413c / service ugc-api fba2eafe-7708-4afe-905b-8ba665f8bde3 / volume /data;
  Netlify site 8cce25f7-6f7f-4485-8bbf-2c1228335c4a (ugc-studio-mindlab). The app is locked (503) until APP_PASSWORD is
  set on Railway. If the Railway domain changes, update the two proxy targets in netlify.toml and redeploy Netlify.
- Permanent avatar: `ugc/avatar.py`, endpoints `/api/avatar*`, files under `data/avatar/` (base.png, sheet/<view>.png + .json check, history/). Product shoots use `avatar.reference_files()` as identity references (provider `avatar_ref` may be a list). Face-lock wording lives in `avatar.FACE_LOCK`; views are made front first, then closeup/left/right with base+front as references.
