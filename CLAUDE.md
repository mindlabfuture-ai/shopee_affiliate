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
