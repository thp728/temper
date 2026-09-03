---
version: alpha
name: Linear-design-analysis
description: "A near-black product-focused marketing canvas built around #010102 (the deepest dark surface of any tool in this collection), light gray text (#f7f8f8), and the signature Linear lavender-blue (#5e6ad2) as the single chromatic accent. The system reads as software-craft documentation: dense and technical. Display type is set in the Linear custom sans (SF Pro Display fallback) at 500–700 with measured negative tracking. Cards are charcoal panels (#0f1011) with hairline borders. The accent lavender appears on the brand mark, focus rings, and a few deliberate CTAs, never decoratively. Page rhythm leans on product UI screenshots framed in dark panels rather than atmospheric color."

colors:
  primary: "#5e6ad2"
  on-primary: "#ffffff"
  primary-hover: "#828fff"
  primary-focus: "#5e69d1"
  ink: "#f7f8f8"
  ink-muted: "#d0d6e0"
  ink-subtle: "#8a8f98"
  ink-tertiary: "#62666d"
  canvas: "#010102"
  surface-1: "#0f1011"
  surface-2: "#141516"
  surface-3: "#18191a"
  surface-4: "#191a1b"
  hairline: "#23252a"
  hairline-strong: "#34343a"
  hairline-tertiary: "#3e3e44"
  inverse-canvas: "#ffffff"
  inverse-surface-1: "#f5f6f6"
  inverse-surface-2: "#f6f7f7"
  inverse-ink: "#000000"
  brand-secure: "#7a7fad"
  semantic-success: "#27a644"
  semantic-overlay: "#000000"

typography:
  display-xl:
    fontFamily: Linear Display
    fontSize: 80px
    fontWeight: 600
    lineHeight: 1.05
    letterSpacing: -3.0px
  display-lg:
    fontFamily: Linear Display
    fontSize: 56px
    fontWeight: 600
    lineHeight: 1.10
    letterSpacing: -1.8px
  display-md:
    fontFamily: Linear Display
    fontSize: 40px
    fontWeight: 600
    lineHeight: 1.15
    letterSpacing: -1.0px
  headline:
    fontFamily: Linear Display
    fontSize: 28px
    fontWeight: 600
    lineHeight: 1.20
    letterSpacing: -0.6px
  card-title:
    fontFamily: Linear Display
    fontSize: 22px
    fontWeight: 500
    lineHeight: 1.25
    letterSpacing: -0.4px
  subhead:
    fontFamily: Linear Display
    fontSize: 20px
    fontWeight: 400
    lineHeight: 1.40
    letterSpacing: -0.2px
  body-lg:
    fontFamily: Linear Text
    fontSize: 18px
    fontWeight: 400
    lineHeight: 1.50
    letterSpacing: -0.1px
  body:
    fontFamily: Linear Text
    fontSize: 16px
    fontWeight: 400
    lineHeight: 1.50
    letterSpacing: -0.05px
  body-sm:
    fontFamily: Linear Text
    fontSize: 14px
    fontWeight: 400
    lineHeight: 1.50
    letterSpacing: 0
  caption:
    fontFamily: Linear Text
    fontSize: 12px
    fontWeight: 400
    lineHeight: 1.40
    letterSpacing: 0
  button:
    fontFamily: Linear Text
    fontSize: 14px
    fontWeight: 500
    lineHeight: 1.20
    letterSpacing: 0
  eyebrow:
    fontFamily: Linear Text
    fontSize: 13px
    fontWeight: 500
    lineHeight: 1.30
    letterSpacing: 0.4px
  mono:
    fontFamily: Linear Mono
    fontSize: 13px
    fontWeight: 400
    lineHeight: 1.50
    letterSpacing: 0

rounded:
  xs: 4px
  sm: 6px
  md: 8px
  lg: 12px
  xl: 16px
  xxl: 24px
  pill: 9999px
  full: 9999px

spacing:
  xxs: 4px
  xs: 8px
  sm: 12px
  md: 16px
  lg: 24px
  xl: 32px
  xxl: 48px
  section: 96px

components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.on-primary}"
    typography: "{typography.button}"
    rounded: "{rounded.md}"
    padding: 8px 14px
  button-primary-pressed:
    backgroundColor: "{colors.primary-focus}"
    textColor: "{colors.on-primary}"
    typography: "{typography.button}"
    rounded: "{rounded.md}"
  button-primary-hover:
    backgroundColor: "{colors.primary-hover}"
    textColor: "{colors.on-primary}"
    typography: "{typography.button}"
    rounded: "{rounded.md}"
  button-secondary:
    backgroundColor: "{colors.surface-1}"
    textColor: "{colors.ink}"
    typography: "{typography.button}"
    rounded: "{rounded.md}"
    padding: 8px 14px
  button-tertiary:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.ink}"
    typography: "{typography.button}"
    rounded: "{rounded.md}"
    padding: 8px 14px
  button-inverse:
    backgroundColor: "{colors.inverse-canvas}"
    textColor: "{colors.inverse-ink}"
    typography: "{typography.button}"
    rounded: "{rounded.md}"
    padding: 8px 14px
  pricing-card:
    backgroundColor: "{colors.surface-1}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.lg}"
    padding: 24px
  pricing-card-featured:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.lg}"
    padding: 24px
  feature-card:
    backgroundColor: "{colors.surface-1}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.lg}"
    padding: 24px
  product-screenshot-card:
    backgroundColor: "{colors.surface-1}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.xl}"
    padding: 24px
  testimonial-card:
    backgroundColor: "{colors.surface-1}"
    textColor: "{colors.ink}"
    typography: "{typography.body-lg}"
    rounded: "{rounded.lg}"
    padding: 32px
  customer-logo-tile:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.ink-subtle}"
    typography: "{typography.caption}"
    rounded: "{rounded.xs}"
    padding: 16px
  text-input:
    backgroundColor: "{colors.surface-1}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.md}"
    padding: 8px 12px
  text-input-focused:
    backgroundColor: "{colors.surface-1}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.md}"
    padding: 8px 12px
  pricing-tab-default:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.ink-subtle}"
    typography: "{typography.button}"
    rounded: "{rounded.pill}"
    padding: 6px 14px
  pricing-tab-selected:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.ink}"
    typography: "{typography.button}"
    rounded: "{rounded.pill}"
    padding: 6px 14px
  cta-banner:
    backgroundColor: "{colors.surface-1}"
    textColor: "{colors.ink}"
    typography: "{typography.headline}"
    rounded: "{rounded.lg}"
    padding: 48px
  changelog-row:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.ink}"
    typography: "{typography.body}"
    rounded: "{rounded.xs}"
    padding: 24px 0
  status-badge:
    backgroundColor: "{colors.surface-2}"
    textColor: "{colors.ink-muted}"
    typography: "{typography.caption}"
    rounded: "{rounded.pill}"
    padding: 2px 8px
  top-nav:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.ink}"
    typography: "{typography.body-sm}"
    rounded: "{rounded.xs}"
    height: 56px
  footer:
    backgroundColor: "{colors.canvas}"
    textColor: "{colors.ink-subtle}"
    typography: "{typography.caption}"
    rounded: "{rounded.xs}"
    padding: 64px 32px
---

## Overview

> **What this file is.** A design analysis of [Linear](https://linear.app)'s
> public marketing site, captured as tokens and rules so the interface in
> `apps/web` has a consistent visual direction to work against. It documents
> someone else's design system rather than describing Temper's own product
> decisions, which live in [docs/adr/](docs/adr/). Linear's name, wordmark and
> typefaces are theirs; nothing here redistributes them.

Linear's marketing canvas is the deepest dark surface in this collection. `{colors.canvas}` is #010102, essentially pure black with a faint blue tint. On top sits a four-step surface ladder (`{colors.surface-1}` through `{colors.surface-4}`) for cards, panels, and lifted tiles, with hairline borders running from `{colors.hairline}` (#23252a) up through `{colors.hairline-strong}` and `{colors.hairline-tertiary}`. Light gray text (`{colors.ink}` #f7f8f8) carries the body and headlines.

The single chromatic accent is Linear lavender-blue, `{colors.primary}` (#5e6ad2), used on the brand mark, focus rings, and the primary CTA button. A lighter hover state (`{colors.primary-hover}` #828fff) and a focus-tinted variant (`{colors.primary-focus}` #5e69d1) extend the same hue. Linear avoids saturated greens, oranges and reds on the marketing canvas. The only semantic color is `{colors.semantic-success}` (#27a644), for status pills and the rare success indicator.

Display type runs Linear's custom sans (with `SF Pro Display` fallback) at weight 500–700 with negative letter-spacing scaling from -3.0px at 80px down to 0 at body. The body family is Linear's text cut, and a Linear Mono is reserved for code snippets in product screenshots.

The page rhythm is dense product screenshots. Linear's marketing leads with high-fidelity captures of the product UI (issue list, project view, dashboard) framed in `{colors.surface-1}` panels with `{rounded.xl}` 16px corners. The chrome is minimal so the app screenshots carry the page.

**Key characteristics**

- **Dark-canvas marketing system.** `{colors.canvas}` (#010102) is the deepest dark in this collection.
- **Lavender-blue brand accent** (`{colors.primary}` #5e6ad2), used scarcely on the brand mark, focus, and the primary CTA.
- Four-step surface ladder (canvas → surface-1 → surface-2 → surface-3 → surface-4) carries hierarchy without shadow.
- Display tracking pulls aggressively negative (-3.0px at 80px); body holds at -0.05px.
- Cards use `{rounded.lg}` 12px corners with 1px hairline borders. Never pill, rarely 16px.
- **Product UI screenshots** dominate the page. The marketing chrome is a dark frame for the app.
- No second chromatic color. No atmospheric gradients. No spotlight cards.

## Colors

> Source pages: linear.app (home), /intake, /pricing, /contact/sales, /build.

### Brand and accent

- **Lavender-Blue** ({colors.primary}): The signature Linear accent, used on the primary CTA, brand mark and link emphasis.
- **Lavender Hover** ({colors.primary-hover}): Lighter lavender (#828fff) for the hovered state of the primary CTA.
- **Lavender Focus** ({colors.primary-focus}): Focus-ring tint (#5e69d1) for focused inputs and focused buttons.
- **Brand Secure** ({colors.brand-secure}): Muted lavender-gray (#7a7fad), used in "Linear Security" surfaces.

### Surface

- **Canvas** ({colors.canvas}): Default page background, #010102, near-pure black with a faint blue tint.
- **Surface 1** ({colors.surface-1}): One step above canvas: feature cards, pricing cards, product screenshot panels.
- **Surface 2** ({colors.surface-2}): Two steps above: the featured pricing card and hovered cards.
- **Surface 3** ({colors.surface-3}): Three steps above: line-tertiary backgrounds and sub-nav.
- **Surface 4** ({colors.surface-4}): Four steps above: bg-level-3, the deepest lifted surface.
- **Hairline** ({colors.hairline}): 1px borders on cards and dividers.
- **Hairline Strong** ({colors.hairline-strong}): Stronger 1px borders, for input focus rings.
- **Hairline Tertiary** ({colors.hairline-tertiary}): Tertiary borders for nested surfaces.
- **Inverse Canvas** ({colors.inverse-canvas}): Pure white, the surface of the inverse pill CTA on a small set of section openers.
- **Inverse Surface 1** ({colors.inverse-surface-1}): One step above inverse canvas.
- **Inverse Surface 2** ({colors.inverse-surface-2}): Two steps above inverse canvas.

### Text

- **Ink** ({colors.ink}): All headlines and emphasized body type, light gray #f7f8f8.
- **Ink Muted** ({colors.ink-muted}): Secondary type at #d0d6e0, for meta info on hero panels.
- **Ink Subtle** ({colors.ink-subtle}): Tertiary type at #8a8f98: deselected pricing tabs and footer columns.
- **Ink Tertiary** ({colors.ink-tertiary}): Quaternary at #62666d: disabled states and footnotes.

### Semantic

- **Success Green** ({colors.semantic-success}): Status pills, success indicators. The only semantic color on marketing.
- **Overlay** ({colors.semantic-overlay}): Pure black overlay scrim for modals.

## Typography

### Font family

- **Linear Display.** Linear's custom display sans, falling back to `SF Pro Display, -apple-system, system-ui, Segoe UI, Roboto`. Carries display-xl through subhead.
- **Linear Text.** Linear's custom text sans, a different cut tuned for body sizes, on the same fallback stack. Carries body sizes, button labels and captions.
- **Linear Mono.** Linear's custom mono, falling back to `ui-monospace, SF Mono, Menlo`. Used for code snippets in product screenshots and for status and ID tokens.

The marketing surface treats Display and Text as one continuous voice; the family change is silent.

### Hierarchy

| Token                     | Size | Weight | Line Height | Letter Spacing | Use                                         |
| ------------------------- | ---- | ------ | ----------- | -------------- | ------------------------------------------- |
| `{typography.display-xl}` | 80px | 600    | 1.05        | -3.0px         | Largest hero headline                       |
| `{typography.display-lg}` | 56px | 600    | 1.10        | -1.8px         | Section opener headlines                    |
| `{typography.display-md}` | 40px | 600    | 1.15        | -1.0px         | Sub-section headlines                       |
| `{typography.headline}`   | 28px | 600    | 1.20        | -0.6px         | Pricing tier titles, CTA banner heading     |
| `{typography.card-title}` | 22px | 500    | 1.25        | -0.4px         | Feature card title                          |
| `{typography.subhead}`    | 20px | 400    | 1.40        | -0.2px         | Lead body, intro paragraphs                 |
| `{typography.body-lg}`    | 18px | 400    | 1.50        | -0.1px         | Hero subhead, lead paragraphs               |
| `{typography.body}`       | 16px | 400    | 1.50        | -0.05px        | Default body                                |
| `{typography.body-sm}`    | 14px | 400    | 1.50        | 0              | Card body, footer columns                   |
| `{typography.caption}`    | 12px | 400    | 1.40        | 0              | Captions, meta, status                      |
| `{typography.button}`     | 14px | 500    | 1.20        | 0              | All button labels                           |
| `{typography.eyebrow}`    | 13px | 500    | 1.30        | 0.4px          | Section eyebrow (slight positive tracking)  |
| `{typography.mono}`       | 13px | 400    | 1.50        | 0              | Linear Mono for code in product screenshots |

### Principles

- **Aggressive negative tracking on display** (-3.0px at 80px ≈ 4% of size).
- **Single voice from display to body.** Display-xl at 600 down to body at 400, the same family at narrower weights.
- **Eyebrow uses positive tracking** (+0.4px). The contrast against negative-tracked display marks the eyebrow as taxonomy.
- **Mono only in code contexts.** Linear Mono lives inside product screenshots, not on marketing chrome.

### Note on font substitutes

Linear's custom typeface isn't publicly distributed; the documented fallback `SF Pro Display, -apple-system, system-ui` is the recommended substitute on macOS. For cross-platform implementation, **Inter** at weight 500 / 600 / 700 is the closest free substitute. **Geist Sans** is also viable. For mono, **JetBrains Mono** or **Geist Mono** at weight 400 closely approximates Linear Mono.

## Layout

### Spacing system

- **Base unit**: 4px.
- **Tokens (front matter)**: `{spacing.xxs}` 4px · `{spacing.xs}` 8px · `{spacing.sm}` 12px · `{spacing.md}` 16px · `{spacing.lg}` 24px · `{spacing.xl}` 32px · `{spacing.xxl}` 48px · `{spacing.section}` 96px.
- Card interior padding: `{spacing.lg}` 24px on feature/pricing cards; `{spacing.xl}` 32px on testimonial cards; `{spacing.xxl}` 48px on CTA banners.
- Pill button padding: 8px vertical, 14px horizontal, which is Linear's compact button spec.
- Form input padding: 8px vertical · 12px horizontal.

### Grid and container

- Max content width sits around 1280px.
- Card grids are 3-up at desktop, 2-up at tablet, 1-up at mobile.
- Pricing tier grid is 3-up; comparison strip below shows checkmarks per tier.
- Product screenshot panels span the full content width. They are the main element on the page.

### Whitespace philosophy

The dark canvas IS the whitespace. Sections separate by lift onto surface-1 panels, not by gaps in white. Within a panel, generous `{spacing.lg}` 24px gaps between content blocks; `{spacing.section}` 96px between sections.

## Elevation and depth

| Level              | Treatment                                                          | Use                                      |
| ------------------ | ------------------------------------------------------------------ | ---------------------------------------- |
| 0 (flat)           | No shadow, no border                                               | Default for body type, hero text, footer |
| 1 (charcoal lift)  | `{colors.surface-1}` background on canvas, 1px `{colors.hairline}` | Default cards, product panels            |
| 2 (surface-2 lift) | `{colors.surface-2}` background, 1px `{colors.hairline-strong}`    | Featured pricing card, hovered cards     |
| 3 (surface-3 lift) | `{colors.surface-3}` background                                    | Sub-nav, dropdown menus                  |
| 4 (focus ring)     | 2px `{colors.primary-focus}` outline at 50% opacity                | Focused input, focused button            |

Linear's depth is carried by the surface ladder and hairline borders. The brand almost entirely resists drop shadows on dark.

### Decorative depth

- **Product UI screenshots** dominate as decorative depth.
- **No atmospheric gradients, no spotlight cards.**
- **Subtle white edge highlight** on the top edge of lifted panels, which gives the dark surface a faint "pixel rendered" feel.

## Shapes

### Border radius scale

| Token            | Value  | Use                                             |
| ---------------- | ------ | ----------------------------------------------- |
| `{rounded.xs}`   | 4px    | Small chips, status badges                      |
| `{rounded.sm}`   | 6px    | Inline tags                                     |
| `{rounded.md}`   | 8px    | All buttons, form inputs                        |
| `{rounded.lg}`   | 12px   | Pricing cards, feature cards, testimonial cards |
| `{rounded.xl}`   | 16px   | Product screenshot panels                       |
| `{rounded.xxl}`  | 24px   | Oversized CTA banners (rare)                    |
| `{rounded.pill}` | 9999px | Pricing tab toggles, status pills               |
| `{rounded.full}` | 9999px | Avatar circles                                  |

### Photography and illustration geometry

- Product UI screenshots dominate; they sit in `{rounded.xl}` 16px tiles with `{spacing.lg}` 24px outer padding.
- Customer logo tiles render at small sizes (~24px logo height) on `{colors.canvas}` with no border.
- Avatar circles in testimonial cards use `{rounded.full}` at 32–40px sizes.

## Components

### Buttons

**`button-primary`.** The default primary CTA across all pages, in lavender.

- Background `{colors.primary}`, text `{colors.on-primary}`, type `{typography.button}`, padding 8px 14px, rounded `{rounded.md}`.
- Pressed state lives in `button-primary-pressed` (background shifts to `{colors.primary-focus}`).
- Hover state lives in `button-primary-hover` (background shifts to `{colors.primary-hover}` lighter lavender).

**`button-secondary`.** A charcoal button, used for secondary CTAs such as "Sign in" and "Read changelog".

- Background `{colors.surface-1}`, text `{colors.ink}`, type `{typography.button}`, padding 8px 14px, rounded `{rounded.md}`. 1px `{colors.hairline}` border.

**`button-tertiary`.** A plain text button.

- Background `{colors.canvas}`, text `{colors.ink}`, type `{typography.button}`, rounded `{rounded.md}`, padding 8px 14px.

**`button-inverse`.** A white-on-dark inverse CTA.

- Background `{colors.inverse-canvas}`, text `{colors.inverse-ink}`, type `{typography.button}`, rounded `{rounded.md}`, padding 8px 14px.

### Pricing tabs

**`pricing-tab-default`** and **`pricing-tab-selected`.** The pill toggle on `/pricing`.

- Default: `{colors.canvas}` background, `{colors.ink-subtle}` text, rounded `{rounded.pill}`, padding 6px 14px.
- Selected: `{colors.surface-2}` background, `{colors.ink}` text. Selection is shown as a surface lift.

### Cards and containers

**`pricing-card`.** Each tier on `/pricing`.

- Background `{colors.surface-1}`, text `{colors.ink}`, type `{typography.body}`, rounded `{rounded.lg}`, padding 24px. 1px `{colors.hairline}` border.

**`pricing-card-featured`.** The recommended tier, lifted to surface-2.

- Background `{colors.surface-2}`, otherwise identical structure.

**`feature-card`.** A generic feature highlight tile.

- Background `{colors.surface-1}`, text `{colors.ink}`, type `{typography.body}`, rounded `{rounded.lg}`, padding 24px.

**`product-screenshot-card`.** The dominant card type. It frames a high-fidelity Linear app UI screenshot.

- Background `{colors.surface-1}`, text `{colors.ink}`, type `{typography.body}`, rounded `{rounded.xl}`, padding 24px.

**`testimonial-card`.** A customer quote with avatar, name and role.

- Background `{colors.surface-1}`, text `{colors.ink}`, type `{typography.body-lg}`, rounded `{rounded.lg}`, padding 32px.

**`customer-logo-tile`.** A small tile in the customer marquee.

- Background `{colors.canvas}`, text `{colors.ink-subtle}`, type `{typography.caption}`, rounded `{rounded.xs}`, padding 16px.

**`cta-banner`.** The closing CTA panel near the page bottom.

- Background `{colors.surface-1}`, text `{colors.ink}`, type `{typography.headline}`, rounded `{rounded.lg}`, padding 48px.

### Inputs and forms

**`text-input`** and **`text-input-focused`.** Form fields on `/contact/sales` and signup overlays.

- Background `{colors.surface-1}`, text `{colors.ink}`, type `{typography.body}`, rounded `{rounded.md}`, padding 8px 12px.
- Focused state retains the same surface; the focus ring is a 2px `{colors.primary-focus}` outline at 50% opacity.

### Status and build page

**`changelog-row`.** Each row in `/build`, the changelog page, listing version, date and changes.

- Background `{colors.canvas}`, text `{colors.ink}`, type `{typography.body}`, rounded `{rounded.xs}`, padding 24px 0. 1px `{colors.hairline}` bottom rule.

**`status-badge`.** A small status pill.

- Background `{colors.surface-2}`, text `{colors.ink-muted}`, type `{typography.caption}`, rounded `{rounded.pill}`, padding 2px 8px.

### Navigation

**`top-nav`.** A sticky dark bar with the Linear wordmark on the left, primary nav links centered, and a `button-secondary` ("Sign in") and `button-primary` ("Get started") pair on the right.

- Background `{colors.canvas}`, text `{colors.ink}`, type `{typography.body-sm}`, height 56px.

### Footer

**`footer`.** A dense link grid on `{colors.canvas}` with the Linear wordmark on the left.

- Background `{colors.canvas}`, text `{colors.ink-subtle}`, type `{typography.caption}`, padding 64px 32px.

## Do's and don'ts

### Do

- Reserve `{colors.canvas}` (#010102) as the system's anchor surface. The faint blue tint is deliberate.
- Use `{colors.primary}` lavender ONLY for: brand mark, primary CTA, focus ring, link emphasis.
- Use the four-step surface ladder for hierarchy. Avoid skipping levels.
- Pair display weight 600 with body weight 400. Linear resists 700+ display weights.
- Apply negative letter-spacing aggressively on display.
- Make a product UI screenshot the focus of every section.
- Compose CTAs as `{rounded.md}` 8px corners.

### Don't

- Don't ship a light-mode marketing page.
- Don't use lavender as a section background or card fill.
- Don't introduce a second chromatic accent (orange, pink, green for marketing).
- Don't add atmospheric gradients or spotlight cards.
- Don't pill-round CTAs.
- Don't use `#000000` true black as the canvas.
- Don't combine multiple bright accents in product screenshot mockups.

## Responsive behavior

### Breakpoints

| Name       | Width  | Key Changes                                         |
| ---------- | ------ | --------------------------------------------------- |
| Desktop-XL | 1440px | Default desktop layout                              |
| Desktop    | 1280px | Card grid 3-up maintained                           |
| Tablet     | 1024px | Card grid 3-up → 2-up                               |
| Mobile-Lg  | 768px  | Pricing comparison becomes accordion; nav hamburger |
| Mobile     | 480px  | Single-column; display-xl scales 80px → ~36px       |

### Touch targets

- CTAs hold ≥40px tap height across viewports.
- Pricing tab pills hold ≥36px tap height; touch viewports grow to ≥44px.
- Form inputs hold ≥44px tap target on touch.

### Collapsing strategy

- **Top nav**: links collapse to hamburger below 768px.
- **Card grids**: 3-up → 2-up at 1024px → 1-up below 768px.
- **Pricing comparison**: per-tier accordion below 768px.
- **Display type**: `{typography.display-xl}` 80px scales toward `{typography.display-md}` 40px on mobile.

### Image behavior

- Product UI screenshots maintain aspect ratio and never crop.
- Customer logos in the marquee may collapse from 6-up to 3-up below 768px.

## Iteration guide

1. Focus on ONE component at a time and reference it by its `components:` token name.
2. When introducing a section, decide first which surface lift it lives on.
3. Default body to `{typography.body}` at weight 400.
4. Run `npx @google/design.md lint DESIGN.md` after edits.
5. Add new variants as separate component entries.
6. Treat lavender as scarce: brand mark, primary CTA, focus, link emphasis.
7. Lead every section with a product UI screenshot.

## Known gaps

- The four-step surface ladder values are extracted directly from Linear's `--color-bg-level-3`, `--color-line-tint`, etc. CSS variables; they are Linear's canonical surface spec.
- Form-field error and validation styling is not visible on the inspected pages.
- Light mode is not documented because the marketing site does not ship a light theme.
- Linear's actual product UI uses a richer color-tag palette (red, orange, yellow, green, blue, purple) for issue priorities and project labels. Those colors live in the in-product surfaces shown in mockups.
- The custom display, text, and mono families are proprietary; an open-source substitute is acceptable.
