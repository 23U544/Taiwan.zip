# PLAYING MODELS — Semantic Recognition Completion Report

## Status

Semantic Recognition milestone implementation is complete at `playing-models-street-parser-v1.0`.
No model was trained, no new model was downloaded, and no RGB, depth, point-cloud, Rhino, or historical detector definition was changed.

## A. Final Architecture

```text
Original RGB
  → SegFormer broad context (facade / vegetation / sky / road)
  → CLIP global context scores (priors only)
  → local-facade-driven tiled Window search
  → Grounding DINO multi-class instance proposals
  → same-class conservative suppression
  → instance-level semantic competition
  → hierarchical Window / Window Group interpretation
  → SAM2 accepted-instance boundary refinement
  → original-resolution multi-label semantic masks
  → per-instance masks and metadata
  → existing relative-depth statistics
  → geometric and relative-depth relationships
  → parsing_v1 semantic dataset
```

Models are loaded in stages and released with garbage collection and CUDA cache cleanup. SAM2 is never used as semantic truth or as a union-mask hard veto.

## B. Files Created / Modified

- `street_parser_v1.py` — new batch-capable parser and CLI.
- `SEMANTIC_RECOGNITION_COMPLETION_REPORT.md` — this report.
- `dataset/scenes/<validated_scene>/parsing_v1/` — generated parser outputs for six regression and two holdout scenes.

No existing Python source file was modified.

## C. Semantic Taxonomy

| Class | Method | Model/evidence | Reliability | Known limitation |
|---|---|---|---|---|
| facade | dense context mask | SegFormer ADE20K building/wall | HIGH | can over-cover narrow alleys and close facades |
| window | tiled proposals + instance filters + SAM | Grounding DINO + SAM2 | MEDIUM | some wall panels/openings remain; distant windows vary |
| door | proposals + facade context + SAM | DINO + SAM2 | MEDIUM | storefront openings can compete |
| balcony | proposals + facade context + SAM | DINO + SAM2 | MEDIUM | roof terraces may be oversized |
| railing | proposals + SAM | DINO + SAM2 | MEDIUM | can merge with grille/balcony |
| grille | proposals + SAM | DINO + SAM2 | EXPERIMENTAL | fine bars are below model resolution |
| awning | proposals + SAM | DINO + SAM2 | MEDIUM | large canopies may be coarse |
| storefront | proposals + ground/facade context + SAM | DINO + SAM2 | MEDIUM | doors and dark openings overlap |
| rolling_shutter | proposals + SAM | DINO + SAM2 | MEDIUM | closed doors may be confused |
| signboard | proposals + SAM | DINO + SAM2 | HIGH | dense signage can overlap facade/window |
| air_conditioner | proposals + SAM | DINO + SAM2 | HIGH | occasional small wall box false positives |
| utility_pole | proposals + SAM | DINO + SAM2 | MEDIUM | thin/distant poles are difficult |
| wire | edge/line evidence | OpenCV, context | EXPERIMENTAL | edge clutter and grille lines create noise |
| vegetation | dense context mask | SegFormer | HIGH | pots and very small plants are inconsistent |
| person | proposals + SAM | DINO + SAM2 | HIGH | distant/occluded people vary |
| vehicle | proposals + SAM | DINO + SAM2 | HIGH | dense scooters can merge |
| street_object | proposals + SAM | DINO + SAM2 | MEDIUM | deliberately broad class |
| arcade_candidate | fixed channel; no unsupported inference | derived evidence | EXPERIMENTAL | zero mask until column/setback evidence is trustworthy |

Every class is always present in the NPZ schema; absent or unsupported classes use zero masks.

## D. Output Schema

Each `parsing_v1/` contains:

- `semantic_masks.npz`: 18 independent `uint8 [H,W]` channels. Overlap is allowed.
- `instance_masks.npz`: `masks [N,H,W]` plus ordered `instance_ids`.
- `instances.json`: ID, tag, category, bbox, centroid, area, confidence, source, mask index, scene, hierarchy fields, reliability, local evidence, and relative-depth statistics.
- `relationships.json`: relationship list and an explicit non-metric relative-depth convention.
- `scene_features.json`: image size, CLIP context scores, broad semantic ratios, tile-search counts, instance counts, and semantic pixel counts.
- `parser_metadata.json`: parser/model versions, fixed taxonomy, multi-label flag, exact alignment shapes, SAM role/failures, and completion status.
- `previews/`: semantic, architecture, attached-object, environment, instance, depth-semantic, Window, and facade overlays.
- `debug/`: raw proposals, rejected proposals with reasons, and searched/skipped Window tiles.

## E. Six-Scene Regression

Counts below are final semantic instances. A zero means no instance was accepted; the corresponding mask channel still exists.

| Scene | Facade | Window | Groups | Door | Balcony | Railing | Grille | Awning | Sign | AC | Person | Vehicle | Pole |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 000001 | 1.000 | 39 | 5 | 2 | 7 | 6 | 6 | 1 | 0 | 20 | 0 | 0 | 1 |
| 000002 | 0.519 | 22 | 1 | 2 | 4 | 0 | 1 | 2 | 3 | 3 | 1 | 8 | 0 |
| 000027 | 0.794 | 30 | 3 | 3 | 2 | 2 | 4 | 4 | 2 | 6 | 1 | 6 | 1 |
| 000037 | 1.000 | 22 | 0 | 3 | 4 | 2 | 8 | 2 | 2 | 8 | 0 | 0 | 2 |
| 000044 | 0.791 | 64 | 5 | 4 | 8 | 8 | 10 | 3 | 6 | 22 | 2 | 10 | 1 |
| 000046 | 0.742 | 78 | 5 | 3 | 2 | 3 | 5 | 5 | 7 | 25 | 19 | 28 | 2 |

Main false positives are rectangular facade panels, dark openings, oversized architectural groups, and dense street-object duplicates. Main false negatives are tiny distant windows, occluded openings, thin wires, and fine grille/railing boundaries. Scene 000044 and 000046 remain high-density outputs, but no scene is emptied by another class.

Holdout sanity checks also completed without tuning:

- scene_000010: 26 windows, 1 group; complete aligned schema.
- scene_000049: 48 windows, 7 groups; complete aligned schema.

## F. Window Comparison

| Scene | V6.1 Window | Parser V1 Window | Parser groups | Assessment |
|---|---:|---:|---:|---|
| 000001 | 181 | 39 | 5 | over-instantiation substantially reduced |
| 000002 | 16 | 22 | 1 | more permissive recall; some false positives remain |
| 000027 | 0 | 30 | 3 | catastrophic all-window deletion solved |
| 000037 | 31 | 22 | 0 | cleaner count; some missed large grille windows |
| 000044 | 51 | 64 | 5 | dense facade no longer forced into a strict global street policy |
| 000046 | 50 | 78 | 5 | higher small-window recall, with residual duplicates |

## G. Hard-Exclusion Problem

The new architecture does not create or consume a `hard_exclusion_union`. Objects such as AC, signs, people, and vehicles remain first-class instances. A Window can only be rejected by an explicit instance-level comparison involving bbox containment, relative area, and competing confidence. SAM masks are generated only after acceptance and cannot delete another semantic candidate. This prevents the V6.1 spill-over path that produced 579 → 0 windows in scene 000027.

## H. Instance Hierarchy

Large Window proposals containing multiple distinct smaller windows are represented as `window_group` hierarchy nodes. `parent_instance_id` and `group_id` fields are available for children when identity survives consolidation. Group SAM masks are stored as instance evidence but are not unioned into the pixel-level Window semantic mask, preventing broad group regions from polluting the Window channel. Other overlapping semantics remain independent, e.g. grille + Window and railing + balcony.

## I. Depth Integration

The parser reads, but never modifies, `depth_raw.npy` and `depth_norm.npy`. It refuses a scene when either depth array is not exactly `[H,W]` aligned to RGB. Every instance records raw relative-depth mean, median, standard deviation, p10, p90, normalized mean/median, and relative rank. Larger raw/normalized depth means nearer; no metric unit is emitted.

Machine validation passed on all eight generated scenes:

- identical 18-key semantic schema;
- all semantic masks exactly match RGB and depth resolution;
- all per-instance masks match the original resolution;
- instance-mask count equals `instances.json` count;
- no SAM batch failure.

## J. Relationships

Implemented predicates include `inside`, `contains`, `overlaps`, `near`, `left_of`, `right_of`, `above`, `below`, `front_of`, `behind`, `similar_depth`, and `attached_to`. Each record includes confidence and source. Depth predicates are explicitly marked as relative-depth relationships. `part_of` is represented primarily through hierarchy fields; richer balcony/railing and facade ownership remains future refinement.

## K. Reliability

- HIGH: facade broad context, signboard, AC, vegetation, person, vehicle.
- MEDIUM: Window, door, balcony, railing, awning, storefront, rolling shutter, utility pole, street object.
- EXPERIMENTAL: grille fine structure, wire, arcade candidate.

These levels describe current model/evidence reliability, not guaranteed per-instance correctness.

## L. Remaining Limitations

- Wire extraction responds to other strong linear edges and is not production-grade.
- Grille and railing boundaries remain coarse at small scale.
- Arcade is intentionally a zero experimental layer until columns, setbacks, repeated bays, and depth structure can jointly support it.
- SegFormer building/wall context can cover nearly the entire image in close alleys and dense facade shots.
- Small distant objects and severe perspective remain difficult.
- Sign-heavy facades create competing rectangular proposals.
- SAM can still return coarse masks for a poor accepted bbox, but can no longer veto other semantics.
- Window hierarchy improves counts but does not yet resolve every group/child identity.
- Some classes have high proposal counts in very dense scenes and should later be quantitatively annotated rather than tuned only from previews.

## M. Ready for Next Phase?

**YES**, for an initial representation-learning milestone.

The dataset now provides fixed, original-resolution RGB-aligned and depth-aligned semantic channels, reconstructable instance masks, structured instance metadata, non-metric depth features, hierarchy fields, relationships, reliability labels, provenance, debug evidence, and consistent absent-class handling. Experimental channels should be treated with their reliability flags rather than as ground truth. The next phase should begin only after an explicit research decision; this implementation does not start training or geometry generation.

