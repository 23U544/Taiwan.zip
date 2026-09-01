# PLAYING MODELS — Project Context

## Taiwan Streetscape → RGBD / Semantic Spatial Dataset → Generative Spatial Model → Rhino / Grasshopper

> This file is the persistent technical and research context for the **Playing Models** project.
> Codex should read this file before modifying the local repository.
>
> The goal is **not** to optimize one detector for one image. The goal is to build a generalizable spatial parsing and generative workflow for Taiwanese streetscapes.

---

# 1. Project Role

This repository is part of a research workflow investigating how Taiwanese streetscape images can be transformed into a learnable spatial representation.

The long-term pipeline is:

```text
Taiwan Streetscape Image
        ↓
RGB + Relative Depth
        ↓
Semantic / Instance Parsing
        ↓
Spatial Relationships
        ↓
Learnable Spatial Representation
        ↓
Latent / Generative Spatial Model
        ↓
Generated Spatial Configuration
        ↓
Rhino / Grasshopper
        ↓
2.5D / 3D Geometry
        ↓
Playing Models
```

The current computer-vision work is therefore **infrastructure for a larger architectural research project**.

Do not treat Window Detection as the final goal.

---

# 2. Research Framework

The project currently has two broad research directions.

## 2.1 Model for Thinking

The research is not simply asking:

> “How does AI think like a human?”

Instead, it asks how AI can describe spatial environments through:

- latent representations,
- feature relationships,
- repetition,
- density,
- alignment,
- spatial statistics,
- semantic layers,
- relative depth,
- foreground/background relationships,
- architectural and non-architectural elements.

Potential Taiwanese streetscape information includes:

### Architectural
- Facade
- Window
- Door
- Balcony
- Railing
- Metal grille / 鐵窗
- Awning / 雨遮
- Arcade / 騎樓
- Columns
- Structural / enclosure components

### Attached / Non-architectural
- Signboard
- Air conditioner
- Utility pole
- Wire
- Vehicle
- Person
- Furniture
- Street object
- Vegetation
- Plant pots
- Temporary additions

### Future behavioral information
- pedestrian flow
- gathering
- movement
- occupation
- events

“Urban texture” is therefore broader than buildings alone.

---

## 2.2 Model for Making

The long-term aim is:

```text
Learned Spatial Representation
        ↓
Generate New Semantic / Spatial Fields
        ↓
Convert to 2.5D / 3D Geometry
        ↓
Rhino / Grasshopper
        ↓
Manipulation / Combination / Fabrication
```

The current dataset is being created to enable this later stage.

---

# 3. Local Project Environment

Main project root:

```text
D:\GIA\3rd Up\DepthAnything\Depth-Anything-V2
```

Primary environment:

```text
Windows
Miniconda / Python
NVIDIA GeForce RTX 4050 Laptop GPU
VRAM ≈ 6 GB
```

Current PyTorch environment:

```text
torch 2.13.0+cu130
CUDA available = True
```

Depth Anything V2 checkpoint:

```text
checkpoints\depth_anything_v2_vitb.pth
```

Encoder:

```text
vitb
```

`xFormers not available` is only an acceleration warning and is not considered an error.

---

# 4. GPU Memory Constraint

The RTX 4050 Laptop GPU has only about 6 GB VRAM.

Do **not** keep all major models loaded simultaneously.

Avoid:

```text
SegFormer
+ Grounding DINO
+ SAM2
+ CLIP
+ Depth Anything
```

all resident in GPU memory at once.

Preferred pattern:

```python
load_model()
inference()

del model

import gc
gc.collect()

import torch
torch.cuda.empty_cache()
```

Then load the next model.

This behavior should be preserved during future refactoring.

---

# 5. Core Dataset Representation Decision

The primary training representation should **not** be `.ply`.

The core representation should preserve pixel alignment:

```text
RGB
+
Depth
+
Semantic Masks
+
Instances
+
Spatial Relationships
```

Point clouds are derived data.

The preferred logic is:

```text
RGB + Depth + Semantics
        ↓
Training / Feature Learning
        ↓
Optional Back-projection
        ↓
PLY / Rhino Geometry
```

rather than:

```text
RGB
↓
PLY
↓
Everything happens in point-cloud space
```

This preserves:

- pixel correspondence,
- semantic segmentation compatibility,
- RGB/depth alignment,
- CNN / ViT compatibility,
- flexible downstream geometry generation.

---

# 6. Dataset Structure

Current `build_dataset.py` generates approximately:

```text
dataset/
├── dataset.json
├── manifest.csv
└── scenes/
    ├── scene_000001/
    │   ├── rgb.jpg
    │   ├── depth_raw.npy
    │   ├── depth_norm.npy
    │   ├── depth_preview.png
    │   └── metadata.json
    │
    ├── scene_000002/
    │   └── ...
    │
    └── ...
```

Core scene files should be treated as primary source data.

Do not delete:

```text
rgb.jpg
depth_raw.npy
depth_norm.npy
depth_preview.png
metadata.json
```

Generated detector folders may be recreated.

---

# 7. Depth Pipeline

Depth model:

```text
Depth Anything V2
Encoder: ViT-B
```

Current depth is **relative depth**, not metric depth.

Important convention:

```text
larger raw value = nearer
smaller raw value = farther
```

Normalized representation:

```text
0 = farther
1 = nearer
```

Do not interpret these values as meters.

---

# 8. Depth File Policy

Keep:

```text
depth_raw.npy
```

as `float32`.

Do not replace it with only an 8-bit PNG.

`depth_preview.png` is for visualization only.

Also keep:

```text
depth_norm.npy
```

for normalized downstream features.

---

# 9. Relative Depth → Pseudo Distance

Current geometry generation uses a pseudo-distance mapping.

It is **not metric reconstruction**.

Example conceptual range:

```text
near = 1
far = 8
```

Inverse-depth style mapping:

```text
inverse_far  = 1 / far
inverse_near = 1 / near

inverse_distance =
    inverse_far
    +
    relative_gamma * (inverse_near - inverse_far)

distance = 1 / inverse_distance
```

Purpose:

- preserve foreground/background hierarchy,
- create usable 2.5D geometry,
- support Rhino / Grasshopper experimentation.

Do not claim metric accuracy.

---

# 10. Image → 2.5D Point Cloud

Current script:

```text
img2pointclouds.py
```

If camera calibration is unavailable, synthetic camera intrinsics may use approximately:

```text
Horizontal FOV ≈ 60°
```

Then:

```text
fx = 0.5 * width / tan(FOV / 2)
fy = fx
cx = width / 2
cy = height / 2
```

Back projection:

```text
Xcam = (u - cx) * D / fx
Ycam = -(v - cy) * D / fy
Zcam = D
```

---

# 11. Rhino Coordinate Convention

This was explicitly tested and confirmed.

Use:

```text
X_rhino = X_cam
Y_rhino = Z_cam
Z_rhino = Y_cam
```

Interpretation:

```text
+X = image right
+Y = forward / scene depth
+Z = up
```

Camera origin:

```text
(0, 0, 0)
```

Near objects:

```text
smaller positive Y
```

Far objects:

```text
larger positive Y
```

Do **not** invert this convention without explicit re-validation.

---

# 12. Semantic Representation Strategy

Prefer **multi-label semantic masks**, not one mutually exclusive semantic image.

Real Taiwanese streetscape objects overlap.

Examples:

```text
Window + Metal Grille
Window + Balcony
Facade + Signboard
Facade + AC
Balcony + Railing
```

Therefore semantic layers should remain independent.

Potential layers:

```text
window
door
balcony
railing
grille
awning
facade
signboard
ac
vehicle
person
vegetation
wire
utility_pole
...
```

---

# 13. Architectural vs Non-Architectural Distinction

This distinction is important for detector logic.

## Architectural Elements

Examples:

```text
Window
Door
Balcony
Railing
Metal grille
Awning
Facade
Column
Arcade
Structural components
Enclosure components
```

## Non-Architectural / Attached / Street Elements

Examples:

```text
AC
Signboard
Vehicle
Person
Furniture
Street object
Temporary object
Plant pot
Other attached equipment
```

Important:

Being “non-architectural” does **not** mean being discarded from the final Playing Models dataset.

For example:

```text
AC = negative evidence for Window
```

but:

```text
AC = useful semantic information for Playing Models
```

Never confuse:

```text
not a window
```

with:

```text
not important
```

---

# 14. Some Elements Should Be Derived

Not every tag should use Grounding DINO.

Examples:

## Arcade / 騎樓

May be better derived from:

```text
Facade
+ Columns
+ Ground level
+ Setback
+ Depth
+ Repetition
```

## Wire

Thin structures may require specialized line / thin-object methods.

## Railing / Grille

High-frequency repeated lines may require:

- fine segmentation,
- geometry analysis,
- repetition analysis,
- edge / line features.

Do not force every tag into the same detector architecture.

---

# 15. Model-by-Tag Strategy

Current likely direction:

```text
Facade
→ SegFormer / dense segmentation

Window
→ Multi-scale Grounding DINO
→ Scene-aware context
→ SAM2
→ Instance consolidation
→ Repetition analysis

Signboard
→ Grounding DINO + SAM2

AC
→ Grounding DINO + SAM2

Vehicle
→ Object detection / segmentation

Person
→ Object detection / segmentation

Balcony
→ Grounding DINO + SAM2 + context

Railing
→ detection + fine segmentation / geometry

Grille
→ fine structure / repeated-line analysis

Wire
→ thin-structure / line-specialized method

Arcade
→ relational / spatial inference
```

This is intentionally heterogeneous.

---

# 16. Early Semantic Prototype

Initial prototype:

```text
semantic_test.py
```

Used:

```text
Grounding DINO
+
SAM2
```

for classes including:

```text
building
signboard
awning
window
balcony
utility pole
wire
person
vehicle
vegetation
```

It demonstrated that DINO + SAM2 can create useful masks.

However, Window recall was not sufficient.

Therefore Window became the first specialized module.

---

# 17. SegFormer Experiment

SegFormer was tested as a possible Window segmentation model.

Window probability was extremely weak in test scenes.

Example observed distribution:

```text
min  ≈ 1.16e-09
max  ≈ 0.00616
mean ≈ 5.89e-05
```

Conclusion:

SegFormer is **not** currently the primary Window detector.

SegFormer remains useful for broad semantic context:

```text
building
wall
sky
road
vegetation
...
```

It is therefore used as a facade/context gate.

---

# 18. Window Detector History

The history matters because it explains current decisions.

## V3 — Multi-scale Grounding DINO

Introduced tiled Window detection.

Typical scales included:

```text
320
384
512
768
Global
```

Reason:

Small distant windows become larger relative to smaller crops.

Result:

- recall improved significantly,
- duplicate boxes increased.

## V4 — Aggressive Union-Find Consolidation

Attempted to merge duplicates.

Problem:

Transitive merging:

```text
A matches B
B matches C
→ A + B + C merge
```

This caused neighboring real windows to collapse into incorrect groups.

V4 should not be used as the preferred merge strategy.

## V4.1 — Conservative Anchor Clustering

Important improvements:

- no transitive Union-Find chain merging,
- anchor-based clustering,
- median box fusion,
- late group suppression,
- weaker C1 detections retained during validation.

Scene001 became close to usable.

V4.1 remains a useful historical baseline.

## V5 — Context / Exclusion

Added:

```text
SegFormer facade gate
+
Signboard Grounding DINO/SAM2 exclusion
```

Motivation:

Street-perspective Scene002 produced false windows on:

```text
mountain
sky
road
cars
signs
storefronts
```

Facade gating alone was insufficient.

## V5.1 — Semantic Negatives + CLIP

Added:

```text
semantic negative mask
+
non-building object exclusions
+
CLIP crop verifier
```

CLIP compared Window evidence against negatives such as:

```text
tile
wall
door
sign
vehicle
furniture
...
```

There was a Transformers compatibility bug where:

```python
get_text_features()
get_image_features()
```

returned `BaseModelOutputWithPooling` rather than directly returning a Tensor.

The fixed implementation extracts embeddings through fields such as:

```text
.pooler_output
.text_embeds
.image_embeds
```

before normalization.

---

# 19. Critical Scene Difference

The most important conceptual insight so far is the difference between two scene types.

## scene_000001

Frontal dense facade.

Characteristics:

```text
most of image = facade
many windows
small perspective distortion
high window density
```

This scene benefits from:

```text
high-recall search
many tiles
global pass
lower threshold
```

## scene_000002

Street-perspective scene.

Contains:

```text
road
mountain
sky
vehicles
signboards
storefronts
perspective buildings
relatively few visible windows
```

Repeatedly asking every tile:

```text
"find window"
```

creates a search bias.

Even if false-positive probability is low:

```text
many tiles
×
many DINO window queries
=
many false rectangular candidates
```

Typical false positives:

```text
cars
doors
signs
tiles
storefronts
mountain edges
street objects
facade rectangles
```

This insight led to V6.

---

# 20. V6 — Scene-Aware Window Search

Core architecture:

```text
RGB
│
├── SegFormer
│      ↓
│   Facade Mask
│   Semantic Negative Mask
│      ↓
│   Allowed Facade Mask
│
└── CLIP Whole-Image Classification
       ↓
   Dense Facade
   Street Perspective
   Mixed Urban
   Landscape

Allowed Facade
+
Scene Mode
        ↓
Adaptive Window Search
```

---

# 21. Scene Classification

CLIP whole-image scene categories currently include concepts similar to:

```text
dense_facade
street_perspective
mixed_urban
landscape
```

Scene mode selects between:

```text
dense
street
```

based on:

```text
CLIP scene evidence
+
effective facade ratio
```

---

# 22. Dense Mode

Designed for frontal facade images.

Priority:

```text
RECALL
```

Typical behavior:

- lower DINO threshold,
- more eligible tiles,
- global window pass allowed,
- weaker C1 candidates tolerated,
- higher expected window density.

---

# 23. Street Mode

Designed for perspective street scenes.

Priority:

```text
PRECISION
```

Typical behavior:

- higher DINO threshold,
- only search tiles with sufficient facade support,
- skip mountain / sky / road,
- global full-image Window pass disabled by default,
- weak C1 candidates require stronger evidence,
- large single-scale boxes are suspicious.

The detector must not assume equal expected window density across scene types.

---

# 24. Important V6 Diagnostic

Key output:

```text
window_search_tiles_preview.jpg
```

Interpretation:

```text
GREEN = Window detector actually searched this tile
GRAY  = Tile intentionally skipped
```

Desired behavior:

### Scene001
Most facade tiles should be green.

### Scene002
Sky / mountain / road should mostly be skipped.

This output is often more informative than looking only at final boxes.

---

# 25. V6 Major Bug

Scene001 V6 test:

```text
Scene mode: dense
Effective facade ratio: 0.9973

Raw windows: 2143
After context gate: 1
Context rejected: 2142
```

Rejection reason:

```text
Counter({'grounded_object_overlap': 2142})
```

Cause:

The exclusion vocabulary mixed:

### Genuine negative / attached objects

```text
sign
vehicle
person
furniture
...
```

with architectural objects that commonly overlap windows:

```text
door
storefront
balcony railing
metal railing
metal grille
awning
...
```

SAM2 masks for these elements overlapped real windows.

Almost every real Window candidate was therefore rejected.

---

# 26. V6.1 Fix

Current main experimental script:

```text
window_detector_v6_1.py
```

V6.1 separates exclusion logic.

## HARD EXCLUSION

Objects that may genuinely veto Window evidence.

Examples:

```text
sign
vehicle
person
furniture
street object
AC
```

Important:

AC is considered a non-architectural attached element.

Detecting AC on a frontal facade is **not an error**.

The error would be:

```text
AC mask expands too far
→ overlaps neighboring true window
→ deletes real window
```

## SOFT NOT-WINDOW / COMPETING ARCHITECTURAL EVIDENCE

Architectural elements that may overlap windows:

```text
door
rolling shutter
storefront
balcony railing
metal railing
metal grille
security grille
awning
...
```

These should not immediately delete Window candidates.

They provide competing semantic evidence.

---

# 27. Current Scene001 V6.1 Result

Known result:

```text
Scene mode: dense
Effective facade ratio: 0.9973

Hard exclusions: 45
Soft not-window detections: 187

Raw windows: 2143
After context gate: 1955
Context rejected: 188

CLIP rejected: 0
Groups removed: 58

Final windows: 181
SAM masks: 181
```

This confirms the V6 hard-exclusion bug was fixed.

Comparison:

```text
V6:
2143 → 1

V6.1:
2143 → 1955 → 181 final
```

---

# 28. Current Scene001 Visual Interpretation

Scene001 is now broadly usable.

Observed strengths:

- high Window recall,
- many real windows detected,
- context gate no longer destroys the result,
- dense mode is appropriate.

Observed remaining issues:

- duplicate / overlapping boxes,
- multiple scales may represent the same window,
- some window-group boxes remain,
- some rectangular non-window architectural regions may survive,
- final count of 181 may be high.

Current likely bottleneck:

```text
candidate interpretation
+
instance consolidation
```

rather than:

```text
ability to find windows
```

Do not optimize Scene001 alone.

---

# 29. Scene002 Status

Scene002 has already been run using V6.1.

The results should exist locally under approximately:

```text
dataset\scenes\scene_000002\window_v6_1\
```

Codex should inspect these directly.

Important files include:

```text
scene_context.json
features_window_v6_1.json

window_search_tiles_preview.jpg
allowed_facade_overlay.png
facade_gate_overlay.png
semantic_negative_overlay.png
hard_exclusion_overlay.png
soft_not_window_boxes_preview.jpg
raw_windows_and_exclusions_preview.jpg
gated_window_candidates_preview.jpg
clip_verifier_preview.jpg
detections_preview.jpg
window_instances_preview.jpg

rejected_window_candidates.json
clip_rejected_candidates.json
```

---

# 30. Current Immediate Research Question

The immediate question is:

> Can the same Window Detector adapt its search behavior according to scene context?

Specifically:

```text
Dense frontal facade
→ high window density
→ high-recall search

Street perspective
→ low relative window density
→ selective facade search
→ higher precision
```

This scene-aware behavior is currently more important than adding new semantic categories.

---

# 31. Immediate Codex Audit Task

Before modifying any code, Codex should compare:

```text
scene_000001/window_v6_1
scene_000002/window_v6_1
```

Questions to answer:

1. Did scene classification correctly choose:
   - scene001 → dense
   - scene002 → street?

2. Is effective facade ratio reasonable?

3. Does `window_search_tiles_preview.jpg` demonstrate different search behavior?

4. In Scene002, are:
   - sky,
   - mountain,
   - road

   mostly skipped?

5. Are actual facade tiles still searched?

6. Are hard exclusions reasonable?

7. Are AC detections legitimate?

8. Are hard-exclusion masks spilling into nearby windows?

9. Which stage creates the majority of Scene002 false positives?

Potential stages:

```text
tile search
raw Grounding DINO
context gate
hard exclusion
soft evidence
CLIP
instance consolidation
group suppression
SAM2
```

10. Which stage creates false negatives?

11. In Scene001, are most remaining errors:
   - duplicates,
   - window groups,
   - architectural competing elements,
   - segmentation errors?

12. Does V6.1 generalize, or does it mainly work on Scene001?

Do **not** modify code before completing this audit.

---

# 32. No Scene-Specific Hacks

Never write logic such as:

```python
if scene_name == "scene_000001":
    threshold = ...
```

or:

```python
if scene_name == "scene_000002":
    threshold = ...
```

The system should generalize by:

```text
scene type
semantic evidence
spatial evidence
feature distributions
```

not file name.

---

# 33. Possible V6.2 Direction

Do not automatically build V6.2.

First inspect actual V6.1 results.

If needed, V6.2 should likely move away from endless binary veto rules and toward **evidence fusion**.

Potential Window evidence:

```text
Grounding DINO max score
Grounding DINO mean score
cross-scale support
facade overlap
context facade overlap
hard exclusion overlap
soft architectural overlap
CLIP positive score
CLIP negative score
box area
scene mode
repetition / alignment evidence
```

Conceptually:

```text
WindowScore =
    positive detection evidence
  + cross-scale consensus
  + facade evidence
  + spatial repetition evidence
  - non-window evidence
  - competing semantic evidence
```

Do **not** invent arbitrary weights before inspecting feature distributions.

---

# 34. Statistical Development Principle

Before adding thresholds or scoring weights:

inspect the JSON distributions.

Compare obvious:

```text
true windows
false positives
```

Look at distributions of:

```text
DINO confidence
support count
box dimensions
facade overlap
semantic negative overlap
hard exclusion overlap
soft evidence overlap
CLIP scores
depth statistics
```

Then choose a strategy.

Avoid ad-hoc tuning based only on visual intuition.

---

# 35. Future Repetition / Grid Analysis

Dense facades contain repetition:

```text
□ □ □ □
□ □ □ □
□ □ □ □
```

Possible future logic:

```text
W1
W2
W3
?
W5
```

If instances have:

- similar Y,
- similar dimensions,
- regular spacing,
- repeated facade context,

then the missing location may become a new proposal.

However:

Do not implement repetition completion until Window instances are reasonably clean.

Otherwise false positives will reinforce themselves.

Preferred order:

```text
Detection
↓
Instance Consolidation
↓
Clean Window Instances
↓
Row / Column Clustering
↓
Spacing Analysis
↓
Missing-Window Proposal
```

---

# 36. Window Module Freeze Condition

Do not chase 100% performance on Scene001.

The Window module should eventually be validated on a broader set.

Recommended validation categories:

```text
frontal apartment facade
old apartment
townhouse
commercial shophouse
street perspective
metal-sheet additions
modern residential
dense signage street
mixed vegetation/building street
```

Prefer at least:

```text
5–10+ images
```

using the same parameters.

When generalization is acceptable:

freeze Window module.

---

# 37. Future Parser Architecture

Once Window stabilizes, do not keep adding every semantic category into one Window script.

Future architecture should become modular.

Possible structure:

```text
architectural_parser.py
│
├── detect_facade()
├── detect_windows()
├── detect_doors()
├── detect_balconies()
├── detect_railings()
├── detect_grilles()
├── detect_awnings()
└── derive_arcade()
```

and:

```text
non_architectural_parser.py
│
├── detect_signboards()
├── detect_ac()
├── detect_people()
├── detect_vehicles()
├── detect_vegetation()
└── detect_street_objects()
```

Potential top-level orchestration:

```text
street_parser.py
```

---

# 38. Future Scene Representation

Potential future scene structure:

```text
scene_000001/
│
├── rgb.jpg
├── depth_raw.npy
├── depth_norm.npy
├── metadata.json
│
├── semantics/
│   ├── semantic_masks.npz
│   ├── instances.json
│   └── semantic_metadata.json
│
├── spatial/
│   ├── relationships.json
│   ├── repetition.json
│   └── facade_structure.json
│
└── derived/
    └── pointcloud.ply
```

This is not yet frozen.

Do not redesign the entire dataset prematurely.

---

# 39. Future Instance Data

Potential per-instance information:

```text
tag
category
bbox
mask
centroid
pixel_area

DINO score
multi-scale support
semantic confidence

depth_mean
depth_median
depth_std
depth_p10
depth_p90
relative_depth_offset

facade membership
row
column
nearest neighbours
spacing
alignment
repetition group
```

For Playing Models, these spatial relationships may ultimately be more important than object labels alone.

---

# 40. Training Is Not Yet the Main Task

Do not start training the final generative model yet.

Current priority:

```text
Reliable Representation
```

We first need:

```text
RGB
Depth
Semantic Masks
Instances
Spatial Relationships
```

Only then choose an appropriate neural architecture.

Do not currently train:

```text
RGB → Depth
```

from Depth Anything pseudo-labels.

Depth Anything already performs that task.

There is little research value in merely reproducing its pseudo-label behavior.

---

# 41. Future Generative Model

Future inputs may include:

```text
RGB
relative depth
semantic fields
spatial relations
```

Potential learned representation:

```text
latent spatial representation
```

Potential outputs:

```text
depth field
semantic field
spatial relationships
```

Then:

```text
generated fields
↓
point cloud / geometry conversion
↓
Rhino / Grasshopper
↓
architectural spatial model
```

The final model type is **not yet fixed**.

Do not prematurely lock the project into:

```text
GAN
VAE
Diffusion
Point-cloud network
```

without a later research decision.

---

# 42. Local vs Colab

Current development should remain local.

Reason:

```text
edit
→ run
→ inspect
→ debug
→ Rhino / Grasshopper
```

The local RTX 4050 is sufficient for iterative development.

Colab is more appropriate later for:

```text
batch preprocessing
hundreds / thousands of images
long experiments
training
```

Expected long-term division:

```text
LOCAL
→ development
→ validation
→ Rhino integration

COLAB
→ batch processing
→ training
```

The data format should remain compatible across both.

---

# 43. Important Project Files

Keep:

```text
build_dataset.py
```

Purpose:

```text
RGB → Depth Anything → Scene Dataset
```

Keep:

```text
img2pointclouds.py
```

Purpose:

```text
RGB + relative depth → 2.5D point cloud / PLY
```

Keep:

```text
window_detector_v6_1.py
```

Current Window detector baseline.

Keep:

```text
clean_scene_outputs.py
```

Scene-output cleanup utility.

It should default to dry-run behavior.

Keep Depth Anything official:

```text
run.py
```

for reference / debugging.

---

# 44. Historical Prototype Files

Older files may include:

```text
semantic_test.py

architectural_elements_v1.py
architectural_elements_v2.py
architectural_elements_v3.py
architectural_elements_v4.py
architectural_elements_v4_1.py

window_detector_v5.py
window_detector_v5_1.py
window_detector_v5_1_fixed.py
window_detector_v6.py
```

These are historical experiments.

Recommended:

```text
archive/
```

rather than immediately deleting them.

Do not delete files automatically without user approval.

---

# 45. Generated Detector Folders

Scene folders may accumulate:

```text
architectural_v*
window_v*
semantic_test_output
```

These are generated artifacts.

They can be cleaned using:

```text
clean_scene_outputs.py
```

Core scene source files must remain untouched.

---

# 46. Code Quality Expectations

When modifying detector code:

1. Preserve CLI clarity.
2. Preserve reproducibility.
3. Save diagnostics.
4. Save useful preview images.
5. Save JSON evidence.
6. Record algorithm version.
7. Preserve depth conventions.
8. Preserve Rhino coordinate convention.
9. Keep GPU cleanup.
10. Avoid unnecessary simultaneous model loading.
11. Prefer modular functions.
12. Preserve raw evidence even if final candidate is rejected.
13. Record rejection reasons.
14. Avoid hidden behavior changes.
15. Avoid per-scene hacks.

---

# 47. Future Refactor

Current detector files are becoming large.

After V6.x behavior is validated, consider refactoring into modules such as:

```text
playing_models/
│
├── depth/
│   ├── inference.py
│   └── pointcloud.py
│
├── vision/
│   ├── grounding_dino.py
│   ├── sam2.py
│   ├── segformer.py
│   └── clip.py
│
├── semantics/
│   ├── facade.py
│   ├── window.py
│   ├── signboard.py
│   └── ...
│
├── spatial/
│   ├── repetition.py
│   ├── relationships.py
│   └── facade_grid.py
│
└── pipeline/
    └── street_parser.py
```

But:

**do not refactor while V6.1 validation is still unresolved.**

First validate behavior.

Then refactor.

---

# 48. Validation Principle

Do not optimize only for:

```text
"the overlay looks good"
```

Relevant evaluation dimensions:

```text
Recall
Precision
Cross-scene generalization
Spatial consistency
Depth consistency
Repeatability
Interpretability
```

Preserve information about:

```text
why a candidate was rejected
```

not just the final binary result.

---

# 49. Version Comparison

Every meaningful change should be tested on the same validation scenes.

At minimum currently:

```text
scene_000001
scene_000002
```

Eventually expand.

Recommended summary table:

```text
scene
scene_mode
effective_facade_ratio
searched_tiles
skipped_tiles
raw_candidates
after_gate
context_rejected
clip_rejected
groups_removed
final_instances
```

Do not evaluate a new version from only one scene.

---

# 50. Immediate Codex Operating Procedure

Whenever Codex enters this repository:

## First

Read:

```text
PLAYING_MODELS_CONTEXT.md
```

Then inspect actual files.

Do not assume this document is more current than the filesystem if concrete implementation differs.

## Before code modification

1. Read the relevant Python file completely.
2. Inspect the current output JSON.
3. Inspect the diagnostic images.
4. Identify the stage creating the error.
5. Report findings.
6. Wait for user approval before structural changes.

## When code modification is approved

Do not overwrite the current stable baseline.

Example:

```text
window_detector_v6_1.py
```

should remain.

Create:

```text
window_detector_v6_2.py
```

for structural iteration.

## After modification

Run regression tests on at least:

```text
scene_000001
scene_000002
```

Compare metrics and diagnostics.

Report:

```text
what changed
why it changed
which files changed
which parameters changed
scene001 result
scene002 result
regressions
remaining issues
```

---

# 51. Current First Task for Codex

The next local task should be:

> Audit Scene001 and Scene002 V6.1 results without modifying code.

Inspect:

```text
dataset/scenes/scene_000001/window_v6_1/
dataset/scenes/scene_000002/window_v6_1/
```

Compare:

```text
scene_context.json
features_window_v6_1.json
window_search_tiles_preview.jpg
allowed_facade_overlay.png
hard_exclusion_overlay.png
soft_not_window_boxes_preview.jpg
gated_window_candidates_preview.jpg
clip_verifier_preview.jpg
detections_preview.jpg
window_instances_preview.jpg
rejected_window_candidates.json
clip_rejected_candidates.json
```

Report:

### Current architecture status

### Scene001
- strengths
- weaknesses
- metrics

### Scene002
- strengths
- weaknesses
- metrics

### Cross-scene comparison

### Primary bottleneck

### Secondary bottlenecks

### Is V6.1 architecture fundamentally correct?
- Yes
- Partially
- No

### Recommended next step
Choose one:

```text
A. no code change yet
B. parameter adjustment only
C. V6.2 structural change
```

Explain the reason.

Do not modify code until approval.

---

# 52. Long-Term Success Condition

The success condition of the current phase is **not**:

```text
perfect Window boxes
```

It is:

```text
A reliable Taiwanese streetscape spatial parser
```

that converts street images into:

```text
RGB
+
Depth
+
Semantic Layers
+
Instances
+
Spatial Relationships
```

which can later become the learnable representation for:

```text
PLAYING MODELS
```

Window Detection is the first specialized module used to prove the architecture.

Do not lose sight of the larger research objective.
