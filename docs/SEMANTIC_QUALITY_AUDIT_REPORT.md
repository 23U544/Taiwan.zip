# Semantic Quality Audit Report

## Scope and metric limits

This audit reads all 49 frozen Parser V1 outputs and 3,888 instances. It does not use human ground truth, so none of the statistics are precision, recall, or benchmark IoU. Coverage, overlap, containment, mask/bbox consistency, confidence, and depth are heuristic quality signals used to prioritize review and training-channel policy.

## Quantitative findings

| Tag | Occurrence | Instances | Median scene coverage | P95 | Maximum | Main finding |
|---|---:|---:|---:|---:|---:|---|
| facade | 49/49 | 0 | 64.08% | 100.00% | 100.00% | useful broad context; some full-image masks |
| window | 47/49 | 1,412 | 8.97% | 25.16% | 31.36% | best architectural instance layer, but signs/openings remain |
| door | 49/49 | 146 | 37.83% | 96.55% | 99.98% | catastrophic oversized proposals |
| balcony | 49/49 | 175 | 45.15% | 96.41% | 99.98% | catastrophic oversized proposals and class overlap |
| railing | 47/49 | 133 | 6.82% | 57.07% | 99.77% | mixed localized and broad masks |
| grille | 49/49 | 207 | 4.93% | 66.75% | 88.31% | unresolved fine structure and railing confusion |
| awning | 47/49 | 125 | 35.01% | 96.09% | 99.97% | facade/sign/storefront confusion |
| storefront | 49/49 | 114 | 39.98% | 96.16% | 99.97% | door/opening confusion |
| rolling shutter | 49/49 | 147 | 7.07% | 69.71% | 99.53% | valid small cases plus broad opening outliers |
| signboard | 48/49 | 171 | 5.23% | 50.58% | 92.64% | many valid signs; occasional whole-facade proposal |
| air conditioner | 48/49 | 389 | 2.41% | 90.98% | 99.98% | small cases good, catastrophic global outliers |
| utility pole | 45/49 | 72 | 0.99% | 7.48% | 70.28% | mostly sparse, one severe outlier |
| wire | 45/49 | 0 | 1.40% | 5.88% | 9.04% | bounded but line evidence is not semantic proof |
| vegetation | 27/49 | 0 | 0.003% | 3.67% | 14.28% | sparse corpus coverage |
| person | 42/49 | 176 | 0.65% | 3.64% | 86.19% | mostly small, one catastrophic outlier |
| vehicle | 36/49 | 321 | 1.65% | 10.68% | 13.00% | most consistently bounded instance class |
| street object | 44/49 | 173 | 1.89% | 58.63% | 95.40% | broad class with severe outliers |
| arcade candidate | 0/49 | 0 | 0% | 0% | 0% | intentionally empty |
| window group | hierarchy only | 127 | n/a | n/a | n/a | not unioned into Window semantic mask |

All instance depth values are finite. No centroid falls outside its recorded bbox. Only 12 of 3,888 masks exceed `mask_area / bbox_area > 1.2`, so spill beyond the accepted bbox is not the dominant failure.

## Coverage anomaly diagnosis

The repeated worst scenes are `scene_000001`, `scene_000013`, `scene_000037`, `scene_000039`, and `scene_000015`. Door, balcony, awning, storefront, and AC often approach the entire image in the same scenes. Contact sheets show accepted bboxes spanning a facade or full frame. SAM masks usually occupy 69–84% of their bbox, so SAM follows an already oversized proposal; this is primarily Grounding DINO semantic/scale confusion, with SAM boundary coarseness secondary.

The four special classes are therefore not examples of genuinely high occurrence:

- Door: whole facades, grille windows, and generic rectangular openings are accepted as doors.
- Balcony: whole buildings, temple facades, and grilles are accepted alongside some valid balcony railings.
- Awning: whole facades and signboards are frequent; some local awnings are valid.
- Storefront: whole facades/buildings and generic openings compete strongly with doors.

## Multi-label overlap and confusion proxies

Reasonable overlap exists for facade–Window (84.8% of the smaller mask), facade–AC (83.6%), and balcony–railing (90.3%). However, the following aggregate overlap-over-smaller values are too systematic to interpret as independent semantic evidence without review:

- storefront–door: **95.6%**, 255 bbox containment pairs;
- awning–storefront: **95.3%**, 193 containment pairs;
- balcony–awning: **95.0%**, 242 containment pairs;
- grille–railing: **79.3%**;
- Window–balcony: **76.8%**, 1,680 containment pairs;
- door–Window: **74.5%**, 1,100 containment pairs.

These remain proxies: multi-label overlap can be legitimate, and no ground-truth confusion matrix is claimed.

## Visual audit

Fifteen contact sheets cover facade, Window, door, balcony, railing, awning, storefront, rolling shutter, signboard, AC, vegetation, person, vehicle, grille, and wire. Each sheet combines high/median/lowest non-zero scene coverage with high/median/low confidence, largest, and deterministic random instances where instances exist. Tiles show RGB, colored mask overlay, bbox, scene/instance ID, confidence, area, and selection reason.

The review queue contains 249 scene/instance rows in `dataset/semantic_review_manifest.csv`, initially `unchecked`. Highest priority is the repeated catastrophic scene/class combinations above, followed by AC/person/street-object coverage outliers and the highest-coverage Window scene (`scene_000025`).

## Training recommendation

- CORE: facade at reduced weight; vehicle.
- AUXILIARY: Window, signboard, vegetation, person, all with conservative class weights.
- LOW_WEIGHT but excluded from candidate V1 pending review: railing, rolling shutter, utility pole.
- EXCLUDE_V1: door, balcony, grille, awning, storefront, AC, wire, street object, arcade candidate.

This is a training policy, not an accuracy score. Canonical masks remain untouched. The conservative candidate is 7 channels: depth normalization plus facade, Window, signboard, vegetation, person, and vehicle.

## Readiness

- Most reliable current instance channel: vehicle.
- Most useful but imperfect architectural layers: facade context and Window.
- Main blocker for all-channel training: catastrophic oversized cross-class proposals, not data corruption or depth alignment.
- Human review is still required before promoting excluded/low-weight classes.
