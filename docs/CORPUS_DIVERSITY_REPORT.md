# Corpus Diversity Report

This is a diagnostic of the current **49-scene parsed subset**, not the final 500–1000-scene corpus.

| Class | Scene occurrence | Positive pixel ratio | Instances |
|---|---:|---:|---:|
| facade | 49/49 | 0.629569 | 0 |
| window | 47/49 | 0.153536 | 1412 |
| door | 49/49 | 0.319841 | 146 |
| balcony | 49/49 | 0.385142 | 175 |
| railing | 47/49 | 0.097038 | 133 |
| grille | 49/49 | 0.090374 | 207 |
| awning | 47/49 | 0.356107 | 125 |
| storefront | 49/49 | 0.339343 | 114 |
| rolling_shutter | 49/49 | 0.110079 | 147 |
| signboard | 48/49 | 0.090378 | 171 |
| air_conditioner | 48/49 | 0.098776 | 389 |
| utility_pole | 45/49 | 0.017847 | 72 |
| wire | 45/49 | 0.025639 | 0 |
| vegetation | 27/49 | 0.008254 | 0 |
| person | 42/49 | 0.010987 | 176 |
| vehicle | 36/49 | 0.026669 | 321 |
| street_object | 44/49 | 0.050515 | 173 |
| arcade_candidate | 0/49 | 0.000000 | 0 |

## Acquisition priorities

The least represented evidence is: **arcade_candidate, vegetation, vehicle, person, street_object, utility_pole**. Future lawful sourcing should prioritize independent source groups containing narrow alleys, arcade/ground-floor commercial structure, old apartments and townhouses, sign-heavy shophouses, metal-sheet additions, vegetation-heavy and vehicle-heavy streets, while retaining street-perspective and dense-façade balance. Current CLIP scores are soft descriptors, not hard scene labels. Suggested inverse-frequency loss statistics can be derived from the positive pixel ratios above; loss design remains out of scope.
