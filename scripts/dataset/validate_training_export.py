import argparse,csv,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument("path");args=p.parse_args();root=Path(args.path)
schema=json.loads((root/"channel_schema.json").read_text()); rows=list(csv.DictReader((root/"samples.csv").open(encoding="utf-8-sig"))); errors=[]; groups={}
for row in rows:
    group=row["split_group"]; split=row["split"]
    if group in groups and groups[group]!=split: errors.append(f"split_leakage:{group}")
    groups[group]=split
    with np.load(root/"samples"/f'{row["sample_id"]}.npz') as data:
        tensor=data["tensor"]
        if tensor.shape!=(len(schema["channels"]),int(row["target_height"]),int(row["target_width"])): errors.append(f"shape:{row['sample_id']}")
        if not np.isfinite(tensor).all(): errors.append(f"non_finite:{row['sample_id']}")
        if not np.isin(tensor[1:],(0,1)).all(): errors.append(f"semantic_non_binary:{row['sample_id']}")
print(json.dumps({"samples":len(rows),"groups":len(groups),"valid":not errors,"errors":errors[:100]},indent=2));raise SystemExit(1 if errors else 0)
