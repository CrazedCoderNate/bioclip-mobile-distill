"""Step 9: pre-generate the full rich write-up for every species, with Claude.

The insight: the model's output space is closed, so every answer it can give is
known at build time. Instead of one Sonnet vision call per identification, we
make one Opus text call per SPECIES here, once, and bake the results into the
app. The phone then needs no API key and no network. The identification is
local (the distilled model) and so is the encyclopedia entry (this).

Uses the Message Batches API: 4,271 requests in one batch at 50% of the standard
price, structured-output-constrained so every record matches the app's schema
exactly. Typical completion is under an hour.

    # set your key first (this script never sees it in plaintext beyond the SDK):
    export ANTHROPIC_API_KEY=sk-ant-...     # or: ant auth login

    python s09_ai_enrich.py --taxa ../BotanicalBuddy/ml/data/taxa.txt --limit 10   # smoke test
    python s09_ai_enrich.py --taxa ../BotanicalBuddy/ml/data/taxa.txt              # full run

Resumable: the batch id is cached, so a rerun retrieves the in-flight or finished
batch rather than paying for a second one. Curated safety rows
(curated_safety.csv) override the model's toxicity/edibility, and Wikipedia URLs
from s08 are merged in if present.

COST (rough, one-time): ~4,271 requests, ~400 in / ~700 out tokens each, on
claude-opus-4-8 at $5/$25 per M with the Batch API's 50% discount ≈ $40-45.
Pass --model claude-sonnet-5 to roughly halve that at some quality cost; the
model card's toxicity warning matters most for the long tail of obscure species.

SAFETY: the model is told to answer "unknown" rather than guess on toxicity and
edibility, and curated_safety.csv wins where it has an entry. Absent data must
render as "Unknown" in the app, never as "safe".
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# --- Output schema ---------------------------------------------------------
# Structured outputs constrain the response to exactly this shape, so every
# record parses. Enums carry an explicit "unknown" rather than allowing null,
# which keeps the schema simple and maps cleanly onto the app's Kotlin enums
# (which already resolve unrecognized values to UNKNOWN).
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "commonNames", "family", "description", "lifecycle", "nativeStatus",
        "toxicitySeverity", "toxicityDetail", "isEdible", "edibleParts",
        "edibleCautions", "benefits", "drawbacks", "careNotes", "careEase",
    ],
    "properties": {
        "commonNames": {"type": "array", "items": {"type": "string"}},
        "family": {"type": "string"},
        "description": {"type": "string"},
        "lifecycle": {"type": "string",
                      "enum": ["annual", "biennial", "perennial", "unknown"]},
        "nativeStatus": {"type": "string",
                         "enum": ["native", "introduced", "invasive", "unknown"]},
        "toxicitySeverity": {"type": "string",
                             "enum": ["none", "mild", "moderate", "severe", "unknown"]},
        "toxicityDetail": {"type": "string"},
        "isEdible": {"type": "string", "enum": ["yes", "no", "unknown"]},
        "edibleParts": {"type": "array", "items": {"type": "string"}},
        "edibleCautions": {"type": "string"},
        "benefits": {"type": "array", "items": {"type": "string"}},
        "drawbacks": {"type": "array", "items": {"type": "string"}},
        "careNotes": {"type": "string"},
        # 0 means unknown; 1 (fussy) .. 10 (thrives on neglect).
        "careEase": {"type": "integer"},
    },
}

SYSTEM_TEMPLATE = """You are a careful botanical reference. Given a plant's \
scientific name, return structured facts about the SPECIES (not any particular \
specimen; you are not looking at a photo).

Rules:
- Judge nativeStatus relative to this region: {region}. If the species does not \
occur there, use "introduced" or "invasive" as appropriate, or "unknown" if you \
are unsure.
- For toxicity and edibility, answer "unknown" rather than guessing. A wrong \
"safe" or "edible" claim can cause real harm. Only state a toxicity severity or \
mark a plant edible when you are confident.
- description: 2-4 plain sentences on what the plant is and how to recognize it.
- benefits / drawbacks: short noun phrases, at most 4 each, omit if none apply.
- careEase: 1 (fussy) to 10 (thrives on neglect) for a home grower, or 0 if not \
a plant anyone cultivates.
- Leave strings empty and arrays empty when you have nothing accurate to say."""


def read_taxa(path: Path) -> list[str]:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line.split("|", 1)[0].strip())
    return names


def read_curated(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("scientificName") or "").strip()
            if not name or name.startswith("#"):
                continue
            out[name] = row
    return out


def read_wikipedia_urls(facts_path: Path) -> dict[str, str]:
    """Pull just the wikipediaUrl from an s08 output, if one exists."""
    if not facts_path.exists():
        return {}
    records = json.loads(facts_path.read_text(encoding="utf-8"))
    return {r["scientificName"]: r["wikipediaUrl"]
            for r in records if r.get("wikipediaUrl")}


def build_request(name: str, model: str, region: str) -> dict:
    return {
        "custom_id": name.replace(" ", "_")[:64],
        "params": {
            "model": model,
            "max_tokens": 1500,
            # Factual extraction; thinking would multiply cost across 4,271
            # calls for little gain, and the JSON is schema-constrained anyway.
            "thinking": {"type": "disabled"},
            "system": SYSTEM_TEMPLATE.format(region=region),
            "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
            "messages": [{"role": "user", "content": f"Species: {name}"}],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--taxa", type=Path, required=True)
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--region",
                    default="Southeastern United States (USDA zones 7b-8a)")
    ap.add_argument("--out", type=Path, default=Path("data/species_facts.json"))
    ap.add_argument("--curated", type=Path, default=Path("curated_safety.csv"))
    ap.add_argument("--wikipedia", type=Path,
                    help="An s08 species_facts.json to borrow wikipediaUrl from.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-id-file", type=Path, default=Path("data/batch_id.txt"))
    args = ap.parse_args()

    try:
        from anthropic import Anthropic
    except ImportError:
        print("pip install anthropic", file=sys.stderr)
        return 1

    names = read_taxa(args.taxa)
    if args.limit:
        names = names[: args.limit]
    print(f"{len(names):,} species | model {args.model} | region '{args.region}'")

    client = Anthropic()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # --- Create or resume the batch --------------------------------------
    by_custom_id = {n.replace(" ", "_")[:64]: n for n in names}
    if args.batch_id_file.exists():
        batch_id = args.batch_id_file.read_text().strip()
        print(f"Resuming batch {batch_id}")
    else:
        print("Creating batch ...")
        batch = client.messages.batches.create(
            requests=[build_request(n, args.model, args.region) for n in names]
        )
        batch_id = batch.id
        args.batch_id_file.write_text(batch_id)
        print(f"  batch {batch_id} ({batch.processing_status})")

    # --- Poll -------------------------------------------------------------
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        c = batch.request_counts
        print(f"  {batch.processing_status}: "
              f"{c.succeeded} ok, {c.processing} processing, {c.errored} errored",
              flush=True)
        time.sleep(30)

    print(f"Batch ended: {batch.request_counts.succeeded} succeeded, "
          f"{batch.request_counts.errored} errored")

    # --- Collect results --------------------------------------------------
    ai: dict[str, dict] = {}
    n_bad = 0
    for result in client.messages.batches.results(batch_id):
        name = by_custom_id.get(result.custom_id)
        if name is None:
            continue
        if result.result.type != "succeeded":
            n_bad += 1
            continue
        msg = result.result.message
        text = next((b.text for b in msg.content if b.type == "text"), None)
        if not text:
            n_bad += 1
            continue
        try:
            ai[name] = json.loads(text)
        except json.JSONDecodeError:
            n_bad += 1
    if n_bad:
        print(f"  {n_bad} results were unusable (errored or unparseable)")

    # --- Merge ------------------------------------------------------------
    curated = read_curated(args.curated)
    wiki = read_wikipedia_urls(args.wikipedia) if args.wikipedia else {}
    if curated:
        print(f"Curated safety overrides: {len(curated)}")

    records = []
    for name in names:
        a = ai.get(name, {})
        commons = a.get("commonNames") or []
        common = min(commons, key=lambda c: (c[:1].isupper(), len(c))) if commons else None

        tox_sev = a.get("toxicitySeverity", "unknown")
        tox_detail = a.get("toxicityDetail", "")
        is_edible = a.get("isEdible", "unknown")
        edible_parts = a.get("edibleParts") or []
        edible_cautions = a.get("edibleCautions", "")

        # Curated wins for the safety-critical fields.
        if name in curated:
            c = curated[name]
            tox_sev = (c.get("toxicitySeverity") or tox_sev).strip().lower() or tox_sev
            tox_detail = (c.get("toxicityDetail") or "").strip() or tox_detail
            raw = (c.get("isEdible") or "").strip().lower()
            if raw in ("true", "yes"):
                is_edible = "yes"
            elif raw in ("false", "no"):
                is_edible = "no"
            if c.get("edibleParts"):
                edible_parts = [p.strip() for p in c["edibleParts"].split()
                                if p.strip()]

        records.append({
            "scientificName": name,
            "commonName": common,
            "family": (a.get("family") or "").strip() or None,
            "description": (a.get("description") or "").strip(),
            "lifecycle": a.get("lifecycle", "unknown"),
            "nativeStatus": a.get("nativeStatus", "unknown"),
            "nativeRegion": args.region,
            "toxicitySeverity": tox_sev,
            "toxicityDetail": tox_detail or None,
            "isEdible": is_edible,
            "edibleParts": edible_parts,
            "edibleCautions": edible_cautions or None,
            "benefits": a.get("benefits") or [],
            "drawbacks": a.get("drawbacks") or [],
            "careNotes": (a.get("careNotes") or "").strip() or None,
            "careEase": int(a.get("careEase") or 0),
            "wikipediaUrl": wiki.get(name),
        })

    args.out.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")

    n = len(records)
    described = sum(1 for r in records if r["description"])
    toxed = sum(1 for r in records if r["toxicitySeverity"] != "unknown")
    print(f"\nWrote {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")
    print(f"  described : {described:,}/{n:,} ({described / n:.0%})")
    print(f"  toxicity  : {toxed:,}/{n:,} ({toxed / n:.0%})")
    print("\nNext: python s07_android_assets.py --facts data/species_facts.json ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
