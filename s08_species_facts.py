"""Step 8: prefetch encyclopedia facts for every species the model can predict.

The model's output space is fixed at build time, so every answer it can ever
give is known in advance. That means the enrichment can be fetched once, here,
and shipped with the app instead of being generated per identification.

Sources, and what each is actually good for:

  Wikidata (SPARQL)   family, English common names. Structured and reliable.
  Wikipedia (REST)    a paragraph of description. Good coverage for plants.

What NEITHER provides, and no comparable open source does either: per-species
toxicity, edibility, or care instructions. Those come from `curated_safety.csv`
if you supply one, and stay UNKNOWN otherwise. See the safety note below.

    python s08_species_facts.py --taxa ../BotanicalBuddy/ml/data/taxa.txt
    python s08_species_facts.py --taxa ... --limit 50      # smoke test

Resumable: every fetched record is cached to disk, so a rerun only requests
what is missing. Expect roughly 10 minutes for 4,271 species.

SAFETY NOTE
-----------
Absent data must render as "Unknown" in the app, never as "safe". A plant with
no toxicity record is one nobody has told us about, which is not the same as
one known to be harmless. The Android enums already default to UNKNOWN; do not
"helpfully" map missing to NONE anywhere downstream.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Wikimedia asks for a descriptive User-Agent with contact information.
# Anonymous or generic agents get rate limited hard.
USER_AGENT = (
    "FloraBotanicalBuddy/1.0 "
    "(https://github.com/CrazedCoderNate/bioclip-mobile-distill; "
    "nathanshanehamilton@gmail.com)"
)

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIPEDIA_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"

# Wikidata: taxon name (P225), parent taxon (P171), taxon rank (P105),
# family rank (Q35409), common name (P1843).
SPARQL_TEMPLATE = """
SELECT ?name ?familyName ?commonName WHERE {
  VALUES ?name { %s }
  ?taxon wdt:P225 ?name .
  OPTIONAL {
    ?taxon wdt:P171* ?fam .
    ?fam wdt:P105 wd:Q35409 ; wdt:P225 ?familyName .
  }
  OPTIONAL {
    ?taxon wdt:P1843 ?commonName .
    FILTER(lang(?commonName) = "en")
  }
}
"""


def http_json(url: str, timeout: int = 60) -> dict:
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def read_taxa(path: Path) -> list[str]:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.append(line.split("|", 1)[0].strip())
    return names


# --- Wikidata --------------------------------------------------------------

def fetch_wikidata(names: list[str], batch_size: int = 150) -> dict[str, dict]:
    """Family and common name for a list of scientific names.

    Batched through a VALUES clause: one query per 150 species rather than
    4,271 individual requests. Wikidata returns the cross product of the
    OPTIONAL matches, so a species with four recorded common names comes back
    as four rows that have to be folded together.
    """
    out: dict[str, dict] = {}
    total = (len(names) + batch_size - 1) // batch_size

    for i in range(0, len(names), batch_size):
        batch = names[i:i + batch_size]
        values = " ".join(f'"{n}"' for n in batch if '"' not in n)
        query = SPARQL_TEMPLATE % values
        url = f"{WIKIDATA_ENDPOINT}?format=json&query={urllib.parse.quote(query)}"

        for attempt in range(4):
            try:
                data = http_json(url, timeout=120)
                break
            except Exception as e:
                if attempt == 3:
                    print(f"  batch {i // batch_size + 1}/{total} failed: {e}",
                          file=sys.stderr)
                    data = {"results": {"bindings": []}}
                    break
                # Wikidata throttles aggressively; back off rather than hammer.
                time.sleep(5 * (attempt + 1))

        for b in data["results"]["bindings"]:
            name = b["name"]["value"]
            rec = out.setdefault(name, {"family": None, "commonNames": []})
            fam = b.get("familyName", {}).get("value")
            if fam and not rec["family"]:
                rec["family"] = fam
            common = b.get("commonName", {}).get("value")
            if common and common not in rec["commonNames"]:
                rec["commonNames"].append(common)

        print(f"  wikidata batch {i // batch_size + 1}/{total} "
              f"({len(out):,} species so far)")
        time.sleep(1.0)   # be a good citizen

    return out


# --- Wikipedia -------------------------------------------------------------

def fetch_wikipedia_one(name: str) -> tuple[str, dict | None]:
    url = WIKIPEDIA_SUMMARY + urllib.parse.quote(name.replace(" ", "_"))
    for attempt in range(3):
        try:
            data = http_json(url, timeout=30)
            # Disambiguation pages describe the word, not the plant.
            if data.get("type") != "standard":
                return name, None
            return name, {
                "description": (data.get("extract") or "").strip(),
                "wikipediaUrl": data.get("content_urls", {})
                                    .get("desktop", {}).get("page"),
                "thumbnail": data.get("thumbnail", {}).get("source"),
            }
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return name, None      # no article, not an error
            if e.code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            return name, None
        except Exception:
            if attempt == 2:
                return name, None
            time.sleep(2)
    return name, None


def fetch_wikipedia(names: list[str], workers: int = 8) -> dict[str, dict]:
    out: dict[str, dict] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name, rec in pool.map(fetch_wikipedia_one, names):
            done += 1
            if rec:
                out[name] = rec
            if done % 250 == 0:
                print(f"  wikipedia {done:,}/{len(names):,} "
                      f"({len(out):,} found)")
    return out


# --- Curated safety --------------------------------------------------------

def read_curated_safety(path: Path) -> dict[str, dict]:
    """Hand-verified toxicity and edibility, keyed by scientific name.

    Deliberately a separate, small, human-maintained file. There is no open
    dataset covering toxicity for thousands of species, and generating it
    automatically would mean shipping guesses about whether plants are safe to
    touch or eat. Anything absent here stays UNKNOWN in the app.

    Columns: scientificName, toxicitySeverity, toxicityDetail, isEdible,
             edibleParts
    toxicitySeverity is one of none, mild, moderate, severe.
    """
    if not path.exists():
        return {}
    out = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("scientificName") or "").strip()
            # Skip blanks and the '#' comment rows, which csv.DictReader would
            # otherwise hand back as a species literally named "# ...".
            if not name or name.startswith("#"):
                continue
            edible_raw = (row.get("isEdible") or "").strip().lower()
            out[name] = {
                "toxicitySeverity": (row.get("toxicitySeverity") or "").strip().lower() or None,
                "toxicityDetail": (row.get("toxicityDetail") or "").strip() or None,
                "isEdible": {"true": True, "yes": True, "false": False, "no": False}
                            .get(edible_raw),
                "edibleParts": (row.get("edibleParts") or "").strip() or None,
            }
    return out


# --- Main ------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--taxa", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("species_facts.json"))
    ap.add_argument("--cache", type=Path, default=Path("data/facts_cache.json"))
    ap.add_argument("--curated", type=Path, default=Path("curated_safety.csv"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if not args.taxa.exists():
        print(f"No taxa list at {args.taxa}", file=sys.stderr)
        return 1

    names = read_taxa(args.taxa)
    if args.limit:
        names = names[: args.limit]
    print(f"{len(names):,} species")

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(args.cache.read_text(encoding="utf-8")) \
        if args.cache.exists() else {"wikidata": {}, "wikipedia": {}}

    # --- Wikidata ---------------------------------------------------------
    missing_wd = [n for n in names if n not in cache["wikidata"]]
    if missing_wd:
        print(f"\nWikidata: fetching {len(missing_wd):,} "
              f"({len(names) - len(missing_wd):,} cached)")
        fetched = fetch_wikidata(missing_wd)
        for n in missing_wd:
            # Record misses too, so a rerun does not retry them forever.
            cache["wikidata"][n] = fetched.get(n, {"family": None, "commonNames": []})
        args.cache.write_text(json.dumps(cache), encoding="utf-8")
    else:
        print("\nWikidata: all cached")

    # --- Wikipedia --------------------------------------------------------
    missing_wp = [n for n in names if n not in cache["wikipedia"]]
    if missing_wp:
        print(f"\nWikipedia: fetching {len(missing_wp):,} "
              f"({len(names) - len(missing_wp):,} cached)")
        fetched = fetch_wikipedia(missing_wp, workers=args.workers)
        for n in missing_wp:
            cache["wikipedia"][n] = fetched.get(n) or {}
        args.cache.write_text(json.dumps(cache), encoding="utf-8")
    else:
        print("\nWikipedia: all cached")

    # --- Merge ------------------------------------------------------------
    curated = read_curated_safety(args.curated)
    if curated:
        print(f"\nCurated safety records: {len(curated):,}")
    else:
        print(f"\nNo {args.curated} found. Toxicity and edibility will be "
              f"UNKNOWN for every species.")

    records = []
    have_desc = have_family = have_common = 0
    for name in names:
        wd = cache["wikidata"].get(name) or {}
        wp = cache["wikipedia"].get(name) or {}
        safety = curated.get(name, {})

        commons = wd.get("commonNames") or []
        # Prefer a lowercase vernacular ("red maple") over a title-cased one;
        # Wikidata holds both and the lowercase form reads better in the UI.
        common = None
        if commons:
            common = min(commons, key=lambda c: (c[0].isupper(), len(c)))

        desc = wp.get("description") or ""
        if desc:
            have_desc += 1
        if wd.get("family"):
            have_family += 1
        if common:
            have_common += 1

        records.append({
            "scientificName": name,
            "commonName": common,
            "family": wd.get("family"),
            "description": desc,
            "wikipediaUrl": wp.get("wikipediaUrl"),
            # Present only when a human put them in curated_safety.csv.
            "toxicitySeverity": safety.get("toxicitySeverity"),
            "toxicityDetail": safety.get("toxicityDetail"),
            "isEdible": safety.get("isEdible"),
            "edibleParts": safety.get("edibleParts"),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, ensure_ascii=False),
                        encoding="utf-8")

    n = len(records)
    print(f"\nWrote {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")
    print(f"  description : {have_desc:,}/{n:,} ({have_desc / n:.0%})")
    print(f"  family      : {have_family:,}/{n:,} ({have_family / n:.0%})")
    print(f"  common name : {have_common:,}/{n:,} ({have_common / n:.0%})")
    print(f"  toxicity    : {len(curated):,}/{n:,} ({len(curated) / n:.0%}) "
          f"curated only, the rest render as Unknown")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
