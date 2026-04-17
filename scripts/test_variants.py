# scripts/test_variants.py
# Compare extraction-v1 (role-based) vs extraction-v2 (instruction-only)
# Requires GROQ_API_KEY to be set

import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.llm import call_groq, load_prompt
from app.schemas import ExtractionResult, FeatureFlag, LOCKED_FEATURES

QUERIES = [
    "3-bedroom house in CollgCr neighborhood, built in 1995, good kitchen, 2-car garage, about 1800 sqft",
    "small older home in OldTown, 1100 sqft, 1 bath, no garage, average quality everything",
    "brand new luxury house in NridgHt, excellent kitchen and exterior, 4 bedrooms, 3200 sqft living area",
]

VARIANTS = {
    "extraction-v1": "extraction",       # prompts/extraction.txt
    "extraction-v2": "extraction_v2",    # prompts/extraction_v2.txt
}


def run_variant(variant_label: str, prompt_name: str, query: str) -> dict:
    """Run one extraction call, return result dict for logging."""
    prompt = load_prompt(prompt_name)
    raw = ""
    extracted_count = 0
    assumed_count = 0
    success = False
    error = ""

    try:
        raw = call_groq(prompt, query, json_mode=True)
        data = json.loads(raw)
        features = {
            k: FeatureFlag(value=v["value"], flag=v.get("flag", "ASSUMED"))
            for k, v in data.items()
            if k in LOCKED_FEATURES
        }
        result = ExtractionResult(features=features, raw_text=raw)
        extracted_count = len(LOCKED_FEATURES) - len(result.assumed_features())
        assumed_count = len(result.assumed_features())
        success = True
    except Exception as exc:
        error = str(exc)

    return {
        "variant": variant_label,
        "query": query,
        "success": success,
        "extracted_count": extracted_count,
        "assumed_count": assumed_count,
        "raw_output": raw[:500],
        "error": error,
    }


def main() -> None:
    results = []

    for variant_label, prompt_name in VARIANTS.items():
        print(f"\nTesting {variant_label}...")
        for query in QUERIES:
            print(f"  Query: {query[:60]}...")
            entry = run_variant(variant_label, prompt_name, query)
            results.append(entry)
            status = f"extracted={entry['extracted_count']}, assumed={entry['assumed_count']}"
            print(f"  Result: {'OK' if entry['success'] else 'FAIL'} | {status}")

    logs_dir = ROOT_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)
    output_path = logs_dir / "variant_results.json"
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {output_path}")

    print("\n=== VARIANT SUMMARY ===")
    for variant_label in VARIANTS:
        variant_results = [r for r in results if r["variant"] == variant_label]
        total_extracted = sum(r["extracted_count"] for r in variant_results)
        success_rate = sum(1 for r in variant_results if r["success"]) / len(variant_results)
        print(f"{variant_label}: total_extracted={total_extracted}, success_rate={success_rate:.0%}")
    print("=== Select the winner and update chain.py default variant label ===")


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set")
        sys.exit(1)
    main()
