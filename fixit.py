#!/usr/bin/env python3
"""FixItAI Terminal CLI - Instant Developer Diagnostic & Error Retrieval.

Usage:
  # 1. Direct query from terminal:
  python fixit.py "Python ModuleNotFoundError: No module named 'fastapi'"

  # 2. Pipe terminal crash / stack trace:
  python main.py 2>&1 | python fixit.py

  # 3. Choose regional diagnostic language:
  python fixit.py "Java NullPointerException" --lang hi
  python fixit.py "Docker port in use" --lang mr
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

# ANSI Color Codes
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def print_banner():
    print(f"\n{BOLD}{CYAN}FixItAI Diagnostic Intelligence{RESET} {DIM}(Local Offline CLI){RESET}")
    print(DIM + "─" * 60 + RESET)


def main():
    parser = argparse.ArgumentParser(
        description="FixItAI Terminal Diagnostic CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("query", nargs="*", help="Technical error message or question")
    parser.add_argument("--lang", default="en", choices=["en", "hi", "mr", "bn", "ta", "te"], help="Target regional language")
    parser.add_argument("--mode", default="Hybrid", choices=["Hybrid", "Linear-Hybrid", "Semantic", "BM25", "TF-IDF"], help="Retrieval scheme")
    parser.add_argument("--top", type=int, default=3, help="Number of evidence passages")
    args = parser.parse_args()

    raw_input = ""
    # Check if input is piped through stdin
    if not sys.stdin.isatty():
        try:
            raw_input = sys.stdin.read().strip()
        except Exception:
            raw_input = ""

    # Or take from CLI arguments
    if not raw_input:
        raw_input = " ".join(args.query).strip()

    if not raw_input:
        print_banner()
        print(f"{YELLOW}No input provided.{RESET}")
        print("Usage examples:")
        print("  python fixit.py \"Python raises ModuleNotFoundError: fastapi\"")
        print("  python my_script.py 2>&1 | python fixit.py")
        sys.exit(0)

    print_banner()

    # 1. Linguistic Stack Trace & NLP Parsing
    from src.nlp_pipeline import get_nlp_pipeline
    nlp = get_nlp_pipeline()

    parsed_trace = nlp.parse_stack_trace(raw_input)
    clean_query = parsed_trace.get("clean_query") or raw_input
    if parsed_trace.get("isolated_frame"):
        print(f"{BOLD}{YELLOW}Isolated Crash Frame:{RESET} {parsed_trace['isolated_frame']}")

    # 2. Cross-Lingual Input Translation if needed
    from src.translator import translate_indic_query_to_english, get_translator
    indic_trans = translate_indic_query_to_english(clean_query)
    search_query = indic_trans["translated_query"]
    target_lang = args.lang
    if indic_trans["was_translated"] and target_lang == "en":
        target_lang = indic_trans["detected_language"]
        print(f"{DIM}Detected Input Language: {indic_trans['language_name']} -> Translated Query for Search: '{search_query}'{RESET}")

    # 3. Domain Guardrail check
    guardrail = nlp.check_domain_guardrail(search_query)
    if not guardrail["is_technical"]:
        print(f"\n{YELLOW}Notice:{RESET} {guardrail['guardrail_message']}")
        print(f"{DIM}Input was analyzed as non-technical/conversational.{RESET}\n")
        sys.exit(0)

    # 4. Typo suggestions
    typo_info = nlp.suggest_typo_correction(search_query)
    if typo_info["has_correction"]:
        suggestions = ", ".join(f"{c['original']} -> {c['suggestion']}" for c in typo_info["corrections"])
        print(f"{DIM}Auto-corrected terms: {suggestions}{RESET}")

    # 5. Execute Retrieval (Try local FastAPI if running, otherwise in-process)
    import urllib.request
    import urllib.error

    results = []
    diagnostic = None

    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8000/api/search",
            data=json.dumps({"query": search_query, "technology": "All", "mode": args.mode, "top_k": args.top}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            results = data.get("results", [])
            diagnostic = data.get("ai_solution", {})
    except Exception:
        # Fallback to local in-process retrieval
        from api import get_runtime
        from src.summarizer import get_summarizer

        retriever, _, _ = get_runtime()
        results = retriever.search(search_query, top_k=args.top, mode=args.mode)
        if results:
            top_match = results[0]
            diagnostic = get_summarizer().summarize(search_query, top_match.get("text", ""), top_match.get("title", ""))

    if not results:
        print(f"\n{RED}No matching technical solutions found for:{RESET} '{search_query}'")
        print("Try rephrasing or checking knowledge base.\n")
        sys.exit(0)

    # 6. Print Structured Diagnostic Output
    if diagnostic:
        root_cause = diagnostic.get("root_cause", "")
        action_steps = diagnostic.get("action_steps", [])
        code_snippet = diagnostic.get("code_snippet")

        # Regional Language Translation if requested
        if target_lang != "en":
            translator = get_translator()
            localized = translator.explain_diagnostic(
                query=search_query,
                root_cause=root_cause,
                action_steps=action_steps,
                code_snippet=code_snippet,
                target_lang=target_lang
            )
            print(f"\n{BOLD}{CYAN}Regional Diagnostic ({localized.get('language_name', 'Vernacular')}):{RESET}")
            print(localized.get("translated_text", ""))
        else:
            print(f"\n{BOLD}{GREEN}Root Cause:{RESET}")
            print(f"  {root_cause}\n")

            if action_steps:
                print(f"{BOLD}{GREEN}Actionable Resolution Checklist:{RESET}")
                for i, step in enumerate(action_steps, 1):
                    print(f"  {BOLD}{i}.{RESET} {step}")
                print()

            if code_snippet:
                print(f"{BOLD}{GREEN}Verified Fix / Command:{RESET}")
                print(f"{CYAN}  $ {code_snippet.strip()}{RESET}\n")

    # 7. Print Verified Citations
    print(f"{BOLD}Top Verified Reference Sources:{RESET}")
    for i, r in enumerate(results[:3], 1):
        score_pct = round(r.get("score", 0) * 100, 1)
        print(f"  [{i}] {BOLD}{r.get('title')}{RESET} {DIM}(Match: {score_pct}%){RESET}")
        print(f"      {DIM}Source: {r.get('source')} | URL: {r.get('url', 'N/A')}{RESET}")

    print(DIM + "─" * 60 + RESET + "\n")


if __name__ == "__main__":
    main()
