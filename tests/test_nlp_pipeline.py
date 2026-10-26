"""Unit tests for NLP preprocessing and linguistic analysis engine."""
import pytest
from src.nlp_pipeline import NLPPipeline


@pytest.fixture
def pipeline():
    return NLPPipeline()


def test_clean_text(pipeline):
    dirty = "Python&nbsp;error:&nbsp;&nbsp;ModuleNotFound&amp;crash\n\n"
    cleaned = pipeline.clean_text(dirty)
    assert "&nbsp;" not in cleaned
    assert "&amp;" not in cleaned
    assert "  " not in cleaned


def test_linguistic_analysis_severity(pipeline):
    query = "Python raises OutOfMemoryError: heap memory exhausted"
    res = pipeline.analyze_linguistics(query)
    assert res["token_count"] > 0
    assert "OutOfMemoryError" in res["entities"]["errors"]
    assert res["severity"]["level"] == "CRITICAL_FATAL"


def test_pos_tagging_and_lemmas(pipeline):
    query = "Developers connecting services"
    res = pipeline.analyze_linguistics(query)
    assert len(res["pos_tags"]) > 0
    assert len(res["lemmas_vs_stems"]) > 0

    # Verify that Porter stem and WordNet lemma are computed
    stem_entry = next((item for item in res["lemmas_vs_stems"] if item["original"] == "connecting"), None)
    if stem_entry:
        assert stem_entry["porter_stem"] == "connect"


def test_typo_correction(pipeline):
    """Test Damerau-Levenshtein typo correction for common dev terms."""
    res = pipeline.suggest_typo_correction("python posgresql timeeout eror")
    corrected = res["corrected_query"]
    corrected_lower = corrected.lower()
    assert "postgresql" in corrected_lower or "timeout" in corrected_lower or "error" in corrected_lower


def test_wordnet_expansion(pipeline):
    """Test WordNet synset expansion for vocabulary enrichment."""
    res = pipeline.expand_query_wordnet("crash connection terminate")
    assert "expanded_query" in res
    assert isinstance(res["expanded_terms"], dict)


def test_perplexity(pipeline):
    """Test bigram language model perplexity scoring."""
    ppl_normal = pipeline.calculate_perplexity("connection refused on port 5432")
    assert ppl_normal > 0.0


def test_domain_guardrail(pipeline):
    """Test technical vs non-technical input classification."""
    non_tech = pipeline.check_domain_guardrail("What is the capital of France?")
    assert non_tech["is_technical"] is False
    assert "specializes in developer runtime errors" in non_tech["guardrail_message"]

    tech = pipeline.check_domain_guardrail("Python raises ModuleNotFoundError: No module named 'fastapi'")
    assert tech["is_technical"] is True


def test_stack_trace_parser(pipeline):
    """Test AST/regex crash log and stack trace sanitizer."""
    raw_trace = """
    Traceback (most recent call last):
      File "/app/backend/server.py", line 42, in process_request
        res = database.query("SELECT * FROM users")
    psycopg2.OperationalError: could not connect to server: Connection refused
    """
    parsed = pipeline.parse_stack_trace(raw_trace)
    assert parsed["is_stack_trace"] is True
    assert "psycopg2.OperationalError" in parsed["clean_query"]
    assert parsed["isolated_frame"] is not None
    assert "server.py" in parsed["isolated_frame"]


def test_diagnostic_wizard(pipeline):
    """Test generation of guided troubleshooter diagnostic chips."""
    query = "Connection refused on port 5432"
    top_doc = {"text": "PostgreSQL service is not running or listening on port 5432.", "technology": "SQL"}
    questions = pipeline.generate_diagnostic_wizard(query, top_doc)
    assert len(questions) > 0
    assert any("PostgreSQL" in q["question"] or "port" in q["question"].lower() for q in questions)


def test_indic_translation():
    """Test multi-lingual Indic translation and cross-lingual detection."""
    from src.translator import get_translator, translate_indic_query_to_english

    # Test detection of Hindi input
    detected = translate_indic_query_to_english("सर्वर से कनेक्शन विफल हो गया")
    assert detected["language_code"] == "hi"
    assert detected["was_translated"] is True
    assert "connection" in detected["translated_query"].lower() or "server" in detected["translated_query"].lower()

    # Test localized explanation in Marathi
    translator = get_translator()
    res = translator.explain_diagnostic(
        query="ModuleNotFoundError: No module named 'fastapi'",
        root_cause="FastAPI package is not installed in the environment.",
        action_steps=["Run pip install fastapi to install the package."],
        code_snippet="pip install fastapi",
        target_lang="mr"
    )
    assert res["language"] == "mr"
    assert "मराठी" in res["language_native"]
    assert len(res["action_steps_localized"]) > 0


def test_json_fetch_guardrail_and_wizard(pipeline):
    """Verify that JSON file fetching queries are accepted by guardrails and generate follow-up questions."""
    query = "I cannot fetch data from json file"
    guard = pipeline.check_domain_guardrail(query)
    assert guard["is_technical"] is True
    assert guard["guardrail_message"] is None

    questions = pipeline.generate_diagnostic_wizard(query, {"text": "Fetch JSON error in frontend React or Node.js"})
    assert len(questions) >= 2
    question_ids = [q["id"] for q in questions]
    assert "json_runtime" in question_ids
    assert "json_symptom" in question_ids


def test_wizard_option_resolution(pipeline):
    """Verify scenario resolution for Python environments, interpreters, and JSON fetching."""
    # Test 1: Virtual Environment (.venv / venv)
    res_venv = pipeline.resolve_wizard_option("No module named 'fastapi'", "py_env", "Virtual Environment (.venv / venv)")
    assert "virtual environment" in res_venv["root_cause"].lower() or ".venv" in res_venv["root_cause"].lower()
    assert any("activate" in s.lower() for s in res_venv["action_steps"])
    assert "source .venv/bin/activate" in res_venv["code_snippet"]

    # Test 2: Installed but IDE using wrong interpreter
    res_ide = pipeline.resolve_wizard_option("No module named 'fastapi'", "py_install", "Installed but IDE using wrong interpreter")
    assert "interpreter" in res_ide["root_cause"].lower()
    assert any("Select Interpreter" in s for s in res_ide["action_steps"])

    # Test 3: Package name differs from import name
    res_diff = pipeline.resolve_wizard_option("No module named 'yaml'", "py_install", "Package name differs from import name")
    assert "pyyaml" in res_diff["code_snippet"].lower()

    # Test 4: JSON Frontend Fetch
    res_json = pipeline.resolve_wizard_option("I cannot fetch data from json file", "json_runtime", "Browser / Frontend JS (fetch / axios / React / Vite)")
    assert "public/" in res_json["root_cause"].lower() or "relative path" in res_json["root_cause"].lower()
    assert any("public" in s.lower() for s in res_json["action_steps"])


def test_compound_wizard_resolution(pipeline):
    """Verify compound resolution when user selects multiple wizard options simultaneously."""
    selected = {
        "py_env": "Virtual Environment (.venv / venv)",
        "py_install": "Installed but IDE using wrong interpreter"
    }
    compound = pipeline.resolve_compound_wizard_options("No module named 'fastapi'", selected)
    assert "misalignment" in compound["scenario_title"].lower() or "interpreter" in compound["scenario_title"].lower()
    assert "virtual environment" in compound["root_cause"].lower()
    assert "interpreter" in compound["root_cause"].lower()
    assert any("Select Interpreter" in s for s in compound["action_steps"])


def test_dynamic_kb_extraction_in_wizard(pipeline):
    """Verify that unmapped queries dynamically extract options from retrieved knowledge base results."""
    all_results = [
        {"title": "Docker Container Networking and Host Port Binding", "technologies": ["Docker", "Networking"]},
        {"title": "Kubernetes Pod CrashLoopBackOff", "technologies": ["Kubernetes", "K8s"]}
    ]
    questions = pipeline.generate_diagnostic_wizard("custom orchestration glitch", all_results=all_results)
    assert len(questions) >= 1
    assert questions[0]["id"] == "kb_scenario"
    assert len(questions[0]["options"]) == 2



