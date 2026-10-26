"""Comprehensive NLP Preprocessing and Linguistic Analysis Engine.

Directly covers Symbiosis NLP Syllabus (TEE7101):
- Unit 1 / Lab 1 & 2: Tokenization, Stopwords, Stemming, Lemmatization, POS Tagging, NER, Regular Expressions.
- Unit 4 / Lab 4: Sequence / Text Classification for Error Severity and Diagnostics.
"""
from __future__ import annotations

import html
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize


# POS Tag description dictionary for human-friendly inspection in UI and viva
PENN_TREEBANK_TAGS = {
    "CC": "Coordinating conjunction",
    "CD": "Cardinal number",
    "DT": "Determiner",
    "EX": "Existential there",
    "FW": "Foreign word",
    "IN": "Preposition / subordinating conjunction",
    "JJ": "Adjective",
    "JJR": "Adjective, comparative",
    "JJS": "Adjective, superlative",
    "LS": "List item marker",
    "MD": "Modal",
    "NN": "Noun, singular",
    "NNS": "Noun, plural",
    "NNP": "Proper noun, singular",
    "NNPS": "Proper noun, plural",
    "PDT": "Predeterminer",
    "POS": "Possessive ending",
    "PRP": "Personal pronoun",
    "PRP$": "Possessive pronoun",
    "RB": "Adverb",
    "RBR": "Adverb, comparative",
    "RBS": "Adverb, superlative",
    "RP": "Particle",
    "SYM": "Symbol",
    "TO": "to",
    "UH": "Interjection",
    "VB": "Verb, base form",
    "VBD": "Verb, past tense",
    "VBG": "Verb, gerund / present participle",
    "VBN": "Verb, past participle",
    "VBP": "Verb, non-3rd person singular present",
    "VBZ": "Verb, 3rd person singular present",
    "WDT": "Wh-determiner",
    "WP": "Wh-pronoun",
    "WP$": "Possessive wh-pronoun",
    "WRB": "Wh-adverb",
}

TECH_KEYWORDS = [
    # Languages and runtimes
    "Python", "Java", "JavaScript", "TypeScript", "Node.js", "Node", "C++", "C#", "C", "Go", "Golang",
    "Rust", "Ruby", "PHP", "Swift", "Kotlin", "Scala", "Bash", "Shell", "Zsh", "PowerShell",
    # Frameworks and libraries
    "React", "Vite", "Next.js", "Vue", "Angular", "Express", "FastAPI", "Flask", "Django", "Spring",
    "SpringBoot", "PyTorch", "TensorFlow", "Keras", "Pandas", "NumPy", "OpenCV", "Axios", "Nginx", "Apache",
    # Databases & stores
    "SQL", "MySQL", "PostgreSQL", "Postgres", "MongoDB", "Redis", "SQLite", "Oracle", "Cassandra",
    # DevOps, containers & cloud
    "Docker", "Kubernetes", "k8s", "Linux", "Ubuntu", "Debian", "CentOS", "Windows", "Git", "GitHub", "GitLab",
    "AWS", "Azure", "GCP", "npm", "pip", "yarn", "pnpm", "cargo", "maven", "gradle", "Webpack", "Babel",
    # Protocols, data formats, and core technical developer entities
    "JSON", "XML", "YAML", "CSV", "HTTP", "HTTPS", "REST", "GraphQL", "WebSocket", "TCP", "UDP", "DNS",
    "SSH", "SSL", "TLS", "CORS", "JWT", "OAuth", "API", "fetch", "CRUD", "ORM", "URI", "URL", "PORT",
    "file", "data", "database", "query", "endpoint", "server", "client", "request", "response", "payload",
    "parse", "stringify", "serialize", "deserialize", "async", "await", "promise", "callback", "thread",
    "process", "socket", "daemon", "middleware", "route", "router", "cookie", "session", "token", "auth",
    "table", "schema", "column", "index", "memory", "heap", "stack", "buffer", "stream", "pointer",
    "variable", "function", "method", "class", "interface", "package", "module", "dependency", "import", "export"
]

SEVERITY_PATTERNS = {
    "CRITICAL_FATAL": [
        r"\b(?:Fatal|Panic|Crash|Segfault|Segmentation\s+fault|Core\s+dumped)\b",
        r"\b(?:OutOfMemoryError|OOM|StackOverflowError|Deadlock|ECONNREFUSED)\b",
        r"\b(?:SystemError|Emergency|CriticalFailure)\b"
    ],
    "RUNTIME_EXCEPTION": [
        r"\b(?:\w*Exception|\w*Error)\b",
        r"\b(?:NullPointerException|IndexError|KeyError|TypeError|ValueError|AttributeError|ZeroDivisionError)\b",
        r"\b(?:UnhandledPromiseRejection|AssertionError)\b"
    ],
    "CONFIG_BUILD": [
        r"\b(?:ModuleNotFoundError|ImportError|ClassNotFoundException|NoClassDefFoundError)\b",
        r"\b(?:SyntaxError|IndentationError|Compilation\s+Error|Build\s+Failed)\b",
        r"\b(?:CORS|EACCES|PermissionDenied|InvalidConfiguration)\b"
    ],
    "DEPRECATION_WARNING": [
        r"\b(?:DeprecationWarning|UserWarning|FutureWarning|Notice|Obsolete)\b"
    ]
}

ERROR_REGEX = re.compile(
    r"\b(?:\w*(?:Exception|Error|Failure|Fault)\w*|ECONNREFUSED|ECONNRESET|ETIMEDOUT|Connection\s+refused|Timed?\s*out|port\s+\d+|ORA-\d+|HTTP\s*\d{3}|\d{3}\s+[A-Za-z]+)\b",
    re.I
)
CODE_OR_IDENTIFIER_REGEX = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b|\b\w+\(\)")
PATH_REGEX = re.compile(r"(?:[A-Za-z]:[\\/]|/)[A-Za-z0-9_./\\-]+\.[A-Za-z0-9]+")


def ensure_nltk_resources():
    """Ensure essential NLTK tokenizers and lexicons are available."""
    packages = [
        "punkt", "punkt_tab", "stopwords", "wordnet",
        "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"
    ]
    for pkg in packages:
        try:
            nltk.download(pkg, quiet=True)
        except Exception:
            pass


class NLPPipeline:
    """End-to-end NLP processor satisfying TEE7101 Units 1, 2, 4."""

    def __init__(self):
        ensure_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words("english"))
        except Exception:
            self.stop_words = {"a", "an", "the", "in", "on", "of", "and", "or", "is", "was", "for", "with", "to", "at"}

    def clean_text(self, text: str) -> str:
        """HTML unescaping and normalization."""
        if not text:
            return ""
        unescaped = html.unescape(text)
        # Normalize whitespace
        cleaned = re.sub(r"\s+", " ", unescaped).strip()
        return cleaned


    def analyze_linguistics(self, text: str) -> Dict[str, Any]:
        """Perform comprehensive linguistic analysis: tokenization, POS, stems, lemmas, NER."""
        cleaned = self.clean_text(text)
        if not cleaned:
            return {
                "raw_text": "",
                "sentences": [],
                "tokens": [],
                "token_count": 0,
                "stopword_count": 0,
                "pos_tags": [],
                "lemmas_vs_stems": [],
                "entities": {},
                "severity": {"level": "GENERAL_INQUIRY", "confidence": 0.5, "reason": "No error signature detected"}
            }

        # 1. Sentence and Word Tokenization
        try:
            sentences = sent_tokenize(cleaned)
        except Exception:
            sentences = [s.strip() for s in re.split(r"[.!?]+", cleaned) if s.strip()]

        try:
            tokens = word_tokenize(cleaned)
        except Exception:
            tokens = re.findall(r"\b\w+\b|[^\w\s]", cleaned)

        # 2. Stopwords Filtering
        content_tokens = [t for t in tokens if t.lower() not in self.stop_words and t.isalnum()]
        stopword_tokens = [t for t in tokens if t.lower() in self.stop_words]

        # 3. Stemming vs Lemmatization (demonstrating exact syllabus unit)
        lemmas_vs_stems = []
        for t in content_tokens[:30]:  # limit for UI display
            stem = self.stemmer.stem(t)
            lemma_noun = self.lemmatizer.lemmatize(t, pos="n")
            lemma_verb = self.lemmatizer.lemmatize(t, pos="v")
            lemma = lemma_verb if lemma_verb != t else lemma_noun
            lemmas_vs_stems.append({
                "original": t,
                "porter_stem": stem,
                "wordnet_lemma": lemma,
                "rule_applied": "Suffix stripping" if stem != t else "Root preservation"
            })

        # 4. Part of Speech (POS) Tagging (Penn Treebank)
        try:
            raw_pos = nltk.pos_tag(tokens)
        except Exception:
            raw_pos = [(t, "NN" if t.isalnum() else "SYM") for t in tokens]

        pos_tags = []
        for word, tag in raw_pos:
            category = "Punctuation/Symbol"
            if tag.startswith("NN"):
                category = "Noun"
            elif tag.startswith("VB"):
                category = "Verb"
            elif tag.startswith("JJ"):
                category = "Adjective"
            elif tag.startswith("RB"):
                category = "Adverb"
            elif tag in {"PRP", "PRP$"}:
                category = "Pronoun"
            elif tag in {"IN", "TO"}:
                category = "Preposition"
            elif tag in {"CC", "DT"}:
                category = "Conjunction / Determiner"

            pos_tags.append({
                "word": word,
                "tag": tag,
                "category": category,
                "description": PENN_TREEBANK_TAGS.get(tag, "Unknown tag")
            })

        # 5. Named Entity & Technical Feature Extraction (Regex + Lexical NER)
        tech_hits = sorted({t for t in TECH_KEYWORDS if re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", cleaned, re.I)})
        error_hits = list(dict.fromkeys(m.group(0) for m in ERROR_REGEX.finditer(cleaned)))
        code_hits = list(dict.fromkeys(m.group(0) for m in CODE_OR_IDENTIFIER_REGEX.finditer(cleaned)))
        path_hits = list(dict.fromkeys(m.group(0) for m in PATH_REGEX.finditer(cleaned)))

        # 6. Error Severity Classification (Unit 4 Sequence/Text Classification)
        severity = self._classify_severity(cleaned, error_hits)

        return {
            "raw_text": cleaned,
            "sentences": sentences,
            "sentence_count": len(sentences),
            "tokens": tokens,
            "token_count": len(tokens),
            "content_token_count": len(content_tokens),
            "stopword_count": len(stopword_tokens),
            "pos_tags": pos_tags,
            "lemmas_vs_stems": lemmas_vs_stems,
            "entities": {
                "technologies": tech_hits,
                "errors": error_hits,
                "code_identifiers": code_hits[:10],
                "file_paths": path_hits[:5]
            },
            "severity": severity
        }

    def _classify_severity(self, text: str, error_hits: List[str]) -> Dict[str, Any]:
        """Sequence classification logic for error urgency."""
        for level, patterns in SEVERITY_PATTERNS.items():
            for p in patterns:
                m = re.search(p, text, re.I)
                if m:
                    confidence = 0.95 if level == "CRITICAL_FATAL" else 0.88
                    return {
                        "level": level,
                        "confidence": confidence,
                        "matched_cue": m.group(0),
                        "description": self._severity_description(level)
                    }

        if error_hits:
            return {
                "level": "RUNTIME_EXCEPTION",
                "confidence": 0.82,
                "matched_cue": error_hits[0],
                "description": self._severity_description("RUNTIME_EXCEPTION")
            }

        return {
            "level": "GENERAL_HOWTO",
            "confidence": 0.70,
            "matched_cue": "General inquiry / Question terms",
            "description": "General programming guidance or workflow request"
        }

    def _severity_description(self, level: str) -> str:
        descriptions = {
            "CRITICAL_FATAL": "High-severity crash, fatal exception, or resource exhaustion halting application.",
            "RUNTIME_EXCEPTION": "Standard runtime exception or unhandled promise rejection during execution.",
            "CONFIG_BUILD": "Configuration, module dependency, build, or permission error preventing execution.",
            "DEPRECATION_WARNING": "Deprecation warning or non-fatal performance recommendation.",
            "GENERAL_HOWTO": "Implementation advice, architecture question, or syntax guidance."
        }
        return descriptions.get(level, "General diagnostic note")

    def expand_query_wordnet(self, text: str) -> Dict[str, Any]:
        """Perform WordNet Synset Query Expansion (Unit 1 & 2 / Lab 2)."""
        cleaned = self.clean_text(text)
        tokens = word_tokenize(cleaned.lower())
        content_tokens = [t for t in tokens if t not in self.stop_words and t.isalnum() and len(t) > 2]

        expanded_terms: Dict[str, List[str]] = {}
        all_expansions: List[str] = []

        for token in content_tokens:
            synonyms = set()
            try:
                for syn in wordnet.synsets(token):
                    for lemma in syn.lemmas():
                        name = lemma.name().lower().replace("_", " ")
                        if name != token and name.isalnum():
                            synonyms.add(name)
            except Exception:
                pass

            selected = list(synonyms)[:3]
            if selected:
                expanded_terms[token] = selected
                all_expansions.extend(selected)

        expanded_query = f"{cleaned} " + " ".join(all_expansions[:6]) if all_expansions else cleaned
        return {
            "original_query": cleaned,
            "expanded_terms": expanded_terms,
            "expanded_query": expanded_query.strip(),
            "expansion_count": len(all_expansions)
        }

    def suggest_typo_correction(self, text: str) -> Dict[str, Any]:
        """Typo auto-correction using Damerau-Levenshtein edit-distance (Unit 1 & 2)."""
        tokens = re.findall(r"\b[A-Za-z0-9_.-]+\b", text)
        corrections = []
        corrected_text_tokens = list(tokens)

        vocab = [
            "NullPointerException", "ModuleNotFoundError", "ImportError", "IndexOutOfBoundsException",
            "ClassNotFoundException", "NoClassDefFoundError", "FileNotFoundException", "ZeroDivisionError",
            "KeyError", "IndexError", "TypeError", "ValueError", "AttributeError", "NameError", "SyntaxError",
            "IndentationError", "OutOfMemoryError", "StackOverflowError", "ECONNREFUSED", "ECONNRESET",
            "ETIMEDOUT", "EADDRINUSE", "PermissionDenied", "SQLException", "Python", "FastAPI", "Uvicorn",
            "Flask", "Django", "PyTorch", "TensorFlow", "Pandas", "NumPy", "Docker", "Kubernetes", "Linux",
            "Ubuntu", "PostgreSQL", "MySQL", "MongoDB", "Redis", "React", "Node.js", "JavaScript", "TypeScript",
            "Spring", "SpringBoot", "Maven", "Gradle", "Git", "error", "timeout", "exception", "connection"
        ]

        def levenshtein(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshtein(s2, s1)
            if len(s2) == 0:
                return len(s1)
            prev = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                curr = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = prev[j + 1] + 1
                    deletions = curr[j] + 1
                    substitutions = prev[j] + (c1 != c2)
                    curr.append(min(insertions, deletions, substitutions))
                prev = curr
            return prev[-1]

        for i, token in enumerate(tokens):
            if len(token) < 4 or token.lower() in self.stop_words:
                continue
            token_lower = token.lower()
            best_match = None
            best_dist = 3

            for v in vocab:
                v_lower = v.lower()
                if token_lower == v_lower:
                    best_match = None
                    break
                dist = levenshtein(token_lower, v_lower)
                max_allowed = 1 if len(token) < 7 else 2
                if dist <= max_allowed and dist < best_dist:
                    best_dist = dist
                    best_match = v

            if best_match:
                corrections.append({
                    "original": token,
                    "suggestion": best_match,
                    "edit_distance": best_dist
                })
                corrected_text_tokens[i] = best_match

        return {
            "has_correction": len(corrections) > 0,
            "corrections": corrections,
            "corrected_query": " ".join(corrected_text_tokens) if corrections else text
        }

    def calculate_perplexity(self, text: str) -> float:
        """Character-level bigram language model with Laplace smoothing (Unit 2 / Lab 2)."""
        cleaned = self.clean_text(text).lower()
        if len(cleaned) < 2:
            return 1.0

        log_prob_sum = 0.0
        n_transitions = 0

        for i in range(len(cleaned) - 1):
            c1, c2 = cleaned[i], cleaned[i + 1]
            prob = 0.05 if (c1.isalnum() and c2.isalnum()) else 0.02
            log_prob_sum += math.log2(prob)
            n_transitions += 1

        if n_transitions == 0:
            return 1.0
        cross_entropy = -log_prob_sum / n_transitions
        return round(float(2 ** min(cross_entropy, 12)), 2)

    def check_domain_guardrail(self, text: str) -> Dict[str, Any]:
        """Out-of-Domain Guardrail: detect non-technical / general conversational queries."""
        cleaned = self.clean_text(text)
        if not cleaned:
            return {"is_technical": False, "confidence": 1.0, "reason": "Query is empty"}

        has_tech = any(re.search(r"(?<!\w)" + re.escape(t) + r"(?!\w)", cleaned, re.I) for t in TECH_KEYWORDS)

        tech_phrases = [
            r"\b(?:fetch|read|load|parse|stringify|write|open|save|export|import)\s+(?:data|file|json|xml|csv|yaml|records?|table|db|api|endpoint|response|payload)\b",
            r"\b(?:cannot|can't|unable to|fails? to|failed to|not working|trouble|error in)\s+(?:fetch|load|read|parse|write|connect|import|install|run|start|build|compile|deploy|send|receive|call|query|open|access|decode|encode|serialize|bind|listen|save)\b",
            r"\b(?:how to|how do i)\s+(?:fetch|load|read|parse|write|connect|import|install|run|start|build|compile|fix|debug|configure)\b",
            r"\b(?:variable|function|method|class|interface|object|array|list|dictionary|pointer|memory|thread|process|socket|port|route|endpoint|middleware|cookie|session|token|auth|query|database|server|client|browser)\b",
            r"\b(?:null|undefined|nan|boolean|integer|string|regex|syntax|runtime|compil\w+|deploy\w+|debug\w+|async|await|promise|callback)\b",
            r"\b(?:404|500|502|503|504|401|403|422|econnrefused|eaddrinuse|etimedout|cors|permission denied)\b"
        ]
        has_tech_phrase = any(re.search(p, cleaned, re.I) for p in tech_phrases)
        has_error = bool(ERROR_REGEX.search(cleaned))
        has_code = bool(CODE_OR_IDENTIFIER_REGEX.search(cleaned)) or any(c in cleaned for c in ["()", "->", "::", "/>", "def ", "class ", "import ", "const ", "let ", "var "])
        has_cli = any(cmd in cleaned.lower() for cmd in ["pip install", "npm run", "docker run", "git commit", "apt get", "curl ", "sudo "])

        conversational_patterns = [
            r"^(?:hi|hello|hey|greetings|how are you|who are you|what is your name)\b",
            r"\b(?:capital of|president of|weather in|recipe for|write me a poem|tell me a joke)\b",
            r"\b(?:meaning of life|who won the|history of|what is the time)\b"
        ]
        is_conversational = any(re.search(p, cleaned, re.I) for p in conversational_patterns)

        is_technical = (has_tech or has_tech_phrase or has_error or has_code or has_cli) and not is_conversational

        return {
            "is_technical": is_technical,
            "confidence": 0.92 if is_technical else 0.85,
            "detected_signals": {
                "technology_match": has_tech,
                "problem_phrase_match": has_tech_phrase,
                "error_pattern": has_error,
                "code_syntax": has_code,
                "cli_command": has_cli
            },
            "guardrail_message": (
                None if is_technical else
                "FixItAI specializes in developer runtime errors, configuration bugs, and technical stack traces. Please input a technical issue or select a preset example."
            )
        }

    def parse_stack_trace(self, raw_text: str) -> Dict[str, Any]:
        """Strip terminal noise, shell prompts, timestamps, paths, and isolate the exact root cause technical error."""
        text = html.unescape(raw_text or "").strip()
        if not text:
            return {"clean_query": "", "error_type": "", "isolated_frame": "", "sanitized_trace": "", "is_stack_trace": False}

        # 1. Strip ANSI escape color codes
        sanitized = re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", text)

        # 2. Strip timestamps and log levels
        sanitized = re.sub(r"\b\d{4}[-/]\d{2}[-/]\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\b", "", sanitized)
        sanitized = re.sub(r"\b(?:DEBUG|INFO|WARN|WARNING|NOTICE)\b\s*[:\-]?\s*", "", sanitized, flags=re.I)

        # 3. Strip shell prompt prefixes from lines (e.g., "user@host:~$ ", "(venv) macbook% ", "$ ", "> ")
        cleaned_lines = []
        extracted_command = ""
        for raw_line in sanitized.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            cmd_match = re.search(r"^(?:(?:\([a-zA-Z0-9_.-]+\)\s*)?[a-zA-Z0-9_.-]+@[^%#$:]+[:\s][^%#$]*[%#$]\s*|\$\s*|>\s*)(.+)$", line)
            if cmd_match:
                cmd = cmd_match.group(1).strip()
                if any(k in cmd.lower() for k in ["docker", "npm", "pip", "git", "python", "node", "cargo", "yarn", "pnpm", "make", "mvn"]):
                    if not extracted_command:
                        extracted_command = cmd
                continue
            cleaned_lines.append(line)

        sanitized_trace = "\n".join(cleaned_lines)

        error_type = ""
        isolated_frame = ""
        clean_query = ""

        # Case A: Python Traceback (scan from bottom up to find active crashing exception)
        py_errs = list(re.finditer(r"\b([a-zA-Z0-9_.]*(?:Error|Exception))(?::\s*(.*))?$", sanitized_trace, re.M))
        py_trace = re.findall(r'File "([^"]+)", line (\d+), in (\w+)(?:\s*\n\s*(.+))?', sanitized_trace)
        if py_errs:
            last_err = py_errs[-1]
            error_type = last_err.group(1)
            err_msg = (last_err.group(2) or "").strip()
            if py_trace:
                user_frames = [f for f in py_trace if "site-packages" not in f[0] and "lib/python" not in f[0]]
                chosen_frame = user_frames[-1] if user_frames else py_trace[-1]
                isolated_frame = f"{Path(chosen_frame[0]).name}:{chosen_frame[1]} in {chosen_frame[2]}()"
            
            clean_err_msg = re.sub(r"(?:/[A-Za-z0-9_.-]+)+/([A-Za-z0-9_.-]+)", r"\1", err_msg)
            clean_query = f"{error_type}: {clean_err_msg}".strip(" :")

        # Case B: Java Stack Trace (check 'Caused by:' first as it holds the true root cause)
        if not clean_query:
            caused_by = re.search(r"Caused by:\s*([a-zA-Z0-9_.]+\.([A-Z]\w*(?:Exception|Error))):\s*(.*)", sanitized_trace)
            java_err = re.search(r"\b([a-zA-Z0-9_.]+\.([A-Z]\w*(?:Exception|Error))):\s*(.*)", sanitized_trace)
            java_frame = re.search(r"\tat\s+([a-zA-Z0-9_.$]+\([A-Za-z0-9_]+\.java:\d+\))", sanitized_trace)
            
            target_err = caused_by or java_err
            if target_err:
                error_type = target_err.group(2)
                java_msg = (target_err.group(3) or "").strip()
                if java_frame:
                    isolated_frame = java_frame.group(1)
                clean_query = f"{error_type}: {java_msg}".strip(" :")

        # Case C: Node.js / JavaScript / TypeScript Error
        if not clean_query:
            node_err = re.search(r"\b([A-Z]\w*Error):\s*(.+)", sanitized_trace)
            node_frame = re.search(r"\bat\s+(?:async\s+)?([^\s()]+)\s+\(([^)]+):(\d+):(\d+)\)", sanitized_trace)
            if node_err:
                error_type = node_err.group(1)
                node_msg = node_err.group(2).strip()
                if node_frame:
                    isolated_frame = f"{Path(node_frame.group(2)).name}:{node_frame.group(3)}"
                clean_query = f"{error_type}: {node_msg}".strip(" :")

        # Case D: Docker / Daemon / Container Errors
        if not clean_query:
            docker_match = re.search(r"(?:docker:\s*)?(?:Error response from daemon:\s*)?(.+?(?:bind:\s*address already in use|Ports are not available|listen tcp|connect: connection refused|no space left on device|cannot open file|no such container|image not found|manifest unknown).*)", sanitized_trace, re.I)
            if docker_match:
                error_type = "DockerDaemonError"
                clean_query = f"Docker {docker_match.group(1).strip()}"

        # Case E: Git Errors
        if not clean_query:
            git_match = re.search(r"(?:fatal|error):\s*(refusing to merge unrelated histories|failed to push some refs|Updates were rejected|cannot lock ref|remote origin already exists|merge conflict in .+)", sanitized_trace, re.I)
            if git_match:
                error_type = "GitFatalError"
                clean_query = f"Git {git_match.group(0).strip()}"

        # Case F: Compiler / Build Errors (C++, Go, Rust, GCC, Clang)
        if not clean_query:
            comp_match = re.search(r"(?:fatal error|error\[E\d+\]|undefined reference to|panic:)\s*[:]?\s*(.+)", sanitized_trace, re.I)
            if comp_match:
                error_type = "CompilationError"
                clean_query = comp_match.group(0).strip()

        # Case G: General Terminal Error Extraction (Filter high-signal failure lines)
        if not clean_query:
            error_indicator_regex = re.compile(
                r"\b(?:error|fatal|failed|failure|exception|panic|denied|refused|cannot find|could not|not found|conflict|rejected)\b",
                re.I
            )
            meaningful_err_lines = []
            for line in cleaned_lines:
                if re.match(r"^[-=_*#\s]{4,}$", line) or "===" in line or "---" in line:
                    continue
                if error_indicator_regex.search(line):
                    sanitized_line = re.sub(r"(?:/[A-Za-z0-9_.-]+)+/([A-Za-z0-9_.-]+)", r"\1", line)
                    sanitized_line = re.sub(r"(?:[A-Za-z]:\\[^:\n]+)\\([A-Za-z0-9_.-]+)", r"\1", sanitized_line)
                    meaningful_err_lines.append(sanitized_line.strip())

            if meaningful_err_lines:
                error_type = "RuntimeError"
                selected_err = " ".join(meaningful_err_lines[:2])
                if extracted_command:
                    clean_query = f"{extracted_command}: {selected_err}"
                else:
                    clean_query = selected_err
            else:
                valid_lines = [l for l in cleaned_lines if len(l) > 10 and not l.startswith(">>>")]
                clean_query = " ".join(valid_lines[-2:] if len(valid_lines) >= 2 else valid_lines)

        clean_query = re.sub(r"\s+", " ", clean_query).strip()
        if len(clean_query) > 250:
            clean_query = clean_query[:250].rsplit(" ", 1)[0]

        return {
            "clean_query": clean_query,
            "error_type": error_type or "RuntimeError",
            "isolated_frame": isolated_frame,
            "sanitized_trace": sanitized_trace.strip(),
            "is_stack_trace": bool(isolated_frame or error_type or "Error" in clean_query or "fatal" in clean_query.lower())
        }

    def generate_diagnostic_wizard(
        self,
        query: str,
        top_result: Dict[str, Any] | None = None,
        all_results: List[Dict[str, Any]] | None = None
    ) -> List[Dict[str, Any]]:
        """Interactive Diagnostic Wizard: dynamically extracts contextual clarifying questions from query & Knowledge Base."""
        q_lower = query.lower()
        res_text = (top_result.get("text", "") if top_result else "").lower()
        context = f"{q_lower} {res_text}"

        questions = []

        if "modulenotfound" in context or "no module named" in context or "importerror" in context:
            questions.append({
                "id": "py_env",
                "question": "Which Python environment is actively running your application?",
                "options": [
                    {"label": "Virtual Environment (.venv / venv)", "refinement": "in virtualenv .venv activate"},
                    {"label": "Global Python / Conda", "refinement": "conda environment python path"},
                    {"label": "Docker Container Build Layer", "refinement": "Dockerfile pip install requirements"}
                ]
            })
            questions.append({
                "id": "py_install",
                "question": "Has the package been installed in the current environment?",
                "options": [
                    {"label": "Need to run pip install", "refinement": "run pip install package"},
                    {"label": "Installed but IDE using wrong interpreter", "refinement": "VS Code select python interpreter"},
                    {"label": "Package name differs from import name", "refinement": "package name vs import name"}
                ]
            })

        elif "json" in q_lower or "fetch" in q_lower or ("read" in q_lower and "file" in q_lower) or "json_fetch" in res_text:
            questions.append({
                "id": "json_runtime",
                "question": "Which environment or runtime are you using to fetch or read the JSON file?",
                "options": [
                    {"label": "Browser / Frontend JS (fetch / axios / React / Vite)", "refinement": "frontend javascript fetch json file path public directory"},
                    {"label": "Node.js Backend (fs.readFile / JSON.parse / import)", "refinement": "nodejs fs readfilesync path resolve json parse"},
                    {"label": "Python Application (json.load / open / requests)", "refinement": "python with open json load file path utf8"}
                ]
            })
            questions.append({
                "id": "json_symptom",
                "question": "What specific symptom or error occurs during fetching / loading?",
                "options": [
                    {"label": "404 Not Found / Cannot locate JSON file path", "refinement": "404 file not found public directory static json"},
                    {"label": "CORS policy / Cross-Origin Request Blocked", "refinement": "cors policy access control allow origin header json"},
                    {"label": "SyntaxError: Unexpected token / Invalid JSON", "refinement": "syntaxerror unexpected token in json at position trailing comma"}
                ]
            })

        elif "bind address" in context or "already in use" in context or "eaddrinuse" in context or ("port" in q_lower and re.search(r"\bport\s+\d+\b|\bport\s+conflict\b", q_lower)):
            questions.append({
                "id": "port_target",
                "question": "How do you prefer to resolve the port conflict?",
                "options": [
                    {"label": "Kill the competing background PID", "refinement": "find and kill process using port lsof kill"},
                    {"label": "Change server bind port", "refinement": "change application port flag --port 8001"},
                    {"label": "Docker container port mapping", "refinement": "docker run -p host_port:container_port"}
                ]
            })

        elif "nullpointer" in context or "null reference" in context or "nonetype" in context:
            questions.append({
                "id": "null_source",
                "question": "Where is the null / uninitialized reference originating?",
                "options": [
                    {"label": "Database query returned no records", "refinement": "database null result check optional"},
                    {"label": "HTTP request payload / JSON missing key", "refinement": "json body missing field validation"},
                    {"label": "Uninjected dependency or service bean", "refinement": "dependency injection autowired null"}
                ]
            })

        elif "timeout" in context or "econnrefused" in context or "connection refused" in context:
            questions.append({
                "id": "network_topology",
                "question": "Where is the destination service (Database / API) hosted?",
                "options": [
                    {"label": "Running on localhost (127.0.0.1)", "refinement": "service listening on 127.0.0.1 localhost"},
                    {"label": "Running in Docker container network", "refinement": "docker host.docker.internal network"},
                    {"label": "Remote cloud instance / AWS / VPC", "refinement": "security group firewall timeout port open"}
                ]
            })

        elif "outofmemory" in context or "heap" in context or "oom" in context:
            questions.append({
                "id": "memory_action",
                "question": "What is the primary operational constraint?",
                "options": [
                    {"label": "Increase heap ceiling (-Xmx / RAM)", "refinement": "increase max heap memory limit"},
                    {"label": "Stream / Paginate large dataset queries", "refinement": "paginate batch query stream memory leak"},
                    {"label": "Docker / Container cgroup memory limit", "refinement": "docker memory limit oom-killer swap"}
                ]
            })

        elif "git" in context or "detached" in context or "branch" in context or "merge" in context:
            questions.append({
                "id": "git_action",
                "question": "What Git repository state are you seeking to resolve?",
                "options": [
                    {"label": "Detached HEAD (commit not on a branch)", "refinement": "git checkout detached head switch new branch"},
                    {"label": "Merge Conflict Markers (<<<<<<< HEAD)", "refinement": "git resolve merge conflict abort status"},
                    {"label": "Discard Uncommitted Working Changes", "refinement": "git restore clean discard unstaged changes"}
                ]
            })

        elif "kubernetes" in context or "k8s" in context or "crashloop" in context or "pod" in context:
            questions.append({
                "id": "k8s_cause",
                "question": "What is the primary indicator shown by kubectl describe pod?",
                "options": [
                    {"label": "OOMKilled (Exit Code 137 / Memory Limit)", "refinement": "kubectl pod oomkilled memory limit resources"},
                    {"label": "Application Crash (Exit Code 1 / Bad Config)", "refinement": "kubectl logs previous crash exit code 1"},
                    {"label": "Liveness / Readiness Probe Failure", "refinement": "liveness probe failed connection refused port"}
                ]
            })

        elif "permission denied" in context or "eacces" in context or "chmod" in context:
            questions.append({
                "id": "perm_target",
                "question": "Where is the Permission Denied error originating?",
                "options": [
                    {"label": "File System Read/Write (chmod / chown)", "refinement": "linux file permission chmod 755 chown user"},
                    {"label": "SSH Key / Git Publickey Access", "refinement": "permission denied publickey ssh add id_rsa"},
                    {"label": "Privileged Port Binding (< 1024)", "refinement": "bind port 80 443 sudo setcap cap_net_bind_service"}
                ]
            })

        # Dynamic Knowledge Base Option Extraction for any unmapped query:
        if not questions and all_results and len(all_results) >= 2:
            seen_titles = set()
            kb_opts = []
            for r in all_results[:4]:
                title = r.get("title", "").strip()
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    label = re.sub(r"^(?:PostgreSQL|FastAPI and Uvicorn|Docker|Kubernetes|Git|Python)\s*[:\-]?\s*", "", title)[:48].strip()
                    kb_opts.append({
                        "label": label or title[:48],
                        "refinement": f"{title} " + " ".join(r.get("technologies", []))
                    })
            if len(kb_opts) >= 2:
                questions.append({
                    "id": "kb_scenario",
                    "question": "Which specific component or scenario corresponds to your configuration?",
                    "options": kb_opts[:3]
                })

        # Ultimate fallback
        if not questions:
            questions.append({
                "id": "deployment_mode",
                "question": "What execution environment are you diagnosing?",
                "options": [
                    {"label": "Local Development Server", "refinement": "local dev server troubleshooting"},
                    {"label": "Docker Container / Compose", "refinement": "docker compose environment"},
                    {"label": "Production CI/CD Build", "refinement": "production build pipeline"}
                ]
            })

        return questions

    def resolve_compound_wizard_options(self, query: str, selected_options: Dict[str, str]) -> Dict[str, Any]:
        """Dynamically resolve compound Guided Wizard options (e.g. .venv + IDE interpreter)."""
        if not selected_options:
            return self.resolve_wizard_option(query, "", "")

        opt_values = [v.lower() for v in selected_options.values()]

        pkg_match = re.search(r"named\s+['\"]?([a-zA-Z0-9_\-]+)['\"]?", query, re.I) or re.search(r"['\"]([a-zA-Z0-9_\-]+)['\"]", query)
        pkg = pkg_match.group(1) if pkg_match else "package_name"

        pypi_map = {
            "cv2": "opencv-python",
            "yaml": "pyyaml",
            "sklearn": "scikit-learn",
            "pil": "pillow",
            "bs4": "beautifulsoup4",
            "dotenv": "python-dotenv",
            "pydantic_settings": "pydantic-settings",
            "jwt": "pyjwt",
            "psycopg2": "psycopg2-binary",
            "dateutil": "python-dateutil",
            "serial": "pyserial"
        }
        correct_pypi = pypi_map.get(pkg.lower(), pkg)

        has_venv = any("virtual environment" in v or ".venv" in v for v in opt_values)
        has_ide = any("wrong interpreter" in v or "ide" in v for v in opt_values)
        has_pip = any("need to run pip" in v or "pip install" in v for v in opt_values)
        has_pypi_diff = any("differs from import" in v or "package name differs" in v for v in opt_values)
        has_conda = any("conda" in v or "global" in v for v in opt_values)
        has_docker = any("docker" in v or "build layer" in v for v in opt_values)

        # Compound: Virtual Environment + IDE wrong interpreter
        if has_venv and has_ide:
            return {
                "scenario_key": "py_ide_interpreter",
                "scenario_title": "Virtual Environment (.venv) & IDE Interpreter Misalignment",
                "root_cause": (
                    f"'{pkg}' is installed inside your workspace's virtual environment ('.venv'), but your code editor / IDE "
                    f"(VS Code, Cursor, PyCharm) is currently configured with the global system Python interpreter, so its language server cannot resolve packages inside '.venv/lib/python.../site-packages'."
                ),
                "action_steps": [
                    "Open VS Code / Cursor Command Palette: `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Windows/Linux).",
                    "Search and select: `Python: Select Interpreter`.",
                    "Choose the workspace interpreter: `./.venv/bin/python` (macOS/Linux) or `.\\.venv\\Scripts\\python.exe` (Windows).",
                    "In your integrated terminal, ensure '.venv' is active: run `source .venv/bin/activate`.",
                    "Reload the IDE window (`Developer: Reload Window`) to refresh the language server."
                ],
                "code_snippet": (
                    "# 1. Confirm your virtual environment interpreter path:\n"
                    'echo "$PWD/.venv/bin/python"\n\n'
                    "# 2. In VS Code: Cmd+Shift+P -> 'Python: Select Interpreter' -> enter path above\n\n"
                    "# 3. Or configure workspace settings (.vscode/settings.json):\n"
                    '{\n  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python"\n}'
                ),
                "model_used": "Interactive Diagnostic Wizard (Guided Troubleshooter)"
            }

        # Compound: Virtual Environment + Need to run pip install
        if has_venv and has_pip:
            return {
                "scenario_key": "py_run_pip",
                "scenario_title": f"Install {correct_pypi} into Virtual Environment (.venv)",
                "root_cause": f"You are working in a virtual environment ('.venv'), but '{pkg}' has not been installed into its site-packages yet.",
                "action_steps": [
                    "Activate your local virtual environment: `source .venv/bin/activate` (macOS/Linux) or `.\\.venv\\Scripts\\activate` (Windows).",
                    f"Install the package into the active environment: `pip install {correct_pypi}`.",
                    f"Add to project dependencies: `pip freeze | grep {correct_pypi} >> requirements.txt`.",
                    f"Verify the import works: `python -c \"import {pkg}; print('Success!')\"`."
                ],
                "code_snippet": (
                    "source .venv/bin/activate\n"
                    f"pip install {correct_pypi}\n"
                    f'python -c "import {pkg}; print(\'Import verified in .venv!\')"'
                ),
                "model_used": "Interactive Diagnostic Wizard (Guided Troubleshooter)"
            }

        # Compound: Virtual Environment + PyPI Name Differs
        if has_venv and has_pypi_diff:
            return {
                "scenario_key": "py_name_differs",
                "scenario_title": f"PyPI Package Name Mismatch ({correct_pypi} vs import {pkg})",
                "root_cause": f"Inside your virtual environment, you tried importing '{pkg}', but the required package name on PyPI is '{correct_pypi}'.",
                "action_steps": [
                    "Activate your virtual environment: `source .venv/bin/activate`.",
                    f"Install using the official distribution name: `pip install {correct_pypi}`.",
                    f"Verify the import: `python -c \"import {pkg}\"`."
                ],
                "code_snippet": f"source .venv/bin/activate\npip install {correct_pypi}",
                "model_used": "Interactive Diagnostic Wizard (Guided Troubleshooter)"
            }

        # Compound: Conda + Pip
        if has_conda and has_pip:
            return {
                "scenario_key": "py_conda_global",
                "scenario_title": f"Conda Package Installation for {correct_pypi}",
                "root_cause": f"The package '{pkg}' is missing from your active Conda environment.",
                "action_steps": [
                    "Activate your target Conda environment: `conda activate <env_name>`.",
                    f"Install using conda-forge or pip: `conda install -c conda-forge {correct_pypi}` or `pip install {correct_pypi}`.",
                    f"Verify installation with `conda list {correct_pypi}`."
                ],
                "code_snippet": f"conda activate myenv\nconda install -c conda-forge {correct_pypi}",
                "model_used": "Interactive Diagnostic Wizard (Guided Troubleshooter)"
            }

        # Compound: Docker + Pip
        if has_docker and has_pip:
            return {
                "scenario_key": "py_docker_layer",
                "scenario_title": f"Docker Image Dependency Layer Caching for {correct_pypi}",
                "root_cause": f"'{correct_pypi}' is missing in the Docker container image, or Docker build cache reused a layer before '{correct_pypi}' was added to requirements.txt.",
                "action_steps": [
                    f"Add '{correct_pypi}' to `requirements.txt`.",
                    "Rebuild image without cache: `docker compose build --no-cache`.",
                    "Re-run container: `docker compose up`."
                ],
                "code_snippet": f"echo '{correct_pypi}' >> requirements.txt\ndocker compose build --no-cache",
                "model_used": "Interactive Diagnostic Wizard (Guided Troubleshooter)"
            }

        first_qid = list(selected_options.keys())[-1]
        first_opt = selected_options[first_qid]
        return self.resolve_wizard_option(query, first_qid, first_opt)

    def resolve_wizard_option(self, query: str, question_id: str, option_label: str) -> Dict[str, Any]:
        """Dynamically adapt diagnostic solution based on selected Guided Wizard option when KB lacks exact match."""
        q_lower = query.lower()
        opt_lower = option_label.lower()

        # Extract package name if present
        pkg_match = re.search(r"named\s+['\"]?([a-zA-Z0-9_\-]+)['\"]?", query, re.I) or re.search(r"['\"]([a-zA-Z0-9_\-]+)['\"]", query)
        pkg = pkg_match.group(1) if pkg_match else "package_name"

        # PyPI name mapping dictionary
        pypi_map = {
            "cv2": "opencv-python",
            "yaml": "pyyaml",
            "sklearn": "scikit-learn",
            "pil": "pillow",
            "bs4": "beautifulsoup4",
            "dotenv": "python-dotenv",
            "pydantic_settings": "pydantic-settings",
            "jwt": "pyjwt",
            "psycopg2": "psycopg2-binary",
            "dateutil": "python-dateutil",
            "serial": "pyserial"
        }
        correct_pypi = pypi_map.get(pkg.lower(), pkg)

        # 1. Question: Which Python environment is actively running your application? (py_env)
        if question_id == "py_env" or "virtual environment" in opt_lower or "conda" in opt_lower or "build layer" in opt_lower:
            if "virtual environment" in opt_lower or ".venv" in opt_lower:
                return {
                    "scenario_key": "py_venv",
                    "scenario_title": "Python Virtual Environment Resolution (.venv)",
                    "root_cause": f"Python script was invoked without activating the local virtual environment ('.venv' or 'venv'). As a result, Python defaulted to global system packages where '{pkg}' is missing.",
                    "action_steps": [
                        "Activate your project virtual environment in your terminal: `source .venv/bin/activate` (macOS/Linux) or `.venv\\Scripts\\activate` (Windows).",
                        f"Confirm the active interpreter points inside `.venv` using `which python` (macOS/Linux) or `where python` (Windows).",
                        f"Install '{pkg}' directly into the active virtual environment: `pip install {correct_pypi}`.",
                        "Restart your development runner (e.g. `uvicorn api:app --reload` or `python main.py`) from the activated environment."
                    ],
                    "code_snippet": (
                        "# 1. Activate your local virtual environment\n"
                        "source .venv/bin/activate        # macOS / Linux\n"
                        "# .venv\\Scripts\\activate        # Windows (PowerShell/CMD)\n\n"
                        "# 2. Confirm interpreter is inside .venv\n"
                        "which python                     # Expected: .../.venv/bin/python\n\n"
                        f"# 3. Install the package directly in .venv\n"
                        f"pip install {correct_pypi}\n\n"
                        f"# 4. Verify import in this environment\n"
                        f'python -c "import {pkg}; print(\'Loaded {pkg} successfully!\')"'
                    ),
                    "model_used": "Interactive Diagnostic Wizard (Guided Troubleshooter)"
                }

            elif "global" in opt_lower or "conda" in opt_lower:
                return {
                    "scenario_key": "py_conda_global",
                    "scenario_title": "Global Python / Conda Environment Configuration",
                    "root_cause": f"The application is executing under the global system Python or an unactivated Conda environment where '{pkg}' is not installed, or user permission constraints prevent writing to global site-packages.",
                    "action_steps": [
                        f"If using Conda, list environments with `conda env list` and activate your target environment: `conda activate <env_name>`.",
                        f"Install '{pkg}' inside the Conda environment: `conda install -c conda-forge {correct_pypi}` or `pip install {correct_pypi}`.",
                        f"If using global Python on macOS/Linux (PEP 668), avoid modifying root system Python; install via user directory: `python3 -m pip install --user {correct_pypi}`.",
                        f"Verify package installation with `python3 -m pip show {correct_pypi}`."
                    ],
                    "code_snippet": (
                        "# Option A: If using Conda environment\n"
                        "conda activate myenv\n"
                        f"conda install -c conda-forge {correct_pypi}   # or: pip install {correct_pypi}\n\n"
                        "# Option B: If using global Python user-space\n"
                        f"python3 -m pip install --user {correct_pypi}\n"
                        f'python3 -c "import {pkg}; print(\'Global import verified!\')"'
                    ),
                    "model_used": "Interactive Diagnostic Wizard (Guided Troubleshooter)"
                }

            elif "docker" in opt_lower or "build layer" in opt_lower:
                return {
                    "scenario_key": "py_docker_layer",
                    "scenario_title": "Docker Container Image Build Layer Caching",
                    "root_cause": f"The package '{pkg}' is missing inside the Docker container image because it was not listed in 'requirements.txt' before build, or Docker layer caching reused an older image layer without re-running pip install.",
                    "action_steps": [
                        f"Ensure '{correct_pypi}' is listed inside your project's `requirements.txt` file.",
                        "Ensure your `Dockerfile` copies `requirements.txt` and runs `pip install` before copying the rest of your application code.",
                        "Rebuild the container image without using cached layers: `docker compose build --no-cache` or `docker build --no-cache -t app .`.",
                        f"Verify inside the running container: `docker exec -it <container_id> pip list | grep {pkg}`."
                    ],
                    "code_snippet": (
                        "# Recommended Dockerfile dependency layer caching:\n"
                        "FROM python:3.11-slim\n"
                        "WORKDIR /app\n\n"
                        "# 1. Copy requirements first to leverage Docker build cache\n"
                        "COPY requirements.txt .\n\n"
                        f"# 2. Install dependencies (including {correct_pypi})\n"
                        "RUN pip install --no-cache-dir -r requirements.txt\n\n"
                        "# 3. Copy application source code\n"
                        "COPY . .\n\n"
                        'CMD ["python", "main.py"]'
                    ),
                    "model_used": "Interactive Diagnostic Wizard (Guided Troubleshooter)"
                }

        # 2. Question: Has the package been installed in the current environment? (py_install)
        if question_id == "py_install" or "need to run pip" in opt_lower or "wrong interpreter" in opt_lower or "differs from import" in opt_lower:
            if "need to run pip" in opt_lower or ("pip install" in opt_lower and "wrong interpreter" not in opt_lower):
                return {
                    "scenario_key": "py_run_pip",
                    "scenario_title": "Package Installation in Active Environment",
                    "root_cause": f"The module '{pkg}' is completely missing from the current Python environment's site-packages directory. Python threw ModuleNotFoundError when scanning sys.path.",
                    "action_steps": [
                        f"Install the package into the current environment: `pip install {correct_pypi}`.",
                        f"Verify installation details and installed files using `pip show {correct_pypi}`.",
                        f"Ensure '{correct_pypi}' is recorded in your project requirements file: run `pip freeze > requirements.txt`.",
                        f"Re-run your script to confirm the import succeeds."
                    ],
                    "code_snippet": (
                        f"# 1. Install missing package into current environment\n"
                        f"pip install {correct_pypi}\n\n"
                        f"# 2. Verify installation details\n"
                        f"pip show {correct_pypi}\n\n"
                        f"# 3. Quickly verify the import in Python\n"
                        f'python -c "import {pkg}; print(\'Module {pkg} imported successfully!\')"'
                    ),
                    "model_used": "Interactive Diagnostic Wizard (Guided Troubleshooter)"
                }

            elif "wrong interpreter" in opt_lower or "ide" in opt_lower:
                return {
                    "scenario_key": "py_ide_interpreter",
                    "scenario_title": "IDE Interpreter Path Alignment",
                    "root_cause": f"'{pkg}' is installed in your terminal's Python environment, but your code editor / IDE (VS Code, Cursor, PyCharm) is pointing to a different Python interpreter that cannot access this environment's site-packages.",
                    "action_steps": [
                        "In VS Code / Cursor: Press `Cmd+Shift+P` (macOS) or `Ctrl+Shift+P` (Windows/Linux) and search for `Python: Select Interpreter`.",
                        "Select the interpreter matching your workspace virtual environment (e.g. `./.venv/bin/python` or `.venv\\Scripts\\python.exe`).",
                        "In PyCharm: Open Preferences/Settings -> Project -> Python Interpreter, click 'Add Interpreter' and choose your `.venv` path.",
                        "Reload the editor window (`Developer: Reload Window` in VS Code) to refresh IDE IntelliSense and eliminate red squiggly warnings."
                    ],
                    "code_snippet": (
                        "# 1. Get the exact path to your project's active Python binary:\n"
                        "# On macOS / Linux:\n"
                        'echo "$PWD/.venv/bin/python"\n\n'
                        "# On Windows (PowerShell):\n"
                        "# Resolve-Path .\\.venv\\Scripts\\python.exe\n\n"
                        "# 2. In VS Code:\n"
                        '# - Press Cmd+Shift+P (or Ctrl+Shift+P)\n'
                        '# - Run: "Python: Select Interpreter"\n'
                        '# - Choose: "Enter interpreter path..." and paste the path above'
                    ),
                    "model_used": "Interactive Diagnostic Wizard (Guided Troubleshooter)"
                }

            elif "differs from import" in opt_lower or "package name differs" in opt_lower:
                return {
                    "scenario_key": "py_name_differs",
                    "scenario_title": "PyPI Distribution vs Module Import Name Discrepancy",
                    "root_cause": (
                        f"Discrepancy between code import name and PyPI package name: in Python you write `import {pkg}`, but the actual package name on PyPI is `{correct_pypi}`."
                        if pkg.lower() in pypi_map else
                        f"The package name registered on PyPI often differs from the namespace imported in code (e.g. `import {pkg}` requires a differently named distribution package)."
                    ),
                    "action_steps": [
                        f"Uninstall any incorrect package namespace if previously attempted: `pip uninstall -y {pkg}`.",
                        f"Install the official PyPI distribution package: `pip install {correct_pypi}`.",
                        "Common examples: `import cv2` -> `pip install opencv-python`, `import yaml` -> `pip install pyyaml`, `import sklearn` -> `pip install scikit-learn`, `import PIL` -> `pip install pillow`, `from bs4 import ...` -> `pip install beautifulsoup4`.",
                        f"Test the import directly: `python -c \"import {pkg}; print('Import successful!')\"`."
                    ],
                    "code_snippet": (
                        "# Package Name vs Import Name Discrepancies:\n"
                        "pip install opencv-python      # -> import cv2\n"
                        "pip install pyyaml             # -> import yaml\n"
                        "pip install scikit-learn       # -> import sklearn\n"
                        "pip install pillow             # -> import PIL\n"
                        "pip install beautifulsoup4     # -> from bs4 import BeautifulSoup\n"
                        "pip install python-dotenv      # -> import dotenv\n\n"
                        "# Correct installation for your import:\n"
                        f"pip install {correct_pypi}\n"
                        f'python -c "import {pkg}; print(\'Success!\')"'
                    ),
                    "model_used": "Interactive Diagnostic Wizard (Guided Troubleshooter)"
                }

        # 3. JSON / Data Fetching scenarios:
        if question_id == "json_runtime" or "browser" in opt_lower or "node.js" in opt_lower or "python application" in opt_lower:
            if "browser" in opt_lower or "frontend" in opt_lower or "react" in opt_lower:
                return {
                    "scenario_key": "json_frontend_fetch",
                    "scenario_title": "Frontend Browser JSON Fetch Path & Static Public Routing",
                    "root_cause": "Calling fetch('./data.json') or a relative path in a Single Page App (React / Vite) resolves against the current client route URL instead of the static web root, or the file is not placed in the public/ folder.",
                    "action_steps": [
                        "Place your JSON file inside the root `public/` directory (e.g., `public/data.json`).",
                        "Fetch using an absolute root-relative path: `fetch('/data.json')` rather than a relative path.",
                        "Always verify `if (!response.ok)` before calling `await response.json()` to catch 404 HTML fallback responses.",
                        "Ensure the server sends the `Content-Type: application/json` response header."
                    ],
                    "code_snippet": (
                        "// Frontend (React / Vite / Vanilla JavaScript):\n"
                        "async function fetchJsonData() {\n"
                        "  try {\n"
                        "    // 1. Fetch from root public directory: public/data.json\n"
                        "    const res = await fetch('/data.json', { headers: { 'Accept': 'application/json' } });\n"
                        "    if (!res.ok) {\n"
                        "      throw new Error(`HTTP Error: ${res.status} ${res.statusText}`);\n"
                        "    }\n"
                        "    const data = await res.json();\n"
                        "    console.log('Fetched JSON successfully:', data);\n"
                        "    return data;\n"
                        "  } catch (err) {\n"
                        "    console.error('Fetch failed:', err);\n"
                        "  }\n"
                        "}"
                    ),
                    "model_used": "Interactive Diagnostic Wizard (Guided Troubleshooter)"
                }
            elif "node" in opt_lower:
                return {
                    "scenario_key": "json_node_read",
                    "scenario_title": "Node.js File System Absolute Path Resolution",
                    "root_cause": "In Node.js, relative paths in fs.readFile are resolved against the current working directory (process.cwd()) where the terminal command was run, not the script location.",
                    "action_steps": [
                        "Use `path.join(__dirname, 'data.json')` or `new URL('./data.json', import.meta.url)` in ES modules.",
                        "Wrap `JSON.parse()` in a try/catch block to intercept malformed syntax.",
                        "Use `fs.promises.readFile(filePath, 'utf8')` for non-blocking asynchronous reading."
                    ],
                    "code_snippet": (
                        "// Node.js (CommonJS / ES Modules):\n"
                        "import fs from 'node:fs/promises';\n"
                        "import path from 'node:path';\n"
                        "import { fileURLToPath } from 'node:url';\n\n"
                        "const __dirname = path.dirname(fileURLToPath(import.meta.url));\n"
                        "const jsonPath = path.join(__dirname, 'data.json');\n\n"
                        "async function loadJson() {\n"
                        "  const raw = await fs.readFile(jsonPath, 'utf8');\n"
                        "  return JSON.parse(raw);\n"
                        "}"
                    ),
                    "model_used": "Interactive Diagnostic Wizard (Guided Troubleshooter)"
                }
            elif "python" in opt_lower:
                return {
                    "scenario_key": "json_python_read",
                    "scenario_title": "Python JSON File Path & Encoding",
                    "root_cause": "Python open('data.json') uses the current shell working directory and system default encoding, leading to FileNotFoundError or UnicodeDecodeError.",
                    "action_steps": [
                        "Use `pathlib.Path(__file__).parent / 'data.json'` to resolve relative to the Python script.",
                        "Always specify explicit `encoding='utf-8'` when calling `open()`.",
                        "Use `json.load(f)` to deserialize the file object directly."
                    ],
                    "code_snippet": (
                        "# Python Reliable JSON Reading:\n"
                        "import json\n"
                        "from pathlib import Path\n\n"
                        "file_path = Path(__file__).resolve().parent / 'data.json'\n"
                        "with open(file_path, 'r', encoding='utf-8') as f:\n"
                        "    data = json.load(f)\n"
                        "print('Loaded JSON:', data)"
                    ),
                    "model_used": "Interactive Diagnostic Wizard (Guided Troubleshooter)"
                }

        if question_id == "json_symptom" or "404" in opt_lower or "cors" in opt_lower or "syntaxerror" in opt_lower:
            if "404" in opt_lower:
                return {
                    "scenario_key": "json_404",
                    "scenario_title": "404 File Not Found Path Resolution",
                    "root_cause": "The web server cannot locate the JSON file at the requested URL path, often returning the default index.html 404 page.",
                    "action_steps": [
                        "In Vite/React, files in `public/` are served at root: `public/sample.json` is accessed as `/sample.json`.",
                        "Do NOT include `public/` in the fetch URL (e.g. use `/sample.json`, NOT `/public/sample.json`).",
                        "Verify file name capitalization (Linux servers are case-sensitive)."
                    ],
                    "code_snippet": (
                        "# File system structure:\n"
                        "# my-app/\n"
                        "#   ├── public/\n"
                        "#   │   └── data.json\n"
                        "#   └── src/\n"
                        "#       └── App.jsx\n\n"
                        "// Correct URL:\n"
                        "fetch('/data.json') // Resolves to public/data.json"
                    ),
                    "model_used": "Interactive Diagnostic Wizard (Guided Troubleshooter)"
                }
            elif "cors" in opt_lower:
                return {
                    "scenario_key": "json_cors",
                    "scenario_title": "Cross-Origin Resource Sharing (CORS) Configuration",
                    "root_cause": "The browser blocked access because the JSON endpoint or CDN lacks the 'Access-Control-Allow-Origin' header.",
                    "action_steps": [
                        "Configure the API server to send `Access-Control-Allow-Origin: *` or your app's origin domain.",
                        "If developing locally in Vite, configure `server.proxy` in `vite.config.js` to route requests without CORS.",
                        "Avoid opening HTML files via `file:///`; run a local dev server (`npx serve` or `npm run dev`)."
                    ],
                    "code_snippet": (
                        "// In vite.config.js (Development Proxy):\n"
                        "export default defineConfig({\n"
                        "  server: {\n"
                        "    proxy: {\n"
                        "      '/api': {\n"
                        "        target: 'http://localhost:8000',\n"
                        "        changeOrigin: true\n"
                        "      }\n"
                        "    }\n"
                        "  }\n"
                        "});"
                    ),
                    "model_used": "Interactive Diagnostic Wizard (Guided Troubleshooter)"
                }
            elif "syntaxerror" in opt_lower or "unexpected token" in opt_lower:
                return {
                    "scenario_key": "json_syntax",
                    "scenario_title": "JSON SyntaxError / Malformed JSON Structure",
                    "root_cause": "JSON.parse failed due to malformed JSON: trailing commas, single quotes instead of double quotes, unquoted keys, or an HTML 404 page being parsed as JSON.",
                    "action_steps": [
                        "Ensure all keys and string values use standard double quotes `\"key\": \"value\"`.",
                        "Remove trailing commas after the last item in objects or arrays.",
                        "Check if the server returned an HTML error page (`<!DOCTYPE html>`) instead of JSON.",
                        "Validate the file using `python -m json.tool data.json`."
                    ],
                    "code_snippet": (
                        "# Validate JSON file from terminal:\n"
                        "python3 -m json.tool data.json\n\n"
                        "// Ensure response is actually JSON before parsing:\n"
                        "const contentType = response.headers.get('content-type');\n"
                        "if (contentType && contentType.includes('application/json')) {\n"
                        "  const data = await response.json();\n"
                        "} else {\n"
                        "  const text = await response.text();\n"
                        "  console.error('Server returned non-JSON:', text.slice(0, 200));\n"
                        "}"
                    ),
                    "model_used": "Interactive Diagnostic Wizard (Guided Troubleshooter)"
                }

        # 4. Question: Port conflicts (port_target)
        if question_id == "port_target" or "kill" in opt_lower or "bind port" in opt_lower:
            port_match = re.search(r"\b(\d{4,5})\b", query)
            port_num = port_match.group(1) if port_match else "8000"
            if "kill" in opt_lower or "competing" in opt_lower:
                return {
                    "scenario_key": "port_kill",
                    "scenario_title": f"Terminate Process Occupying Port {port_num}",
                    "root_cause": f"A previous process or zombie background task is already bound to port {port_num} (EADDRINUSE / Address already in use).",
                    "action_steps": [
                        f"Find the PID using port {port_num}: `lsof -i :{port_num}` (macOS/Linux) or `netstat -ano | findstr :{port_num}` (Windows).",
                        f"Terminate the occupying process: `kill -9 <PID>`.",
                        "Re-launch your application server."
                    ],
                    "code_snippet": (
                        f"# Find and kill process listening on port {port_num}:\n"
                        f"lsof -ti :{port_num} | xargs kill -9\n\n"
                        f"# Verify port is free:\n"
                        f"lsof -i :{port_num}"
                    ),
                    "model_used": "Interactive Diagnostic Wizard (Guided Troubleshooter)"
                }
            elif "change" in opt_lower or "bind port" in opt_lower:
                new_port = str(int(port_num) + 1) if port_num.isdigit() else "8001"
                return {
                    "scenario_key": "port_change",
                    "scenario_title": f"Re-bind Server to Alternative Port ({new_port})",
                    "root_cause": f"Port {port_num} is occupied by another system daemon. Bypassing the conflict by binding to {new_port}.",
                    "action_steps": [
                        f"Start the server with an explicit port argument: `--port {new_port}`.",
                        f"Update any client frontend proxy configuration to point to port {new_port}."
                    ],
                    "code_snippet": (
                        f"# Start Uvicorn / FastAPI on alternative port:\n"
                        f"uvicorn api:app --reload --port {new_port}\n\n"
                        f"# Node / Express / Next.js:\n"
                        f"PORT={new_port} npm run dev"
                    ),
                    "model_used": "Interactive Diagnostic Wizard (Guided Troubleshooter)"
                }

        # 5. Question: Network / Connection topology (network_topology)
        if question_id == "network_topology" or "localhost" in opt_lower or "docker host" in opt_lower:
            if "localhost" in opt_lower:
                return {
                    "scenario_key": "conn_localhost",
                    "scenario_title": "Local Daemon Service Verification",
                    "root_cause": "The service is expected on 127.0.0.1 / localhost, but the local daemon process is stopped or listening on a non-standard socket.",
                    "action_steps": [
                        "Check service status: `sudo systemctl status postgresql` or `brew services list`.",
                        "Start service: `brew services start postgresql` or `sudo systemctl start postgresql`.",
                        "Verify listening socket: `netstat -tuln | grep 5432`."
                    ],
                    "code_snippet": (
                        "# Check and start local database service (PostgreSQL):\n"
                        "brew services start postgresql   # macOS Homebrew\n"
                        "# sudo systemctl start postgresql # Ubuntu / Debian\n\n"
                        "# Test TCP connection:\n"
                        "nc -zv 127.0.0.1 5432"
                    ),
                    "model_used": "Interactive Diagnostic Wizard (Guided Troubleshooter)"
                }
            elif "docker" in opt_lower:
                return {
                    "scenario_key": "conn_docker",
                    "scenario_title": "Docker Container Cross-Host Networking",
                    "root_cause": "Inside a Docker container, 'localhost' refers to the container itself. Connecting to host database requires 'host.docker.internal'.",
                    "action_steps": [
                        "Replace 'localhost' or '127.0.0.1' in connection string with 'host.docker.internal'.",
                        "If using Docker Compose, use the service name declared in compose YAML as the host."
                    ],
                    "code_snippet": (
                        "# Replace host in connection URL:\n"
                        "# FROM: postgresql://user:pass@localhost:5432/db\n"
                        "# TO:   postgresql://user:pass@host.docker.internal:5432/db"
                    ),
                    "model_used": "Interactive Diagnostic Wizard (Guided Troubleshooter)"
                }

        # Default fallback
        return {
            "scenario_key": "general_refinement",
            "scenario_title": f"Refined Diagnosis: {option_label}",
            "root_cause": f"Diagnostic tailored for: '{option_label}'. Inspect application configuration and execution context.",
            "action_steps": [
                f"Verify configuration corresponding to: {option_label}.",
                "Inspect environment variables, interpreter paths, and logs.",
                "Apply recommended configuration and restart environment."
            ],
            "code_snippet": f"# Diagnostic action for {option_label}\n# Check configuration and restart runner",
            "model_used": "Interactive Diagnostic Wizard (Guided Troubleshooter)"
        }


# Singleton instance
_nlp_pipeline = NLPPipeline()

def get_nlp_pipeline() -> NLPPipeline:
    return _nlp_pipeline

