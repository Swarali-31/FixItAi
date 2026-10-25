"""Comprehensive NLP Preprocessing and Linguistic Analysis Engine.

Directly covers Symbiosis NLP Syllabus (TEE7101):
- Unit 1 / Lab 1 & 2: Tokenization, Stopwords, Stemming, Lemmatization, POS Tagging, NER, Regular Expressions.
- Unit 4 / Lab 4: Sequence / Text Classification for Error Severity and Diagnostics.
"""
from __future__ import annotations

import html
import re
from typing import Any, Dict, List, Tuple

import nltk
from nltk.corpus import stopwords
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
    "Python", "Java", "C++", "C#", "JavaScript", "TypeScript", "SQL", "MySQL", 
    "PostgreSQL", "MongoDB", "Redis", "Docker", "Kubernetes", "Linux", "Ubuntu", 
    "Windows", "Git", "HTTP", "REST", "FastAPI", "Flask", "Django", "React", 
    "Vite", "Node.js", "PyTorch", "TensorFlow", "CUDA", "NVIDIA", "npm", "pip", 
    "Webpack", "Babel", "AWS", "Azure", "GCP", "Apache", "Nginx"
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
    r"\b(?:\w*(?:Exception|Error|Failure|Fault)\w*|ECONNREFUSED|ORA-\d+|HTTP\s*\d{3}|\d{3}\s+[A-Za-z]+)\b",
    re.I
)
CODE_OR_IDENTIFIER_REGEX = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b|\b\w+\(\)")
PATH_REGEX = re.compile(r"(?:[A-Za-z]:[\\/]|/)[A-Za-z0-9_./\\-]+\.[A-Za-z0-9]+")


class NLPPipeline:
    """End-to-end NLP processor satisfying TEE7101 Units 1, 2, 4."""

    def __init__(self):
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


# Singleton instance
_nlp_pipeline = NLPPipeline()

def get_nlp_pipeline() -> NLPPipeline:
    return _nlp_pipeline
