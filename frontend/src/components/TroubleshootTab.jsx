import React, { useState } from "react";
import {
  AlertTriangle, ArrowUpRight, CheckCheck, CircleHelp, Copy,
  Cpu, Database, Download, FileText, Globe, HelpCircle, Languages,
  Layers, LoaderCircle, Search, ShieldCheck, Sparkles,
  Terminal, Timer, UploadCloud, Wand2, Zap
} from "lucide-react";
import { getUIText } from "../i18n";

const REGIONAL_LANGUAGES = [
  { code: "en", label: "English" },
  { code: "hi", label: "Hindi (हिंदी)" },
  { code: "mr", label: "Marathi (मराठी)" },
  { code: "bn", label: "Bengali (বাংলা)" },
  { code: "ta", label: "Tamil (தமிழ்)" },
  { code: "te", label: "Telugu (తెలుగు)" },
];

const NATIVE_SECTION_LABELS = {
  hi: { cause: "कारण (Root Cause)", steps: "निवारण चरण (Resolution Steps)", fix: "सत्यापित कोड सुधार (Verified Fix)" },
  mr: { cause: "कारण (Root Cause)", steps: "निवारण पायऱ्या (Resolution Steps)", fix: "सत्यापित कोड सुधार (Verified Fix)" },
  bn: { cause: "কারণ (Root Cause)", steps: "সমাধান পদক্ষেপ (Resolution Steps)", fix: "যাচাইকৃত কোড সমাধান (Verified Fix)" },
  ta: { cause: "மூலக் காரணம் (Root Cause)", steps: "தீர்வு படிகள் (Resolution Steps)", fix: "சரிபார்க்கப்பட்ட குறியீடு (Verified Fix)" },
  te: { cause: "మూల కారణం (Root Cause)", steps: "పరిష్కార దశలు (Resolution Steps)", fix: "ధృవీకరించబడిన కోడ్ సవరణ (Verified Fix)" },
  en: { cause: "Root Cause", steps: "Actionable Steps", fix: "Verified Code Fix" }
};

export function TroubleshootTab({
  systemHealth,
  searchQuery,
  setSearchQuery,
  selectedTech,
  setSelectedTech,
  technologyFilters,
  retrievalMode,
  setRetrievalMode,
  presetExamples,
  executeTroubleshoot,
  isSearching,
  searchErrorMsg,
  hasSearched,
  searchResults,
  setSearchResults,
  diagnosticSolution,
  setDiagnosticSolution,
  linguisticAnalysis,
  showHindiView,
  setShowHindiView,
  hindiSolution,
  setHindiSolution,
  isTranslating,
  toggleHindiTranslation,
  searchMetadata,
  navigateToAnalyzerWithQuery,
  targetLang,
  setTargetLang,
  handleTranslateToLang,
  localizeEntirePage,
  setLocalizeEntirePage,
}) {
  const [copiedCodeSuccess, setCopiedCodeSuccess] = useState(false);
  const [exportNotice, setExportNotice] = useState(false);
  const [showLogParserModal, setShowLogParserModal] = useState(false);
  const [rawLogText, setRawLogText] = useState("");
  const [isParsingLog, setIsParsingLog] = useState(false);
  const [logParseResult, setLogParseResult] = useState(null);
  const [parseError, setParseError] = useState("");
  const [selectedWizardOptions, setSelectedWizardOptions] = useState({});
  const [wizardRefinementNotice, setWizardRefinementNotice] = useState(null);
  const [isRefiningWizard, setIsRefiningWizard] = useState(false);

  const activeLang = localizeEntirePage && targetLang !== "en" ? targetLang : "en";
  const t = (key) => getUIText(activeLang, key);
  const selectedLangMeta = REGIONAL_LANGUAGES.find((l) => l.code === targetLang) || REGIONAL_LANGUAGES[0];

  const acceleratorName = systemHealth?.gpu || "Vector Accelerator";
  const indexedCountFormatted = systemHealth?.indexed_passages?.toLocaleString() || "173,866";

  function handleCopyCode(code) {
    if (!code) return;
    if (navigator?.clipboard?.writeText) {
      navigator.clipboard.writeText(code).then(() => {
        setCopiedCodeSuccess(true);
        setTimeout(() => setCopiedCodeSuccess(false), 2000);
      });
    }
  }

  async function handleParseLog(textToParse, autoDiagnose = false) {
    const text = (textToParse || rawLogText).trim();
    if (!text) return;
    setIsParsingLog(true);
    setParseError("");
    try {
      const res = await fetch("/api/parse-log", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ raw_text: text }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || `Server returned error status ${res.status}`);
      }
      setLogParseResult(data);
      if (data.clean_query) {
        setSearchQuery(data.clean_query);
        if (autoDiagnose) {
          executeTroubleshoot(data.clean_query);
        }
      }
    } catch (err) {
      console.error("Log parse failed:", err);
      setParseError(err.message || "Failed to parse log trace");
    } finally {
      setIsParsingLog(false);
    }
  }

  function handleDropLogFile(e) {
    e.preventDefault();
    const files = e.dataTransfer?.files;
    if (files && files[0]) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const content = event.target?.result;
        if (typeof content === "string") {
          setRawLogText(content);
          handleParseLog(content);
        }
      };
      reader.readAsText(files[0]);
    }
  }

  async function handleSelectWizardOption(q, opt) {
    const isAlreadySelected = selectedWizardOptions[q.id] === opt.label;
    const updated = { ...selectedWizardOptions };
    if (isAlreadySelected) {
      delete updated[q.id];
    } else {
      updated[q.id] = opt.label;
    }
    setSelectedWizardOptions(updated);

    if (Object.keys(updated).length === 0) {
      setWizardRefinementNotice(null);
      executeTroubleshoot(searchQuery);
      return;
    }

    setIsRefiningWizard(true);

    try {
      const res = await fetch("/api/wizard-refine", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: searchQuery,
          question_id: q.id,
          option_label: opt.label,
          refinement: opt.refinement || opt.label,
          selected_options: updated,
          target_lang: targetLang
        })
      });
      const data = await res.json();
      if (res.ok && data.solution) {
        if (typeof setDiagnosticSolution === "function") {
          setDiagnosticSolution(data.solution);
        }
        if (typeof setSearchResults === "function" && data.results) {
          setSearchResults(data.results);
        }
        if (data.translated_solution && typeof setHindiSolution === "function") {
          setHindiSolution(data.translated_solution);
        }
        setWizardRefinementNotice({
          questionId: q.id,
          label: Object.values(updated).join(" + "),
          sourceType: data.solution.source_type,
          sourceTitle: data.solution.source_title,
          kbNotice: data.solution.kb_notice,
          modelUsed: data.solution.model_used
        });
      }
    } catch (err) {
      console.error("Wizard refinement failed:", err);
    } finally {
      setIsRefiningWizard(false);
    }
  }

  function handleExportMarkdown() {
    if (!diagnosticSolution) return;
    const dateStr = new Date().toISOString().split("T")[0];
    const mdContent = `# FixItAI Technical Diagnostic Report
**Date:** ${dateStr}
**Query:** ${searchQuery}
**Severity Level:** ${linguisticAnalysis?.severity?.level || "NORMAL"}
**Retrieval Mode:** ${searchMetadata?.retrieval_mode || retrievalMode} (${searchMetadata?.latency_ms || "N/A"} ms)
**Device Accelerator:** ${searchMetadata?.device || systemHealth?.device || "N/A"}
**Confidence Level:** ${searchMetadata?.confidence?.level || "N/A"} (${searchMetadata?.confidence?.score || "N/A"}%)

---

## 1. Root Cause Summary
${diagnosticSolution.root_cause || "N/A"}

## 2. Actionable Step-by-Step Resolution
${diagnosticSolution.action_steps?.map((step, i) => `${i + 1}. ${step}`).join("\n") || "No steps available."}

${
  diagnosticSolution.code_snippet
    ? `## 3. Verified Code / Command Fix\n\`\`\`bash\n${diagnosticSolution.code_snippet.trim()}\n\`\`\`\n`
    : ""
}

${
  hindiSolution?.translated_text
    ? `## 4. Regional Vernacular Diagnostic (${hindiSolution.language_name || selectedLangMeta.label})\n${hindiSolution.translated_text}\n`
    : ""
}

## 5. Reference Passages
${searchResults
  .slice(0, 3)
  .map(
    (res, i) => `### Match #${i + 1}: ${res.title} (${(res.score * 100).toFixed(1)}% match)
- **Source:** ${res.source}
- **URL:** ${res.url || "N/A"}
${res.text.slice(0, 300)}...
`
  )
  .join("\n")}

---
*Generated by FixItAI Diagnostic Intelligence Engine*
`;

    const blob = new Blob([mdContent], { type: "text/markdown;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `fixitai-report-${Date.now()}.md`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setExportNotice(true);
    setTimeout(() => setExportNotice(false), 2500);
  }

  const confidence = searchMetadata?.confidence;
  const guardrail = searchMetadata?.guardrail;
  const typoCorrection = searchMetadata?.typo_correction;
  const wordnetExp = searchMetadata?.wordnet_expansion;
  const wizardQuestions = searchMetadata?.wizard_questions;
  const indicMeta = searchMetadata?.indic_meta;
  const currentSectionLabels = NATIVE_SECTION_LABELS[targetLang] || NATIVE_SECTION_LABELS.en;

  const isRegionalActive = Boolean(showHindiView && hindiSolution && targetLang !== "en");
  const displayRootCause = isRegionalActive
    ? (hindiSolution.root_cause_localized || hindiSolution.root_cause_hi || diagnosticSolution?.root_cause || "")
    : (diagnosticSolution?.root_cause || "");

  const displayActionSteps = isRegionalActive
    ? (hindiSolution.action_steps_localized || hindiSolution.action_steps_hi || diagnosticSolution?.action_steps || [])
    : (diagnosticSolution?.action_steps || []);

  return (
    <div className="tab-pane">
      {/* Header */}
      <div className="page-header">
        <div>
          <span className="header-eyebrow">{t("eyebrow")}</span>
          <h1>{t("page_title")}</h1>
          <p>{t("page_subtitle")}</p>
        </div>
        <div className="header-actions">
          <div className="status-badge-group">
            <span className="status-badge">
              <Database size={13} /> {indexedCountFormatted} {t("passages")}
            </span>
            <span className="status-badge">
              <Cpu size={13} /> {retrievalMode} (RRF)
            </span>
            <span className="status-badge">
              <Zap size={13} /> {acceleratorName}
            </span>
          </div>
        </div>
      </div>

      {/* Stack Trace / Crash Log Dropzone Toggle & Regional Controls */}
      <div style={{ marginBottom: "14px", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "10px" }}>
        <button
          type="button"
          className="btn btn-sm btn-secondary"
          onClick={() => setShowLogParserModal(!showLogParserModal)}
          style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}
        >
          <UploadCloud size={14} />
          <span>{showLogParserModal ? t("stack_trace_button_hide") : t("stack_trace_button_show")}</span>
        </button>

        {/* Indic Multi-lingual Language Selector & Full Page Localizer */}
        <div style={{ display: "flex", alignItems: "center", gap: "12px", flexWrap: "wrap" }}>
          <label style={{ fontSize: "12px", color: "var(--color-text-secondary, #64748b)", display: "flex", alignItems: "center", gap: "5px", cursor: "pointer", background: "rgba(99, 102, 241, 0.05)", padding: "4px 8px", borderRadius: "6px" }}>
            <input
              type="checkbox"
              checked={localizeEntirePage}
              onChange={(e) => setLocalizeEntirePage(e.target.checked)}
              style={{ accentColor: "#6366f1" }}
            />
            <span style={{ fontWeight: 600 }}>{t("localize_page_toggle")}</span>
          </label>

          <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
            <span style={{ fontSize: "12px", color: "var(--color-text-secondary, #64748b)", display: "flex", alignItems: "center", gap: "4px" }}>
              <Languages size={13} /> {t("regional_view_label")}
            </span>
            <select
              value={targetLang}
              onChange={(e) => {
                const newLang = e.target.value;
                setTargetLang(newLang);
                if (newLang === "en") {
                  setShowHindiView(false);
                } else {
                  setShowHindiView(true);
                  if (diagnosticSolution) {
                    handleTranslateToLang(newLang);
                  }
                }
              }}
              style={{ padding: "4px 8px", fontSize: "12px", borderRadius: "6px", border: "1px solid var(--color-border, #cbd5e1)", background: "#fff", fontWeight: 600 }}
            >
              {REGIONAL_LANGUAGES.map((l) => (
                <option key={l.code} value={l.code}>
                  {l.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Log Parser Dropzone Panel */}
      {showLogParserModal && (
        <section
          className="search-card"
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDropLogFile}
          style={{ border: "2px dashed #6366f1", background: "rgba(99, 102, 241, 0.03)", marginBottom: "16px" }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
            <strong style={{ fontSize: "13px", display: "flex", alignItems: "center", gap: "6px", color: "#4f46e5" }}>
              <Terminal size={15} /> {t("stack_trace_title")}
            </strong>
            <span style={{ fontSize: "11px", color: "#64748b" }}>{t("stack_trace_hint")}</span>
          </div>

          <textarea
            rows={4}
            value={rawLogText}
            onChange={(e) => setRawLogText(e.target.value)}
            placeholder="Paste raw terminal traceback, e.g.:
Traceback (most recent call last):
  File 'server.py', line 45, in <module>
    import fastapi
ModuleNotFoundError: No module named 'fastapi'..."
            style={{ width: "100%", fontFamily: "monospace", fontSize: "12px", padding: "8px", borderRadius: "6px", border: "1px solid #cbd5e1" }}
          />

          {parseError && (
            <div style={{ color: "#dc2626", fontSize: "12px", marginTop: "8px", background: "#fef2f2", padding: "6px 10px", borderRadius: "6px" }}>
              ⚠️ Parse Error: {parseError}
            </div>
          )}

          <div style={{ marginTop: "10px", display: "flex", gap: "10px", alignItems: "center", flexWrap: "wrap" }}>
            <button
              type="button"
              className="btn btn-sm btn-primary"
              onClick={() => handleParseLog()}
              disabled={isParsingLog || !rawLogText.trim()}
            >
              {isParsingLog ? <LoaderCircle className="spinner" size={14} /> : <Wand2 size={14} />}
              <span>{isParsingLog ? "Sanitizing..." : t("parse_populate")}</span>
            </button>

            <button
              type="button"
              className="btn btn-sm btn-secondary"
              onClick={() => handleParseLog(null, true)}
              disabled={isParsingLog || !rawLogText.trim()}
            >
              <Search size={14} />
              <span>{t("parse_diagnose_now")}</span>
            </button>

            {logParseResult?.isolated_frame && (
              <span style={{ fontSize: "12px", color: "#059669", fontWeight: 600 }}>
                ✓ {t("frame_isolated")} {logParseResult.isolated_frame}
              </span>
            )}
          </div>

          {logParseResult?.clean_query && (
            <div style={{ marginTop: "12px", padding: "8px 12px", background: "#f8fafc", border: "1px solid #e2e8f0", borderRadius: "6px", fontSize: "12px", color: "#334155" }}>
              <span style={{ fontWeight: 600, color: "#4f46e5" }}>{t("cleaned_signature")}</span>
              <p style={{ margin: "2px 0 0 0", fontFamily: "monospace" }}>{logParseResult.clean_query}</p>
            </div>
          )}
        </section>
      )}

      {/* Diagnostic Query Card */}
      <section className="search-card">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            executeTroubleshoot();
          }}
        >
          <div className="search-input-header">
            <label htmlFor="diagnostic-query">{t("query_label")}</label>
            <span className="input-tip">{t("query_hint")}</span>
          </div>

          <div className="textarea-wrapper">
            <textarea
              id="diagnostic-query"
              rows={3}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  executeTroubleshoot();
                }
              }}
              placeholder={t("query_placeholder")}
            />
          </div>

          {/* Preset Chips */}
          <div className="preset-strip">
            <span className="preset-label">{t("quick_presets")}</span>
            <div className="preset-buttons">
              {presetExamples.map((example) => (
                <button
                  key={example}
                  type="button"
                  className="chip-btn"
                  onClick={() => {
                    setSearchQuery(example);
                    executeTroubleshoot(example);
                  }}
                >
                  {example}
                </button>
              ))}
            </div>
          </div>

          {/* Search Controls */}
          <div className="search-controls">
            <div className="control-group">
              <label>{t("tech_filter")}</label>
              <select value={selectedTech} onChange={(e) => setSelectedTech(e.target.value)}>
                {technologyFilters.map((tItem) => (
                  <option key={tItem} value={tItem}>
                    {tItem === "All technologies" ? t("all_tech") : tItem}
                  </option>
                ))}
              </select>
            </div>

            <div className="control-group">
              <label>{t("retrieval_scheme")}</label>
              <select value={retrievalMode} onChange={(e) => setRetrievalMode(e.target.value)}>
                <option value="Hybrid">Hybrid (BM25 + Dense RRF)</option>
                <option value="Linear-Hybrid">Linear-Hybrid (Weighted Sum)</option>
                <option value="Semantic">Semantic (Dense BERT Embeddings)</option>
                <option value="BM25">BM25 (Probabilistic BM25Okapi)</option>
                <option value="TF-IDF">TF-IDF (Sublinear N-Gram Vectorizer)</option>
              </select>
            </div>

            <button
              type="submit"
              className="btn btn-primary search-action-btn"
              disabled={isSearching || !searchQuery.trim()}
            >
              {isSearching ? <LoaderCircle className="spinner" size={16} /> : <Search size={16} />}
              <span>{isSearching ? t("diagnosing_btn") : t("diagnose_btn")}</span>
            </button>
          </div>
        </form>
      </section>

      {/* Out-of-Domain Guardrail Alert */}
      {guardrail && !guardrail.is_technical && (
        <div className="alert alert-error" style={{ background: "#fffbeb", border: "1px solid #fcd34d", color: "#92400e" }}>
          <ShieldCheck size={18} style={{ color: "#d97706" }} />
          <div>
            <strong>{t("guardrail_title")}</strong>
            <p style={{ margin: "2px 0 0 0", fontSize: "12px" }}>{guardrail.guardrail_message}</p>
          </div>
        </div>
      )}

      {/* Cross-Lingual Input Translation Notice */}
      {indicMeta?.was_translated && (
        <div style={{ background: "rgba(99, 102, 241, 0.08)", padding: "6px 12px", borderRadius: "6px", fontSize: "12px", marginBottom: "12px", color: "#4f46e5", display: "flex", alignItems: "center", gap: "6px" }}>
          <Languages size={14} />
          <span>
            Detected input in <strong>{indicMeta.language_name}</strong>. Translated query for hybrid search: <em>"{indicMeta.translated_query}"</em>
          </span>
        </div>
      )}

      {/* Typo Auto-Correction Suggestion */}
      {typoCorrection?.has_correction && (
        <div style={{ background: "#f0fdf4", border: "1px solid #bbf7d0", padding: "8px 12px", borderRadius: "6px", fontSize: "12px", marginBottom: "12px", color: "#166534", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span>
            💡 {t("typo_did_you_mean")} <strong>{typoCorrection.corrected_query}</strong>?
          </span>
          <button
            type="button"
            className="btn btn-sm btn-secondary"
            onClick={() => {
              setSearchQuery(typoCorrection.corrected_query);
              executeTroubleshoot(typoCorrection.corrected_query);
            }}
          >
            {t("apply_fix")}
          </button>
        </div>
      )}

      {/* WordNet Synset Expansion Chips */}
      {wordnetExp?.expanded_terms && Object.keys(wordnetExp.expanded_terms).length > 0 && (
        <div style={{ marginBottom: "12px", fontSize: "11px", color: "#64748b", display: "flex", alignItems: "center", gap: "6px", flexWrap: "wrap" }}>
          <span style={{ fontWeight: 600 }}>{t("wordnet_synsets")}</span>
          {Object.entries(wordnetExp.expanded_terms).map(([orig, syns]) => (
            <span key={orig} className="pill" style={{ background: "#f1f5f9" }}>
              {orig} → {syns.slice(0, 2).join(", ")}
            </span>
          ))}
          {searchMetadata?.perplexity && (
            <span className="pill" style={{ marginLeft: "auto" }}>
              Perplexity: {searchMetadata.perplexity}
            </span>
          )}
        </div>
      )}

      {/* Latency & Confidence Gauge */}
      {searchMetadata?.latency_ms !== undefined && !isSearching && (
        <div style={{ display: "flex", gap: "10px", alignItems: "center", marginBottom: "16px", fontSize: "12px", color: "var(--color-text-secondary, #64748b)", flexWrap: "wrap" }}>
          <span style={{ display: "inline-flex", alignItems: "center", gap: "4px", background: "rgba(99, 102, 241, 0.08)", padding: "4px 8px", borderRadius: "6px", color: "#6366f1", fontWeight: 600 }}>
            <Timer size={13} /> {searchMetadata.latency_ms} ms
          </span>

          {confidence && (
            <span
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "4px",
                padding: "4px 8px",
                borderRadius: "6px",
                fontWeight: 600,
                background: confidence.level === "HIGH" ? "#dcfce7" : confidence.level === "MEDIUM" ? "#fef3c7" : "#fee2e2",
                color: confidence.level === "HIGH" ? "#15803d" : confidence.level === "MEDIUM" ? "#b45309" : "#b91c1c"
              }}
            >
              {t("confidence_label")} {confidence.score}% ({confidence.level})
            </span>
          )}

          <span>• Mode: <strong>{searchMetadata.retrieval_mode || retrievalMode}</strong></span>
          <span>• Accelerator: <strong>{searchMetadata.device || systemHealth?.device}</strong></span>
        </div>
      )}

      {searchErrorMsg && (
        <div className="alert alert-error">
          <AlertTriangle size={16} />
          <span>{searchErrorMsg}</span>
        </div>
      )}

      {isSearching && (
        <div className="loading-state">
          <LoaderCircle className="spinner" size={30} />
          <p>Analyzing syntax, querying hybrid indices with RRF, and synthesizing fix...</p>
        </div>
      )}

      {/* Interactive Diagnostic Wizard Questions */}
      {wizardQuestions && wizardQuestions.length > 0 && diagnosticSolution && (
        <section className="search-card" style={{ background: "rgba(248, 250, 252, 0.8)", border: "1px solid #e2e8f0", marginBottom: "16px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "10px" }}>
            <HelpCircle size={15} style={{ color: "#6366f1" }} />
            <strong style={{ fontSize: "13px" }}>{t("wizard_title")}</strong>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            {wizardQuestions.map((q) => (
              <div key={q.id}>
                <span style={{ fontSize: "12px", fontWeight: 600, color: "#334155", display: "block", marginBottom: "4px" }}>
                  {q.question}
                </span>
                <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                  {q.options.map((opt) => {
                    const isSelected = selectedWizardOptions[q.id] === opt.label;
                    return (
                      <button
                        key={opt.label}
                        type="button"
                        className="chip-btn"
                        style={{
                          fontSize: "11px",
                          background: isSelected ? "#eef2ff" : "#fff",
                          border: isSelected ? "2px solid #6366f1" : "1px solid #cbd5e1",
                          color: isSelected ? "#4338ca" : "#334155",
                          fontWeight: isSelected ? 700 : 500,
                          boxShadow: isSelected ? "0 0 0 2px rgba(99, 102, 241, 0.2)" : "none",
                          display: "inline-flex",
                          alignItems: "center",
                          gap: "4px",
                          cursor: "pointer"
                        }}
                        onClick={() => handleSelectWizardOption(q, opt)}
                      >
                        {isSelected && <CheckCheck size={12} style={{ color: "#6366f1" }} />}
                        {opt.label}
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* AI Synthesized Diagnostic Card */}
      {diagnosticSolution && (
        <section className="diagnostic-card">
          {/* Active Scenario Refinement Banner */}
          {wizardRefinementNotice && (
            <div
              style={{
                background: "rgba(99, 102, 241, 0.08)",
                border: "1px solid #c7d2fe",
                borderRadius: "8px",
                padding: "8px 12px",
                marginBottom: "12px",
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                flexWrap: "wrap",
                gap: "8px"
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "12px", color: "#3730a3" }}>
                <Wand2 size={14} style={{ color: "#6366f1" }} />
                <span>
                  <strong>Active Scenario:</strong> {wizardRefinementNotice.label}
                  {wizardRefinementNotice.sourceType === "knowledge_base" ? (
                    <span style={{ marginLeft: "8px", fontSize: "11px", color: "#15803d", background: "#dcfce7", padding: "2px 6px", borderRadius: "4px", fontWeight: 600 }}>
                      ✓ Fetched from Knowledge Base: {wizardRefinementNotice.sourceTitle || "Passage"}
                    </span>
                  ) : (
                    <span style={{ marginLeft: "8px", fontSize: "11px", color: "#b45309", background: "#fef3c7", padding: "2px 6px", borderRadius: "4px", fontWeight: 600 }}>
                      ⚡ Contextual Diagnostic Inference (Out-of-KB)
                    </span>
                  )}
                </span>
              </div>
              <button
                type="button"
                className="btn btn-sm btn-ghost"
                style={{ fontSize: "11px", padding: "2px 8px", color: "#4f46e5", cursor: "pointer" }}
                onClick={() => {
                  setSelectedWizardOptions({});
                  setWizardRefinementNotice(null);
                  executeTroubleshoot(searchQuery);
                }}
              >
                Reset to Default Analysis
              </button>
            </div>
          )}
          <div className="card-top-bar">
            <div className="badge-group">
              <span className="badge badge-accent">
                <Sparkles size={13} /> {t("diagnostic_resolution")}
              </span>
              {isRegionalActive && (
                <span className="badge" style={{ background: "rgba(99, 102, 241, 0.12)", color: "#4338ca", fontWeight: 700 }}>
                  <Languages size={12} /> {hindiSolution.language_native || selectedLangMeta.label}
                </span>
              )}
              {linguisticAnalysis?.severity && (
                <span className={`badge severity-${linguisticAnalysis.severity.level}`}>
                  <AlertTriangle size={12} /> {linguisticAnalysis.severity.level.replace("_", " ")}
                </span>
              )}
              <span className="badge badge-neutral">
                <Cpu size={12} /> {diagnosticSolution.model_used || "Transformer Diagnostic Engine"}
              </span>
            </div>

            <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
              <button
                className="btn btn-sm btn-secondary"
                onClick={handleExportMarkdown}
                title="Export report as Markdown"
              >
                <Download size={13} />
                <span>{exportNotice ? t("exported") : t("export_report")}</span>
              </button>

              <button
                className={`btn btn-sm ${showHindiView ? "btn-active" : "btn-secondary"}`}
                onClick={toggleHindiTranslation}
                disabled={isTranslating}
              >
                <Languages size={14} />
                <span>
                  {isTranslating
                    ? t("translating")
                    : showHindiView
                    ? "View in English"
                    : `${selectedLangMeta.label} View`}
                </span>
              </button>
            </div>
          </div>

          {/* Root Cause in Chosen Language */}
          <div className="diag-section">
            <h3 className="section-title">
              {isRegionalActive ? (currentSectionLabels.cause || t("root_cause")) : t("root_cause")}
            </h3>
            <p className="root-cause-text" style={{ fontSize: isRegionalActive ? "14.5px" : "14px", lineHeight: "1.65" }}>
              {displayRootCause}
            </p>
          </div>

          {/* Actionable Steps in Chosen Language */}
          {displayActionSteps?.length > 0 && (
            <div className="diag-section">
              <h3 className="section-title">
                {isRegionalActive ? (currentSectionLabels.steps || t("action_steps")) : t("action_steps")}
              </h3>
              <ol className="step-numbered-list" style={{ fontSize: isRegionalActive ? "14px" : "13.5px" }}>
                {displayActionSteps.map((step, idx) => (
                  <li key={idx}>{step}</li>
                ))}
              </ol>
            </div>
          )}

          {/* Code Fix */}
          {diagnosticSolution.code_snippet && (
            <div className="code-box">
              <div className="code-box-header">
                <span>
                  <Terminal size={13} /> {t("verified_fix")}
                </span>
                <button
                  className="btn-copy"
                  type="button"
                  onClick={() => handleCopyCode(diagnosticSolution.code_snippet)}
                >
                  {copiedCodeSuccess ? <CheckCheck size={13} /> : <Copy size={13} />}
                  <span>{copiedCodeSuccess ? t("copied") : t("copy_code")}</span>
                </button>
              </div>
              <pre>
                <code>{diagnosticSolution.code_snippet}</code>
              </pre>
            </div>
          )}
        </section>
      )}

      {/* Search Results Evidence List */}
      {hasSearched && !isSearching && searchResults.length > 0 && (
        <section className="results-section">
          <div className="results-header">
            <h2>
              {t("reference_passages")} ({searchResults.length})
            </h2>
            <button
              className="btn btn-sm btn-secondary"
              onClick={() => navigateToAnalyzerWithQuery(searchQuery)}
            >
              <Layers size={14} />
              <span>{t("inspect_analyzer")}</span>
            </button>
          </div>

          <div className="evidence-list">
            {searchResults.map((item, idx) => (
              <article className={`evidence-card ${idx === 0 ? "primary-evidence" : ""}`} key={item.chunk_id}>
                <div className="evidence-card-header">
                  <span className="rank-num">#{idx + 1}</span>
                  <div className="title-area">
                    <span className="source-name">
                      {item.source} {idx === 0 && "· Primary Reference (Synthesized Above)"}
                    </span>
                    <h4>{item.title}</h4>
                  </div>
                  <span className="score-badge">{(item.score * 100).toFixed(1)}% {t("match")}</span>
                </div>

                <p className="passage-body">{item.text}</p>

                <div className="score-pills-row">
                  <span className="pill">BM25: {item.bm25_score?.toFixed(2) || "0.00"}</span>
                  <span className="pill">TF-IDF: {item.tfidf_score?.toFixed(2) || "0.00"}</span>
                  <span className="pill">BERT: {item.semantic_score?.toFixed(2) || "0.00"}</span>
                  {item.matched_terms?.length > 0 && (
                    <span className="pill pill-highlight">
                      Matched: {item.matched_terms.slice(0, 3).join(", ")}
                    </span>
                  )}
                </div>

                <div className="evidence-card-footer">
                  {item.url && (
                    <a href={item.url} target="_blank" rel="noreferrer" className="external-link">
                      <span>Open source discussion</span>
                      <ArrowUpRight size={13} />
                    </a>
                  )}
                  {item.question_url && (
                    <a href={item.question_url} target="_blank" rel="noreferrer" className="external-link">
                      <span>Original inquiry</span>
                      <ArrowUpRight size={13} />
                    </a>
                  )}
                </div>
              </article>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
