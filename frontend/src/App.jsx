import { useEffect, useState } from "react";
import {
  ArrowDownRight, ArrowUpRight, BookOpen, Check,
  CircleHelp, Cpu, Database, FileText, History,
  LoaderCircle, MessageSquareText, Search, Sparkles,
  Upload, X, Terminal, Languages, Layers, Copy, CheckCheck, AlertTriangle, Zap, Play
} from "lucide-react";

const navigationItems = [
  { id: "troubleshoot", label: "Troubleshoot", icon: Search },
  { id: "analyzer", label: "Error Analyzer", icon: Layers },
  { id: "knowledge", label: "Knowledge Base", icon: BookOpen },
  { id: "history", label: "History", icon: History },
];

const presetExamples = [
  "Python raises ModuleNotFoundError: No module named 'fastapi'",
  "Java NullPointerException in service controller",
  "Docker bind address already in use on port 8080",
  "MySQL connection timeout from client application",
];

const technologyFilters = [
  "All technologies", "Python", "Java", "Docker", "JavaScript", "SQL",
  "MySQL", "Linux", "Windows", "Git", "HTTP", "Kubernetes"
];

async function apiRequest(endpoint, options = {}) {
  const response = await fetch(`/api${endpoint}`, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}

function PageHeader({ eyebrow, title, description, actions }) {
  return (
    <div className="page-header">
      <div>
        {eyebrow && <span className="header-eyebrow">{eyebrow}</span>}
        <h1>{title}</h1>
        {description && <p>{description}</p>}
      </div>
      {actions && <div className="header-actions">{actions}</div>}
    </div>
  );
}

function MetricCard({ icon: Icon, label, value, detail }) {
  return (
    <div className="metric-card">
      <div className="metric-icon"><Icon size={18} /></div>
      <div className="metric-content">
        <span className="metric-label">{label}</span>
        <div className="metric-value">{value}</div>
        {detail && <span className="metric-detail">{detail}</span>}
      </div>
    </div>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState("troubleshoot");
  const [systemHealth, setSystemHealth] = useState(null);

  // Troubleshoot State
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedTech, setSelectedTech] = useState("All technologies");
  const [retrievalMode, setRetrievalMode] = useState("Hybrid");
  const [hasSearched, setHasSearched] = useState(false);
  const [searchResults, setSearchResults] = useState([]);
  const [linguisticAnalysis, setLinguisticAnalysis] = useState(null);
  const [diagnosticSolution, setDiagnosticSolution] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const [searchErrorMsg, setSearchErrorMsg] = useState("");
  const [copiedCodeSuccess, setCopiedCodeSuccess] = useState(false);

  // Translation State
  const [isTranslating, setIsTranslating] = useState(false);
  const [hindiSolution, setHindiSolution] = useState(null);
  const [showHindiView, setShowHindiView] = useState(false);

  // Linguistic Analyzer State
  const [analyzerInput, setAnalyzerInput] = useState("Python raises ModuleNotFoundError: No module named 'fastapi' when executing main.py");
  const [analyzerOutput, setAnalyzerOutput] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Knowledge Base State
  const [libraryData, setLibraryData] = useState({ items: [], total: 0 });
  const [librarySearchTerm, setLibrarySearchTerm] = useState("");
  const [isLibraryLoading, setIsLibraryLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  // History State
  const [searchHistoryList, setSearchHistoryList] = useState([]);
  const [feedbackNotice, setFeedbackNotice] = useState("");
  const [generalError, setGeneralError] = useState("");

  useEffect(() => {
    apiRequest("/health")
      .then(setSystemHealth)
      .catch(() => setSystemHealth({ status: "offline" }));
  }, []);

  useEffect(() => {
    setGeneralError("");
    if (activeTab === "history") {
      apiRequest("/history")
        .then((data) => setSearchHistoryList(data.items || []))
        .catch((err) => setGeneralError(err.message));
    }
    if (activeTab === "knowledge") {
      fetchLibraryPassages("");
    }
    if (activeTab === "analyzer" && !analyzerOutput) {
      executeLinguisticAnalysis(analyzerInput);
    }
  }, [activeTab]);

  async function fetchLibraryPassages(filter = librarySearchTerm, append = false) {
    setIsLibraryLoading(true);
    try {
      const offset = append ? libraryData.items.length : 0;
      const params = new URLSearchParams({ search_text: filter, offset: String(offset), limit: "50" });
      const data = await apiRequest(`/knowledge-base?${params.toString()}`);
      setLibraryData((prev) => append ? { ...data, items: [...prev.items, ...data.items] } : data);
    } catch (err) {
      setGeneralError(err.message);
    } finally {
      setIsLibraryLoading(false);
    }
  }

  async function executeTroubleshoot(customQuery) {
    const q = (typeof customQuery === "string" ? customQuery : searchQuery).trim();
    if (!q) return;

    if (typeof customQuery === "string") {
      setSearchQuery(customQuery);
    }

    setIsSearching(true);
    setSearchErrorMsg("");
    setHasSearched(true);
    setSearchResults([]);
    setDiagnosticSolution(null);
    setHindiSolution(null);
    setShowHindiView(false);

    try {
      const data = await apiRequest("/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: q,
          technology: selectedTech === "All technologies" ? "All" : selectedTech,
          mode: retrievalMode,
          top_k: 5,
        }),
      });
      setSearchResults(data.results || []);
      setLinguisticAnalysis(data.analysis || null);
      setDiagnosticSolution(data.ai_solution || null);
    } catch (err) {
      setSearchErrorMsg(err.message);
    } finally {
      setIsSearching(false);
    }
  }

  async function toggleHindiTranslation() {
    if (!diagnosticSolution) return;
    if (hindiSolution) {
      setShowHindiView(!showHindiView);
      return;
    }

    setIsTranslating(true);
    try {
      const data = await apiRequest("/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: searchQuery,
          root_cause: diagnosticSolution.root_cause || "",
          action_steps: diagnosticSolution.action_steps || [],
          code_snippet: diagnosticSolution.code_snippet || "",
          text: diagnosticSolution.root_cause || "",
        }),
      });
      setHindiSolution(data);
      setShowHindiView(true);
    } catch (err) {
      setFeedbackNotice(`Translation notice: ${err.message}`);
    } finally {
      setIsTranslating(false);
    }
  }

  async function executeLinguisticAnalysis(text) {
    if (!text?.trim()) return;
    setIsAnalyzing(true);
    try {
      const data = await apiRequest("/nlp-analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      setAnalyzerOutput(data);
    } catch (err) {
      setGeneralError(err.message);
    } finally {
      setIsAnalyzing(false);
    }
  }

  function handleCopyCode(code) {
    if (!code) return;
    if (navigator?.clipboard?.writeText) {
      navigator.clipboard.writeText(code).then(() => {
        setCopiedCodeSuccess(true);
        setTimeout(() => setCopiedCodeSuccess(false), 2000);
      });
    }
  }

  function navigateToAnalyzerWithQuery(text) {
    setAnalyzerInput(text);
    setActiveTab("analyzer");
    executeLinguisticAnalysis(text);
  }

  async function handleDocumentUpload(files) {
    if (!files?.length) return;
    setIsUploading(true);
    setFeedbackNotice("");
    const form = new FormData();
    [...files].forEach((f) => form.append("files", f));

    try {
      const data = await apiRequest("/knowledge-base/upload", { method: "POST", body: form });
      setFeedbackNotice(data.message);
      await fetchLibraryPassages(librarySearchTerm);
    } catch (err) {
      setGeneralError(err.message);
    } finally {
      setIsUploading(false);
    }
  }

  async function handleFeedback(index, feedbackType) {
    try {
      await apiRequest("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ index_from_newest: index, feedback: feedbackType }),
      });
      const data = await apiRequest("/history");
      setSearchHistoryList(data.items || []);
      setFeedbackNotice("Feedback recorded. Thank you!");
      setTimeout(() => setFeedbackNotice(""), 3000);
    } catch (err) {
      setGeneralError(err.message);
    }
  }

  function handleRerunQuery(query) {
    setActiveTab("troubleshoot");
    setSearchQuery(query);
    executeTroubleshoot(query);
  }

  const isModelReady = systemHealth?.status === "ready";
  const acceleratorName = systemHealth?.gpu?.replace("NVIDIA GeForce ", "") || "CUDA Acceleration";
  const indexedCountFormatted = systemHealth?.indexed_passages?.toLocaleString() || "173,866";

  return (
    <div className="product-layout">
      {/* Sleek Top Navigation Bar */}
      <header className="navbar">
        <div className="navbar-container">
          <div className="nav-brand" onClick={() => setActiveTab("troubleshoot")}>
            <div className="brand-logo"><Sparkles size={20} /></div>
            <div className="brand-titles">
              <span className="brand-name">FixItAI</span>
              <span className="brand-tag">Diagnostic Intelligence</span>
            </div>
          </div>

          <nav className="nav-links">
            {navigationItems.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                className={`nav-button ${activeTab === id ? "active" : ""}`}
                onClick={() => setActiveTab(id)}
              >
                <Icon size={16} />
                <span>{label}</span>
              </button>
            ))}
          </nav>

          <div className="nav-status">
            <span className={`status-indicator ${isModelReady ? "ready" : "warning"}`} />
            <span>{isModelReady ? "Engine Ready" : "Initializing"}</span>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <main className="content-container">
        {feedbackNotice && (
          <div className="alert alert-success">
            <Check size={16} />
            <span>{feedbackNotice}</span>
            <button className="alert-close" onClick={() => setFeedbackNotice("")}><X size={14} /></button>
          </div>
        )}
        {generalError && (
          <div className="alert alert-error">
            <AlertTriangle size={16} />
            <span>{generalError}</span>
            <button className="alert-close" onClick={() => setGeneralError("")}><X size={14} /></button>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 1: TROUBLESHOOT */}
        {/* ========================================================================= */}
        {activeTab === "troubleshoot" && (
          <div className="tab-pane">
            <PageHeader
              eyebrow="INTELLIGENT DIAGNOSTICS"
              title="Technical Error Diagnostic"
              description="Synthesize root-cause analysis, step-by-step resolutions, and code fixes from 170k+ developer solutions."
              actions={
                <div className="status-badge-group">
                  <span className="status-badge"><Database size={13} /> {indexedCountFormatted} passages</span>
                  <span className="status-badge"><Cpu size={13} /> Hybrid Search (BM25 + SBERT)</span>
                  <span className="status-badge"><Zap size={13} /> {acceleratorName}</span>
                </div>
              }
            />

            {/* Diagnostic Query Card */}
            <section className="search-card">
              <form onSubmit={(e) => { e.preventDefault(); executeTroubleshoot(); }}>
                <div className="search-input-header">
                  <label htmlFor="diagnostic-query">What error or technical issue are you facing?</label>
                  <span className="input-tip">Press Enter to diagnose</span>
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
                    placeholder="Enter an error message or terminal stack trace, e.g. 'Python raises ModuleNotFoundError: No module named fastapi'..."
                  />
                </div>

                {/* Preset Chips */}
                <div className="preset-strip">
                  <span className="preset-label">Quick presets:</span>
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

                {/* Search Options & Submit */}
                <div className="search-controls">
                  <div className="control-group">
                    <label>Technology Filter</label>
                    <select value={selectedTech} onChange={(e) => setSelectedTech(e.target.value)}>
                      {technologyFilters.map((t) => (
                        <option key={t} value={t}>{t}</option>
                      ))}
                    </select>
                  </div>

                  <div className="control-group">
                    <label>Retrieval Scheme</label>
                    <select value={retrievalMode} onChange={(e) => setRetrievalMode(e.target.value)}>
                      <option value="Hybrid">Hybrid (BM25 + Dense RRF)</option>
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
                    <span>{isSearching ? "Diagnosing..." : "Diagnose Issue"}</span>
                  </button>
                </div>
              </form>
            </section>

            {searchErrorMsg && (
              <div className="alert alert-error">
                <AlertTriangle size={16} />
                <span>{searchErrorMsg}</span>
              </div>
            )}

            {isSearching && (
              <div className="loading-state">
                <LoaderCircle className="spinner" size={30} />
                <p>Analyzing syntax, querying hybrid indices, and synthesizing fix...</p>
              </div>
            )}

            {/* AI Synthesized Diagnostic Card */}
            {diagnosticSolution && (
              <section className="diagnostic-card">
                <div className="card-top-bar">
                  <div className="badge-group">
                    <span className="badge badge-accent"><Sparkles size={13} /> Diagnostic Resolution</span>
                    {linguisticAnalysis?.severity && (
                      <span className={`badge severity-${linguisticAnalysis.severity.level}`}>
                        <AlertTriangle size={12} /> {linguisticAnalysis.severity.level.replace("_", " ")}
                      </span>
                    )}
                    <span className="badge badge-neutral"><Cpu size={12} /> {diagnosticSolution.model_used}</span>
                  </div>

                  <button
                    className={`btn btn-sm ${showHindiView ? "btn-active" : "btn-secondary"}`}
                    onClick={toggleHindiTranslation}
                    disabled={isTranslating}
                  >
                    <Languages size={14} />
                    <span>{isTranslating ? "Translating..." : showHindiView ? "English View" : "हिंदी में समझें (Hindi)"}</span>
                  </button>
                </div>

                {/* Multilingual View */}
                {showHindiView && hindiSolution && (
                  <div className="translation-panel">
                    <div className="translation-heading">
                      <Languages size={15} />
                      <strong>हिंदी में विस्तृत समाधान ({hindiSolution.title_hi || "Neural Diagnostic Resolution"}):</strong>
                    </div>
                    {hindiSolution.root_cause_hi && (
                      <div style={{ marginBottom: "12px" }}>
                        <span style={{ fontSize: "11px", fontWeight: "800", textTransform: "uppercase", letterSpacing: "0.04em", color: "#92400e" }}>
                          समस्या का मूल कारण (Root Cause):
                        </span>
                        <p className="translation-body" style={{ marginTop: "4px" }}>
                          {hindiSolution.root_cause_hi}
                        </p>
                      </div>
                    )}
                    {hindiSolution.action_steps_hi?.length > 0 && (
                      <div>
                        <span style={{ fontSize: "11px", fontWeight: "800", textTransform: "uppercase", letterSpacing: "0.04em", color: "#92400e" }}>
                          निवारण के मुख्य चरण (Step-by-Step Resolution):
                        </span>
                        <ol className="step-numbered-list" style={{ marginTop: "4px", color: "#451a03" }}>
                          {hindiSolution.action_steps_hi.map((step, idx) => (
                            <li key={idx}>{step}</li>
                          ))}
                        </ol>
                      </div>
                    )}
                    {!hindiSolution.root_cause_hi && !hindiSolution.action_steps_hi && (
                      <p className="translation-body">{typeof hindiSolution === "string" ? hindiSolution : hindiSolution.translated_text}</p>
                    )}
                  </div>
                )}

                {/* Root Cause */}
                <div className="diag-section">
                  <h3 className="section-title">Root Cause</h3>
                  <p className="root-cause-text">{diagnosticSolution.root_cause}</p>
                </div>

                {/* Actionable Steps */}
                {diagnosticSolution.action_steps?.length > 0 && (
                  <div className="diag-section">
                    <h3 className="section-title">Actionable Resolution</h3>
                    <ol className="step-numbered-list">
                      {diagnosticSolution.action_steps.map((step, idx) => (
                        <li key={idx}>{step}</li>
                      ))}
                    </ol>
                  </div>
                )}

                {/* Code Fix */}
                {diagnosticSolution.code_snippet && (
                  <div className="code-box">
                    <div className="code-box-header">
                      <span><Terminal size={13} /> Verified Command / Code Fix</span>
                      <button className="btn-copy" onClick={() => handleCopyCode(diagnosticSolution.code_snippet)}>
                        {copiedCodeSuccess ? <CheckCheck size={13} /> : <Copy size={13} />}
                        <span>{copiedCodeSuccess ? "Copied" : "Copy"}</span>
                      </button>
                    </div>
                    <pre><code>{diagnosticSolution.code_snippet}</code></pre>
                  </div>
                )}

                <div className="card-bottom-bar">
                  <span>Synthesized from reference candidate #{searchResults[0]?.chunk_id}</span>
                  {searchResults[0]?.url && (
                    <a href={searchResults[0].url} target="_blank" rel="noreferrer" className="external-link">
                      <span>View Verified Thread</span>
                      <ArrowUpRight size={13} />
                    </a>
                  )}
                </div>
              </section>
            )}

            {/* Empty Search State */}
            {hasSearched && searchResults.length === 0 && !isSearching && (
              <div className="empty-state">
                <CircleHelp size={32} />
                <h3>No matching solutions found</h3>
                <p>Try rephrasing the error description or switching the technology filter to "All technologies".</p>
              </div>
            )}

            {/* Ranked Evidence Results */}
            {searchResults.length > 0 && (
              <section className="evidence-section">
                <div className="evidence-header">
                  <div>
                    <h2>Referenced Technical Documentation</h2>
                    <p>{searchResults.length} source passages ranked by relevance scores.</p>
                  </div>
                  <button
                    className="btn btn-sm btn-secondary"
                    onClick={() => navigateToAnalyzerWithQuery(searchQuery)}
                  >
                    <Layers size={14} />
                    <span>Open in Error Analyzer</span>
                  </button>
                </div>

                <div className="evidence-list">
                  {searchResults.map((item, idx) => (
                    <article className={`evidence-card ${idx === 0 ? "primary-evidence" : ""}`} key={item.chunk_id}>
                      <div className="evidence-card-header">
                        <span className="rank-num">#{idx + 1}</span>
                        <div className="title-area">
                          <span className="source-name">{item.source} {idx === 0 && "· Primary Reference (Synthesized Above)"}</span>
                          <h4>{item.title}</h4>
                        </div>
                        <span className="score-badge">{(item.score * 100).toFixed(1)}% Match</span>
                      </div>

                      <p className="passage-body">{item.text}</p>

                      <div className="score-pills-row">
                        <span className="pill">BM25: {item.bm25_score?.toFixed(2) || "0.00"}</span>
                        <span className="pill">TF-IDF: {item.tfidf_score?.toFixed(2) || "0.00"}</span>
                        <span className="pill">BERT: {item.semantic_score?.toFixed(2) || "0.00"}</span>
                        {item.matched_terms?.length > 0 && (
                          <span className="pill pill-highlight">Matched: {item.matched_terms.slice(0, 3).join(", ")}</span>
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
        )}

        {/* ========================================================================= */}
        {/* TAB 2: ERROR ANALYZER */}
        {/* ========================================================================= */}
        {activeTab === "analyzer" && (
          <div className="tab-pane">
            <PageHeader
              eyebrow="LINGUISTIC INSPECTION"
              title="Error Syntax Analyzer"
              description="Inspect tokens, grammatical parts of speech, morphological roots, and automated urgency classification."
            />

            <section className="search-card">
              <label htmlFor="analyzer-text" style={{ fontSize: "14px", fontWeight: "700", display: "block", marginBottom: "10px" }}>
                Input error message, log line, or query
              </label>
              <div className="textarea-wrapper">
                <textarea
                  id="analyzer-text"
                  rows={3}
                  value={analyzerInput}
                  onChange={(e) => setAnalyzerInput(e.target.value)}
                  placeholder="Paste an error or terminal log..."
                />
              </div>

              <div className="preset-strip">
                <span className="preset-label">Presets:</span>
                <div className="preset-buttons">
                  <button
                    type="button"
                    className="chip-btn"
                    onClick={() => {
                      const sample = "Python raises ModuleNotFoundError: No module named 'fastapi' when executing main.py";
                      setAnalyzerInput(sample);
                      executeLinguisticAnalysis(sample);
                    }}
                  >
                    Python ModuleNotFound
                  </button>
                  <button
                    type="button"
                    className="chip-btn"
                    onClick={() => {
                      const sample = "NullPointerException in Java Spring Boot controller when finding user id";
                      setAnalyzerInput(sample);
                      executeLinguisticAnalysis(sample);
                    }}
                  >
                    Java NullPointer
                  </button>
                  <button
                    type="button"
                    className="chip-btn"
                    onClick={() => {
                      const sample = "Docker daemon failed with error: bind address already in use on port 8080";
                      setAnalyzerInput(sample);
                      executeLinguisticAnalysis(sample);
                    }}
                  >
                    Docker Port Conflict
                  </button>
                </div>
              </div>

              <div style={{ marginTop: "16px" }}>
                <button
                  className="btn btn-primary"
                  onClick={() => executeLinguisticAnalysis(analyzerInput)}
                  disabled={isAnalyzing || !analyzerInput.trim()}
                >
                  {isAnalyzing ? <LoaderCircle className="spinner" size={16} /> : <Layers size={16} />}
                  <span>{isAnalyzing ? "Analyzing..." : "Analyze Syntax"}</span>
                </button>
              </div>
            </section>

            {analyzerOutput && (
              <div className="analyzer-grid">
                {/* Metrics */}
                <div className="metrics-strip">
                  <MetricCard icon={Layers} label="Total Tokens" value={analyzerOutput.token_count} detail="Word tokens" />
                  <MetricCard icon={Check} label="Content Words" value={analyzerOutput.content_token_count} detail="Meaningful tokens" />
                  <MetricCard icon={X} label="Filtered Stopwords" value={analyzerOutput.stopword_count} detail="Syntactic noise" />
                  <MetricCard icon={AlertTriangle} label="Error Severity" value={analyzerOutput.severity?.level || "N/A"} detail={`Confidence: ${Math.round((analyzerOutput.severity?.confidence || 0) * 100)}%`} />
                </div>

                {/* Part of Speech Tagging */}
                <section className="detail-card">
                  <div className="detail-header">
                    <h3>Part-of-Speech (POS) Tagging</h3>
                    <p>Token-level grammatical tagging using Penn Treebank tags.</p>
                  </div>
                  <div className="tag-chips-wrapper">
                    {analyzerOutput.pos_tags?.map((item, idx) => (
                      <div className={`pos-badge pos-${item.category.toLowerCase().replace(/[^a-z]/g, "")}`} key={idx} title={item.description}>
                        <span className="pos-word">{item.word}</span>
                        <span className="pos-tag-code">{item.tag}</span>
                      </div>
                    ))}
                  </div>
                  <div className="pos-legend-bar">
                    <span><i className="legend-dot noun" /> Noun</span>
                    <span><i className="legend-dot verb" /> Verb</span>
                    <span><i className="legend-dot adjective" /> Adjective</span>
                    <span><i className="legend-dot adverb" /> Adverb</span>
                    <span><i className="legend-dot prep" /> Preposition</span>
                    <span><i className="legend-dot other" /> Pronoun / Conjunction</span>
                  </div>
                </section>

                {/* Stemming vs Lemmatization */}
                <section className="detail-card">
                  <div className="detail-header">
                    <h3>Stemming vs. Lemmatization</h3>
                    <p>Comparison between heuristic suffix truncation and vocabulary root normalization.</p>
                  </div>
                  <div className="table-responsive">
                    <table className="clean-table">
                      <thead>
                        <tr>
                          <th>Original Word</th>
                          <th>Porter Stem (Stemming)</th>
                          <th>WordNet Lemma (Lemmatization)</th>
                          <th>Rule Category</th>
                        </tr>
                      </thead>
                      <tbody>
                        {analyzerOutput.lemmas_vs_stems?.map((row, idx) => (
                          <tr key={idx}>
                            <td><strong>{row.original}</strong></td>
                            <td className="mono">{row.porter_stem}</td>
                            <td className="mono text-highlight">{row.wordnet_lemma}</td>
                            <td><span className="chip-badge">{row.rule_applied}</span></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>

                {/* Technical Entity Extraction */}
                <section className="detail-card">
                  <div className="detail-header">
                    <h3>Technical Entity Recognition (NER)</h3>
                    <p>Automated identification of programming frameworks, exception types, and identifiers.</p>
                  </div>
                  <div className="ner-boxes-row">
                    <div className="ner-box">
                      <h4>Frameworks & Stacks</h4>
                      <div className="ner-chips">
                        {analyzerOutput.entities?.technologies?.length > 0 ? (
                          analyzerOutput.entities.technologies.map((t) => <span className="chip" key={t}>{t}</span>)
                        ) : <span className="muted-text">None detected</span>}
                      </div>
                    </div>
                    <div className="ner-box">
                      <h4>Exception & Error Types</h4>
                      <div className="ner-chips">
                        {analyzerOutput.entities?.errors?.length > 0 ? (
                          analyzerOutput.entities.errors.map((e) => <span className="chip chip-danger" key={e}>{e}</span>)
                        ) : <span className="muted-text">None detected</span>}
                      </div>
                    </div>
                    <div className="ner-box">
                      <h4>Code Identifiers</h4>
                      <div className="ner-chips">
                        {analyzerOutput.entities?.code_identifiers?.length > 0 ? (
                          analyzerOutput.entities.code_identifiers.map((c) => <span className="chip chip-code" key={c}>{c}</span>)
                        ) : <span className="muted-text">None detected</span>}
                      </div>
                    </div>
                  </div>
                </section>
              </div>
            )}
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 3: KNOWLEDGE BASE */}
        {/* ========================================================================= */}
        {activeTab === "knowledge" && (
          <div className="tab-pane">
            <PageHeader
              eyebrow="TECHNICAL REPOSITORY"
              title="Knowledge Base"
              description="Browse through indexed solutions or upload proprietary guides, runbooks, and documents."
            />

            <div className="library-actions-row">
              <div className="library-search-box">
                <Search size={16} />
                <input
                  type="text"
                  value={librarySearchTerm}
                  onChange={(e) => setLibrarySearchTerm(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && fetchLibraryPassages(librarySearchTerm)}
                  placeholder="Filter by title, technology, or keywords..."
                />
                <button className="btn btn-sm btn-primary" onClick={() => fetchLibraryPassages(librarySearchTerm)}>
                  Search
                </button>
              </div>

              <label className="btn btn-primary upload-label">
                <Upload size={15} />
                <span>{isUploading ? "Uploading..." : "Upload Document"}</span>
                <input
                  type="file"
                  multiple
                  accept=".txt,.md,.markdown,.pdf,.csv,.json"
                  onChange={(e) => handleDocumentUpload(e.target.files)}
                  disabled={isUploading}
                />
              </label>
            </div>

            <div className="library-meta-strip">
              <span><Database size={14} /> {libraryData.total.toLocaleString()} total passages</span>
              <span>Supported formats: TXT, Markdown, PDF, CSV, JSON</span>
            </div>

            {isLibraryLoading ? (
              <div className="loading-state">
                <LoaderCircle className="spinner" size={28} />
                <p>Loading documentation passages...</p>
              </div>
            ) : (
              <div className="library-grid">
                {libraryData.items.map((item) => (
                  <article key={item.chunk_id} className="library-card">
                    <div className="library-card-icon"><FileText size={18} /></div>
                    <div className="library-card-body">
                      <span className="library-category">{item.category || item.source}</span>
                      <h4>{item.title}</h4>
                      <p>{item.text}</p>
                      <div className="tag-chips">
                        {(item.technologies || []).slice(0, 4).map((tech) => (
                          <span className="chip" key={tech}>{tech}</span>
                        ))}
                      </div>
                    </div>
                    {item.url && (
                      <a href={item.url} target="_blank" rel="noreferrer" className="card-corner-link">
                        <ArrowUpRight size={15} />
                      </a>
                    )}
                  </article>
                ))}
              </div>
            )}

            {!isLibraryLoading && libraryData.items.length < libraryData.total && (
              <div className="load-more-row">
                <button className="btn btn-secondary" onClick={() => fetchLibraryPassages(librarySearchTerm, true)}>
                  <span>Load more passages</span>
                  <ArrowDownRight size={15} />
                </button>
              </div>
            )}
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 4: QUERY HISTORY */}
        {/* ========================================================================= */}
        {activeTab === "history" && (
          <div className="tab-pane">
            <PageHeader
              eyebrow="AUDIT TRAIL"
              title="Query History"
              description="Review past troubleshooting searches, record relevance feedback, or rerun any inquiry."
            />

            {searchHistoryList.length > 0 ? (
              <div className="history-list">
                {searchHistoryList.map((item, idx) => (
                  <article className="history-row" key={`${item.query}-${idx}`}>
                    <div className="history-icon-box"><MessageSquareText size={18} /></div>
                    <div className="history-content">
                      <h4>{item.query}</h4>
                      <span className="history-meta">{item.result_ids?.length || 0} retrieved passages · Status: {item.feedback || "Feedback pending"}</span>
                    </div>
                    <div className="history-actions">
                      <button
                        className="feedback-btn"
                        onClick={() => handleRerunQuery(item.query)}
                        title="Rerun this diagnostic inquiry in Troubleshoot"
                      >
                        <Play size={13} />
                        <span>Rerun</span>
                      </button>
                      <button
                        className={`feedback-btn ${item.feedback === "Useful" ? "active-useful" : ""}`}
                        onClick={() => handleFeedback(idx, "Useful")}
                      >
                        <Check size={14} />
                        <span>Useful</span>
                      </button>
                      <button
                        className={`feedback-btn ${item.feedback === "Not useful" ? "active-not-useful" : ""}`}
                        onClick={() => handleFeedback(idx, "Not useful")}
                      >
                        <X size={14} />
                        <span>Not useful</span>
                      </button>
                    </div>
                  </article>
                ))}
              </div>
            ) : (
              <div className="empty-state">
                <History size={32} />
                <h3>No queries recorded yet</h3>
                <p>Troubleshoot a technical error to see search history here.</p>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Clean Footer */}
      <footer className="footer">
        <div className="footer-container">
          <div className="footer-left">
            <strong>FixItAI</strong>
            <span>Technical Diagnostic Intelligence Platform</span>
          </div>
          <div className="footer-right">
            <span>Indexed Passages: {indexedCountFormatted}</span>
            <span className="separator">·</span>
            <span>{acceleratorName}</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
