import { useEffect, useState } from "react";
import { AlertTriangle, BookOpen, Check, History, Layers, Search, X } from "lucide-react";
import { Navbar } from "./components/Navbar";
import { TroubleshootTab } from "./components/TroubleshootTab";
import { AnalyzerTab } from "./components/AnalyzerTab";
import { KnowledgeBaseTab } from "./components/KnowledgeBaseTab";
import { HistoryTab } from "./components/HistoryTab";

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
  const [searchMetadata, setSearchMetadata] = useState(null);
  const [isSearching, setIsSearching] = useState(false);
  const [searchErrorMsg, setSearchErrorMsg] = useState("");

  // Translation State
  const [targetLang, setTargetLang] = useState("hi");
  const [localizeEntirePage, setLocalizeEntirePage] = useState(false);
  const [isTranslating, setIsTranslating] = useState(false);
  const [hindiSolution, setHindiSolution] = useState(null);
  const [showHindiView, setShowHindiView] = useState(false);

  // Linguistic Analyzer State
  const [analyzerInput, setAnalyzerInput] = useState(
    "Python raises ModuleNotFoundError: No module named 'fastapi' when executing main.py"
  );
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
      setLibraryData((prev) => (append ? { ...data, items: [...prev.items, ...data.items] } : data));
    } catch (err) {
      setGeneralError(err.message);
    } finally {
      setIsLibraryLoading(false);
    }
  }

  async function executeTroubleshoot(customQuery, wizardOption = null) {
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
    setSearchMetadata(null);

    // Keep previous translation state if target language is already active
    if (targetLang === "en") {
      setHindiSolution(null);
      setShowHindiView(false);
    }

    try {
      const data = await apiRequest("/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: q,
          technology: selectedTech === "All technologies" ? "All" : selectedTech,
          mode: "Hybrid",
          top_k: 5,
          wizard_option: wizardOption || null,
          target_lang: targetLang,
        }),
      });
      setSearchResults(data.results || []);
      setLinguisticAnalysis(data.analysis || null);
      setDiagnosticSolution(data.ai_solution || null);
      if (data.translated_solution) {
        setHindiSolution(data.translated_solution);
        if (targetLang !== "en") {
          setShowHindiView(true);
        }
      } else if (targetLang !== "en" && data.ai_solution) {
        handleTranslateToLang(targetLang, data.parsed_trace?.clean_query || q, data.ai_solution);
      }
      setSearchMetadata({
        latency_ms: data.latency_ms,
        device: data.device,
        retrieval_mode: "Hybrid",
        confidence: data.confidence,
        guardrail: data.guardrail,
        typo_correction: data.typo_correction,
        wordnet_expansion: data.wordnet_expansion,
        perplexity: data.perplexity,
        wizard_questions: data.wizard_questions,
        indic_meta: data.indic_meta,
        parsed_trace: data.parsed_trace,
      });
    } catch (err) {
      setSearchErrorMsg(err.message);
    } finally {
      setIsSearching(false);
    }
  }

  async function handleTranslateToLang(langCode, queryOverride = null, solutionOverride = null) {
    const sol = solutionOverride || diagnosticSolution;
    if (!sol) return;
    const reqLang = langCode || targetLang;
    if (reqLang === "en") {
      setShowHindiView(false);
      return;
    }
    setIsTranslating(true);
    try {
      const qText = queryOverride || searchMetadata?.parsed_trace?.clean_query || searchQuery;
      let cleanQ = qText;
      if (cleanQ.includes("\n")) {
        const lines = cleanQ.trim().split("\n");
        const excLine = lines.slice().reverse().find(l => l.includes("Error:") || l.includes("Exception:") || l.includes("Exception"));
        cleanQ = excLine ? excLine.trim() : lines[lines.length - 1].trim();
      }
      const data = await apiRequest("/translate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: cleanQ,
          root_cause: sol.root_cause || "",
          action_steps: sol.action_steps || [],
          code_snippet: sol.code_snippet || "",
          text: sol.root_cause || "",
          target_lang: reqLang,
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

  async function toggleHindiTranslation() {
    if (!diagnosticSolution) return;
    if (showHindiView && targetLang !== "en") {
      setShowHindiView(false);
      return;
    }
    const effectiveLang = targetLang === "en" ? "hi" : targetLang;
    if (targetLang === "en") {
      setTargetLang("hi");
    }
    if (hindiSolution && (hindiSolution.language_code === effectiveLang || hindiSolution.language === effectiveLang)) {
      setShowHindiView(true);
      return;
    }
    await handleTranslateToLang(effectiveLang);
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

  async function handleClearHistory() {
    try {
      await apiRequest("/history/clear", { method: "POST" });
      setSearchHistoryList([]);
      setFeedbackNotice("History cleared successfully.");
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

  const acceleratorName = systemHealth?.gpu || "Vector Acceleration";
  const indexedCountFormatted = systemHealth?.indexed_passages?.toLocaleString() || "173,866";

  return (
    <div className="product-layout">
      {/* Top Navigation */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        navigationItems={navigationItems}
        systemHealth={systemHealth}
        targetLang={targetLang}
        localizeEntirePage={localizeEntirePage}
      />

      {/* Main Content Area */}
      <main className="content-container">
        {feedbackNotice && (
          <div className="alert alert-success">
            <Check size={16} />
            <span>{feedbackNotice}</span>
            <button className="alert-close" onClick={() => setFeedbackNotice("")}>
              <X size={14} />
            </button>
          </div>
        )}
        {generalError && (
          <div className="alert alert-error">
            <AlertTriangle size={16} />
            <span>{generalError}</span>
            <button className="alert-close" onClick={() => setGeneralError("")}>
              <X size={14} />
            </button>
          </div>
        )}

        {/* Tab 1: Troubleshoot */}
        {activeTab === "troubleshoot" && (
          <TroubleshootTab
            systemHealth={systemHealth}
            searchQuery={searchQuery}
            setSearchQuery={setSearchQuery}
            selectedTech={selectedTech}
            setSelectedTech={setSelectedTech}
            technologyFilters={technologyFilters}
            retrievalMode={retrievalMode}
            setRetrievalMode={setRetrievalMode}
            presetExamples={presetExamples}
            executeTroubleshoot={executeTroubleshoot}
            isSearching={isSearching}
            searchErrorMsg={searchErrorMsg}
            hasSearched={hasSearched}
            searchResults={searchResults}
            diagnosticSolution={diagnosticSolution}
            setDiagnosticSolution={setDiagnosticSolution}
            setSearchResults={setSearchResults}
            linguisticAnalysis={linguisticAnalysis}
            showHindiView={showHindiView}
            setShowHindiView={setShowHindiView}
            hindiSolution={hindiSolution}
            setHindiSolution={setHindiSolution}
            isTranslating={isTranslating}
            toggleHindiTranslation={toggleHindiTranslation}
            searchMetadata={searchMetadata}
            navigateToAnalyzerWithQuery={navigateToAnalyzerWithQuery}
            targetLang={targetLang}
            setTargetLang={setTargetLang}
            handleTranslateToLang={handleTranslateToLang}
            localizeEntirePage={localizeEntirePage}
            setLocalizeEntirePage={setLocalizeEntirePage}
          />
        )}

        {/* Tab 2: Error Analyzer */}
        {activeTab === "analyzer" && (
          <AnalyzerTab
            analyzerInput={analyzerInput}
            setAnalyzerInput={setAnalyzerInput}
            executeLinguisticAnalysis={executeLinguisticAnalysis}
            isAnalyzing={isAnalyzing}
            analyzerOutput={analyzerOutput}
          />
        )}

        {/* Tab 3: Knowledge Base */}
        {activeTab === "knowledge" && (
          <KnowledgeBaseTab
            librarySearchTerm={librarySearchTerm}
            setLibrarySearchTerm={setLibrarySearchTerm}
            fetchLibraryPassages={fetchLibraryPassages}
            isLibraryLoading={isLibraryLoading}
            isUploading={isUploading}
            handleDocumentUpload={handleDocumentUpload}
            libraryData={libraryData}
          />
        )}

        {/* Tab 4: Query History */}
        {activeTab === "history" && (
          <HistoryTab
            searchHistoryList={searchHistoryList}
            handleRerunQuery={handleRerunQuery}
            handleFeedback={handleFeedback}
            handleClearHistory={handleClearHistory}
          />
        )}
      </main>

      {/* Footer */}
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
