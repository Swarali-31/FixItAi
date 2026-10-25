import { useEffect, useMemo, useState } from "react";
import {
  ArrowDownRight, ArrowLeft, ArrowRight, ArrowUpRight, BookOpen, Check,
  ChevronDown, CircleHelp, Cpu, Database, FileText, Gauge, History,
  Info, LoaderCircle, MessageSquareText, Search, ShieldCheck, Sparkles,
  Upload, X,
} from "lucide-react";

const pages = [
  { id: "ask", label: "Ask FixItAI", icon: Search, description: "Troubleshoot a technical issue" },
  { id: "knowledge", label: "Knowledge base", icon: BookOpen, description: "Explore and add technical notes" },
  { id: "evaluation", label: "Model evaluation", icon: Gauge, description: "See training and retrieval results" },
  { id: "history", label: "Search history", icon: History, description: "Review searches and leave feedback" },
  { id: "about", label: "About the project", icon: Info, description: "Learn how FixItAI works" },
];

const examples = [
  "Java NullPointerException in a service method",
  "Docker says the port is already allocated",
  "Python raises ModuleNotFoundError after install",
  "MySQL connection timeout from a Java app",
  "I am not receiving Microsoft Teams notifications",
];
const technologies = ["All technologies", "Java", "Python", "SQL", "MySQL", "Docker", "Linux", "Windows", "Git", "HTTP", "DNS", "Kubernetes"];

async function request(path, options = {}) {
  const response = await fetch(`/api${path}`, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}

function PageHeading({ eyebrow, title, copy, action }) {
  return (
    <div className="page-heading-row">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        {copy && <p>{copy}</p>}
      </div>
      {action}
    </div>
  );
}

function StatCard({ icon: Icon, label, value, detail }) {
  return (
    <div className="stat-card">
      <div className="stat-icon"><Icon size={18} /></div>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {detail && <div className="stat-detail">{detail}</div>}
    </div>
  );
}

function App() {
  const [page, setPage] = useState("home");
  const [health, setHealth] = useState(null);
  const [query, setQuery] = useState("");
  const [technology, setTechnology] = useState("All technologies");
  const [mode, setMode] = useState("Hybrid");
  const [results, setResults] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState("");
  const [library, setLibrary] = useState({ items: [], total: 0 });
  const [libraryFilter, setLibraryFilter] = useState("");
  const [libraryLoading, setLibraryLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [notice, setNotice] = useState("");
  const [evaluation, setEvaluation] = useState(null);
  const [history, setHistory] = useState([]);
  const [pageError, setPageError] = useState("");

  useEffect(() => {
    request("/health").then(setHealth).catch(() => setHealth({ status: "offline" }));
  }, []);

  useEffect(() => {
    setPageError("");
    if (page === "evaluation") request("/evaluation").then(setEvaluation).catch((error) => setPageError(error.message));
    if (page === "history") request("/history").then((data) => setHistory(data.items || [])).catch((error) => setPageError(error.message));
    if (page === "knowledge") loadLibrary("");
  }, [page]);

  async function loadLibrary(filter = libraryFilter, append = false) {
    setLibraryLoading(true);
    try {
      const offset = append ? library.items.length : 0;
      const params = new URLSearchParams({ search_text: filter, offset: String(offset), limit: "60" });
      const data = await request("/knowledge-base?" + params.toString());
      setLibrary((current) => append ? { ...data, items: [...current.items, ...data.items] } : data);
      setPageError("");
    } catch (error) {
      setPageError(error.message);
    } finally {
      setLibraryLoading(false);
    }
  }

  async function runSearch(event) {
    event?.preventDefault();
    if (!query.trim()) return;
    setSearching(true);
    setSearchError("");
    setResults([]);
    try {
      const data = await request("/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query,
          technology: technology === "All technologies" ? "All" : technology,
          mode,
          top_k: 6,
        }),
      });
      setResults(data.results || []);
      setAnalysis(data.analysis || null);
    } catch (error) {
      setSearchError(error.message);
    } finally {
      setSearching(false);
    }
  }

  async function uploadFiles(files) {
    if (!files?.length) return;
    setUploading(true);
    setNotice("");
    const form = new FormData();
    [...files].forEach((file) => form.append("files", file));
    try {
      const data = await request("/knowledge-base/upload", { method: "POST", body: form });
      setNotice(data.message);
      await loadLibrary(libraryFilter);
    } catch (error) {
      setPageError(error.message);
    } finally {
      setUploading(false);
    }
  }

  async function saveFeedback(index, feedback) {
    try {
      await request("/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ index_from_newest: index, feedback }),
      });
      const data = await request("/history");
      setHistory(data.items || []);
      setNotice("Feedback saved.");
    } catch (error) {
      setPageError(error.message);
    }
  }

  const activePage = pages.find((item) => item.id === page);
  const gpuLabel = health?.gpu || "CUDA model";
  const passageCount = health?.indexed_passages?.toLocaleString() || "...";

  return (
    <div className="app-shell">
      <header className="topbar">
        <button className="brand" onClick={() => setPage("home")} aria-label="FixItAI home">
          <span className="brand-mark"><Sparkles size={20} /></span>
          <span className="brand-copy"><strong>FixItAI</strong><small>Technical support workspace</small></span>
        </button>
        <nav className="top-nav" aria-label="Main navigation">
          <button className={page === "home" ? "nav-link active" : "nav-link"} onClick={() => setPage("home")}>Home</button>
          {pages.map(({ id, label }) => (
            <button key={id} className={page === id ? "nav-link active" : "nav-link"} onClick={() => setPage(id)}>{label}</button>
          ))}
        </nav>
        <div className="top-status"><span className={health?.status === "ready" ? "status-dot ready" : "status-dot"} />{health?.status === "ready" ? "Model ready" : health?.status === "offline" ? "API offline" : "Connecting"}</div>
      </header>

      <main className="main-content">
        {page === "home" && (
          <>
            <section className="home-hero">
              <div className="hero-glow" />
              <div className="hero-content">
                <div className="hero-eyebrow"><span className="hero-eyebrow-dot" /> LOCAL AI FOR TECHNICAL SUPPORT</div>
                <h1>Get unstuck.<br /><span>Understand why.</span></h1>
                <p>Find practical troubleshooting guidance, see the evidence behind every result, and keep your technical knowledge organized.</p>
                <div className="hero-actions">
                  <button className="button button-white" onClick={() => setPage("ask")}>Ask FixItAI <ArrowRight size={17} /></button>
                  <button className="button button-ghost" onClick={() => setPage("knowledge")}>Explore knowledge base</button>
                </div>
              </div>
              <div className="hero-visual" aria-hidden="true">
                <div className="orbit orbit-one" /><div className="orbit orbit-two" />
                <div className="visual-core"><Sparkles size={35} /></div>
                <div className="float-card float-card-top"><Search size={15} /> Error context</div>
                <div className="float-card float-card-bottom"><ShieldCheck size={15} /> Source-backed results</div>
              </div>
              <div className="hero-bottom"><span><Cpu size={15} /> BERT Transformer encoder</span><span><Database size={15} /> Runs locally on your GPU</span><span><ShieldCheck size={15} /> Sources shown with results</span></div>
            </section>

            <section className="home-stats">
              <StatCard icon={Database} label="Indexed passages" value={passageCount} detail="Technical Q&A and local notes" />
              <StatCard icon={Cpu} label="Model" value="BERT encoder" detail={health?.pretrained ? "Fine-tuned pretrained BERT" : "BERT trained from scratch"} />
              <StatCard icon={Gauge} label="GPU" value={gpuLabel.replace("NVIDIA GeForce ", "")} detail="CUDA training and inference" />
            </section>

            <section className="workspace-section">
              <div className="section-heading"><div><div className="eyebrow">YOUR WORKSPACE</div><h2>Where do you want to go?</h2><p>Choose a section to get started.</p></div><span className="section-count">05 sections</span></div>
              <div className="workspace-grid">
                {pages.map(({ id, label, icon: Icon, description }, index) => (
                  <button className="workspace-card" key={id} onClick={() => setPage(id)}>
                    <div className="workspace-card-top"><span className="workspace-icon"><Icon size={20} /></span><span className="workspace-number">0{index + 1}</span></div>
                    <h3>{label}</h3><p>{description}</p>
                    <span className="card-link">Open section <ArrowUpRight size={16} /></span>
                  </button>
                ))}
              </div>
            </section>
          </>
        )}

        {page === "ask" && (
          <>
            <PageHeading eyebrow="TECHNICAL TROUBLESHOOTING" title="Ask FixItAI" copy="Ask a general tech question or describe a problem. FixItAI finds relevant indexed guidance and shows its sources." action={<button className="button button-light" onClick={() => setPage("home")}><ArrowLeft size={16} /> Home</button>} />
            <div className="search-layout">
              <section className="search-panel">
                <form onSubmit={runSearch}>
                  <label className="field-label" htmlFor="question">What tech question can I help with?</label>
                  <textarea id="question" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Ask a question, for example: “Why am I not receiving Teams notifications?” Add your device or app version for more specific guidance." rows={5} />
                  <div className="example-row"><span>Try:</span>{examples.map((example) => <button type="button" key={example} onClick={() => setQuery(example)}>{example}</button>)}</div>
                  <div className="search-options">
                    <label>Technology<select value={technology} onChange={(event) => setTechnology(event.target.value)}>{technologies.map((item) => <option key={item}>{item}</option>)}</select></label>
                    <label>Search mode<select value={mode} onChange={(event) => setMode(event.target.value)}>{["Hybrid", "Semantic", "BM25", "TF-IDF"].map((item) => <option key={item}>{item}</option>)}</select></label>
                    <button className="button button-primary search-submit" type="submit" disabled={searching || !query.trim()}>{searching ? <LoaderCircle className="spin" size={17} /> : <Search size={17} />}{searching ? "Searching..." : "Find solutions"}</button>
                  </div>
                </form>
              </section>
              <aside className="search-aside"><div className="aside-icon"><CircleHelp size={21} /></div><h3>Get better matches</h3><ul><li>Paste the complete error message.</li><li>Name the language, framework, or service.</li><li>Include what you already tried.</li></ul><div className="aside-note"><ShieldCheck size={16} /> Results include their source and attribution.</div></aside>
            </div>
            {searchError && <div className="notice error"><X size={17} />{searchError}</div>}
            {searching && <div className="loading-row"><LoaderCircle className="spin" /> Searching the CUDA model and technical guides...</div>}
            {results.length > 0 && <section className="results-section"><div className="section-heading"><div><div className="eyebrow">MATCHED EVIDENCE</div><h2>Recommended sources</h2><p>{results.length} passages matched your question.</p></div>{analysis && <details className="analysis"><summary>Query signals <ChevronDown size={14} /></summary><pre>{JSON.stringify(analysis, null, 2)}</pre></details>}</div>
              <article className="answer-card"><div className="eyebrow">BEST MATCHING GUIDANCE</div><h3>{results[0].title}</h3><p>{results[0].text}</p><div className="answer-source">From {results[0].source}{results[0].url && <> · <a href={results[0].url} target="_blank" rel="noreferrer">Open source <ArrowUpRight size={13} /></a></>}</div></article>
              {results[0].score < 0.2 && <div className="notice warning">The match is weak. These suggestions may not fit your question; check the linked source and share more details to narrow it down.</div>}
              <div className="result-list">{results.map((result, index) => <article className="result-card" key={result.chunk_id}>
                <div className="result-card-head"><span className="result-rank">{String(index + 1).padStart(2, "0")}</span><div><div className="result-source">{result.source}</div><h3>{result.title}</h3></div><span className="score-pill">{result.score.toFixed(3)} match</span></div>
                <p className="result-text">{result.text}</p>
                <div className="result-meta"><span>TF-IDF {result.tfidf_score.toFixed(2)}</span><span>BM25 {result.bm25_score.toFixed(2)}</span><span>BERT similarity {result.semantic_score.toFixed(2)}</span></div>
                <div className="result-actions">{result.url && <a href={result.url} target="_blank" rel="noreferrer">Open answer source <ArrowUpRight size={14} /></a>}{result.question_url && <a href={result.question_url} target="_blank" rel="noreferrer">View question <ArrowUpRight size={14} /></a>}</div>
                {result.attribution && <details className="attribution"><summary>Attribution details</summary><p>{result.attribution}</p></details>}
              </article>)}</div>
              <p className="disclaimer">Retrieved passages are community or project notes, not generated or independently verified diagnoses. Check commands against your own environment.</p>
            </section>}
          </>
        )}

        {page === "knowledge" && (
          <>
            <PageHeading eyebrow="YOUR TECHNICAL LIBRARY" title="Knowledge base" copy="Browse indexed passages and add technical documents you own or are authorized to use." action={<button className="button button-light" onClick={() => setPage("home")}><ArrowLeft size={16} /> Home</button>} />
            {notice && <div className="notice success"><Check size={17} />{notice}<button onClick={() => setNotice("")} aria-label="Dismiss"><X size={16} /></button></div>}
            {pageError && <div className="notice error"><X size={17} />{pageError}</div>}
            <div className="library-toolbar"><div className="library-search"><Search size={18} /><input value={libraryFilter} onChange={(event) => setLibraryFilter(event.target.value)} onKeyDown={(event) => event.key === "Enter" && loadLibrary(libraryFilter)} placeholder="Filter by title, source, technology, or passage..." /><button onClick={() => loadLibrary(libraryFilter)}>Search</button></div>
              <label className="button button-primary upload-button"><Upload size={17} />{uploading ? "Indexing..." : "Add documents"}<input type="file" multiple accept=".txt,.md,.markdown,.pdf,.csv,.json" onChange={(event) => uploadFiles(event.target.files)} disabled={uploading} /></label></div>
            <div className="library-summary"><span><Database size={16} /> {library.total.toLocaleString()} passages</span><span>Showing {library.items.length.toLocaleString()} of {library.total.toLocaleString()}</span><span>TXT, Markdown, PDF, CSV, JSON</span></div>
            {libraryLoading ? <div className="loading-row"><LoaderCircle className="spin" /> Loading the library...</div> : <div className="library-list">{library.items.map((item) => <article key={item.chunk_id} className="library-item"><div className="library-item-icon"><FileText size={18} /></div><div className="library-item-content"><div className="result-source">{item.category || item.source}</div><h3>{item.title}</h3><p>{item.text}</p><div className="tag-row">{(item.technologies || []).slice(0, 5).map((tag) => <span className="tag" key={tag}>{tag}</span>)}</div></div>{item.url && <a className="icon-link" href={item.url} target="_blank" rel="noreferrer" aria-label="Open source"><ArrowUpRight size={17} /></a>}</article>)}</div>}
            {!libraryLoading && library.items.length < library.total && <div className="library-more"><button className="button button-light" onClick={() => loadLibrary(libraryFilter, true)}>Load more passages <ArrowDownRight size={15} /></button></div>}
          </>
        )}

        {page === "evaluation" && (
          <>
            <PageHeading eyebrow="MODEL PERFORMANCE" title="Evaluation" copy="Held-out retrieval results for the BERT encoder trained for question–answer retrieval." action={<button className="button button-light" onClick={() => setPage("home")}><ArrowLeft size={16} /> Home</button>} />
            {pageError && <div className="notice error">{pageError}</div>}
            {evaluation?.validation ? <>
              <div className="evaluation-stats"><StatCard icon={Search} label="Recall at 1" value={`${(evaluation.validation.recall_at_1 * 100).toFixed(1)}%`} detail="Top result is the target" /><StatCard icon={Database} label="Recall at 5" value={`${(evaluation.validation.recall_at_5 * 100).toFixed(1)}%`} detail="Target in top five" /><StatCard icon={Gauge} label="MRR" value={evaluation.validation.mrr.toFixed(3)} detail="Mean reciprocal rank" /><StatCard icon={MessageSquareText} label="Validation queries" value={evaluation.validation.queries.toLocaleString()} detail={`${evaluation.validation.candidate_answers.toLocaleString()} answer candidates`} /></div>
              <section className="panel chart-panel"><div className="section-heading"><div><div className="eyebrow">TRAINING RUN</div><h2>Progress by epoch</h2><p>Training loss and held-out retrieval metrics.</p></div><span className="gpu-badge"><Cpu size={15} /> {evaluation.gpu}</span></div><TrainingChart rows={evaluation.all_epochs || []} /></section>
              <div className="notice warning"><ShieldCheck size={18} /><span>Labels use the highest-voted answer as a weak relevance signal. These metrics do not prove that answers are technically correct or current.</span></div>
            </> : !pageError ? <div className="empty-state"><Gauge size={28} /><h3>No evaluation report found</h3><p>Train the model with the supplied dataset to generate its held-out metrics.</p></div> : null}
          </>
        )}

        {page === "history" && (
          <>
            <PageHeading eyebrow="RECENT ACTIVITY" title="Search history" copy="Review your recent questions and mark whether the retrieved evidence was useful." action={<button className="button button-light" onClick={() => setPage("home")}><ArrowLeft size={16} /> Home</button>} />
            {pageError && <div className="notice error">{pageError}</div>}{notice && <div className="notice success"><Check size={16} />{notice}</div>}
            {history.length ? <div className="history-list">{history.map((item, index) => <article className="history-item" key={`${item.query}-${index}`}><div className="history-icon"><MessageSquareText size={18} /></div><div className="history-body"><h3>{item.query}</h3><p>{item.result_ids?.length || 0} source passages Â· {item.feedback || "Feedback not given"}</p></div><div className="history-feedback"><button className={item.feedback === "Useful" ? "feedback chosen" : "feedback"} onClick={() => saveFeedback(index, "Useful")}><Check size={15} /> Useful</button><button className={item.feedback === "Not useful" ? "feedback chosen" : "feedback"} onClick={() => saveFeedback(index, "Not useful")}><X size={15} /> Not useful</button></div></article>)}</div> : !pageError && <div className="empty-state"><History size={28} /><h3>No searches yet</h3><p>Your questions will show here after you use Ask FixItAI.</p><button className="button button-primary" onClick={() => setPage("ask")}>Ask a question <ArrowRight size={16} /></button></div>}
            <p className="disclaimer">History is stored locally by default. Configure FIXITAI_MONGODB_URI to save it in MongoDB.</p>
          </>
        )}

        {page === "about" && (
          <>
            <PageHeading eyebrow="THE PROJECT" title="About FixItAI" copy="A local technical troubleshooting and evidence-retrieval workspace." action={<button className="button button-light" onClick={() => setPage("home")}><ArrowLeft size={16} /> Home</button>} />
            <section className="about-hero"><div className="about-icon"><Sparkles size={25} /></div><h2>Find evidence. Make the next step clearer.</h2><p>FixItAI combines searchable technical documentation with a BERT Transformer encoder trained for question–answer retrieval. It returns passages and source links instead of generating unsupported fixes.</p></section>
            <div className="about-grid"><article className="panel"><Cpu size={20} /><h3>BERT retrieval encoder</h3><p>The bidirectional BERT encoder maps questions and answers to vectors. Its initialization status and training setup are reported with the model artifacts.</p></article><article className="panel"><Search size={20} /><h3>Hybrid retrieval</h3><p>TF-IDF and BM25 identify lexical matches; BERT adds dense semantic similarity to the ranking.</p></article><article className="panel"><ShieldCheck size={20} /><h3>Evidence with limits</h3><p>Stack Overflow answers are community contributions. Vote counts provide weak labels, not proof of correctness or freshness.</p></article><article className="panel"><Database size={20} /><h3>Your own library</h3><p>Add TXT, Markdown, PDF, CSV, or JSON files that you are authorized to index. Uploaded passages stay in the local project.</p></article></div>
            <div className="license-note"><Info size={17} /><p>Retrieved Stack Overflow passages keep their post links and attribution. Review applicable CC BY-SA terms before redistributing them.</p></div>
          </>
        )}
      </main>
      <footer className="site-footer"><span>FixItAI</span><span>Local technical retrieval workspace</span><span>{health?.gpu || "CUDA model"}</span></footer>
    </div>
  );
}

function TrainingChart({ rows }) {
  if (!rows.length) return <div className="empty-chart">No epoch history is available.</div>;
  const width = 860;
  const height = 230;
  const pad = 28;
  const metrics = [
    { key: "recall_at_1", label: "Recall @ 1", color: "#2767db" },
    { key: "recall_at_5", label: "Recall @ 5", color: "#13a37b" },
    { key: "mrr", label: "MRR", color: "#8b5bd6" },
  ];
  const x = (index) => pad + (rows.length <= 1 ? 0 : index * (width - pad * 2) / (rows.length - 1));
  const y = (value) => height - pad - Math.max(0, Math.min(1, value)) * (height - pad * 2);
  return <div className="chart-wrap"><div className="chart-legend">{metrics.map((metric) => <span key={metric.key}><i style={{ background: metric.color }} />{metric.label}</span>)}</div><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Model recall and MRR across training epochs"><line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} className="chart-axis" />{[0, 0.5, 1].map((v) => <g key={v}><line x1={pad} y1={y(v)} x2={width - pad} y2={y(v)} className="chart-grid" /><text x="2" y={y(v) + 4} className="chart-label">{v.toFixed(1)}</text></g>)}{metrics.map((metric) => <g key={metric.key}><polyline points={rows.map((row, index) => `${x(index)},${y(row[metric.key] || 0)}`).join(" ")} fill="none" stroke={metric.color} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />{rows.map((row, index) => <circle key={index} cx={x(index)} cy={y(row[metric.key] || 0)} r="4" fill={metric.color} />)}</g>)}{rows.map((row, index) => <text key={row.epoch} x={x(index)} y={height - 5} textAnchor="middle" className="chart-label">{row.epoch}</text>)}</svg></div>;
}

export default App;



