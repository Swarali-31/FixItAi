import React from "react";
import { AlertTriangle, Check, Layers, LoaderCircle, X } from "lucide-react";

function MetricCard({ icon: Icon, label, value, detail }) {
  return (
    <div className="metric-card">
      <div className="metric-icon">
        <Icon size={18} />
      </div>
      <div className="metric-content">
        <span className="metric-label">{label}</span>
        <div className="metric-value">{value}</div>
        {detail && <span className="metric-detail">{detail}</span>}
      </div>
    </div>
  );
}

export function AnalyzerTab({
  analyzerInput,
  setAnalyzerInput,
  executeLinguisticAnalysis,
  isAnalyzing,
  analyzerOutput,
}) {
  return (
    <div className="tab-pane">
      <div className="page-header">
        <div>
          <span className="header-eyebrow">LINGUISTIC INSPECTION</span>
          <h1>Error Syntax Analyzer</h1>
          <p>
            Inspect tokens, grammatical parts of speech, morphological roots, and automated urgency classification.
          </p>
        </div>
      </div>

      <section className="search-card">
        <label
          htmlFor="analyzer-text"
          style={{ fontSize: "14px", fontWeight: "700", display: "block", marginBottom: "10px" }}
        >
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
                const sample =
                  "Python raises ModuleNotFoundError: No module named 'fastapi' when executing main.py";
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
                const sample =
                  "NullPointerException in Java Spring Boot controller when finding user id";
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
                const sample =
                  "Docker daemon failed with error: bind address already in use on port 8080";
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
          {/* Metrics Strip */}
          <div className="metrics-strip">
            <MetricCard
              icon={Layers}
              label="Total Tokens"
              value={analyzerOutput.token_count}
              detail="Word tokens"
            />
            <MetricCard
              icon={Check}
              label="Content Words"
              value={analyzerOutput.content_token_count}
              detail="Meaningful tokens"
            />
            <MetricCard
              icon={X}
              label="Filtered Stopwords"
              value={analyzerOutput.stopword_count}
              detail="Syntactic noise"
            />
            <MetricCard
              icon={AlertTriangle}
              label="Error Severity"
              value={analyzerOutput.severity?.level || "N/A"}
              detail={`Confidence: ${Math.round((analyzerOutput.severity?.confidence || 0) * 100)}%`}
            />
          </div>

          {/* Part of Speech Tagging */}
          <section className="detail-card">
            <div className="detail-header">
              <h3>Part-of-Speech (POS) Tagging</h3>
              <p>Token-level grammatical tagging using Penn Treebank tags.</p>
            </div>
            <div className="tag-chips-wrapper">
              {analyzerOutput.pos_tags?.map((item, idx) => (
                <div
                  className={`pos-badge pos-${item.category.toLowerCase().replace(/[^a-z]/g, "")}`}
                  key={idx}
                  title={item.description}
                >
                  <span className="pos-word">{item.word}</span>
                  <span className="pos-tag-code">{item.tag}</span>
                </div>
              ))}
            </div>
            <div className="pos-legend-bar">
              <span>
                <i className="legend-dot noun" /> Noun
              </span>
              <span>
                <i className="legend-dot verb" /> Verb
              </span>
              <span>
                <i className="legend-dot adjective" /> Adjective
              </span>
              <span>
                <i className="legend-dot adverb" /> Adverb
              </span>
              <span>
                <i className="legend-dot prep" /> Preposition
              </span>
              <span>
                <i className="legend-dot other" /> Pronoun / Conjunction
              </span>
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
                      <td>
                        <strong>{row.original}</strong>
                      </td>
                      <td className="mono">{row.porter_stem}</td>
                      <td className="mono text-highlight">{row.wordnet_lemma}</td>
                      <td>
                        <span className="chip-badge">{row.rule_applied}</span>
                      </td>
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
                    analyzerOutput.entities.technologies.map((t) => (
                      <span className="chip" key={t}>
                        {t}
                      </span>
                    ))
                  ) : (
                    <span className="muted-text">None detected</span>
                  )}
                </div>
              </div>
              <div className="ner-box">
                <h4>Exception & Error Types</h4>
                <div className="ner-chips">
                  {analyzerOutput.entities?.errors?.length > 0 ? (
                    analyzerOutput.entities.errors.map((e) => (
                      <span className="chip chip-danger" key={e}>
                        {e}
                      </span>
                    ))
                  ) : (
                    <span className="muted-text">None detected</span>
                  )}
                </div>
              </div>
              <div className="ner-box">
                <h4>Code Identifiers</h4>
                <div className="ner-chips">
                  {analyzerOutput.entities?.code_identifiers?.length > 0 ? (
                    analyzerOutput.entities.code_identifiers.map((c) => (
                      <span className="chip chip-code" key={c}>
                        {c}
                      </span>
                    ))
                  ) : (
                    <span className="muted-text">None detected</span>
                  )}
                </div>
              </div>
            </div>
          </section>
        </div>
      )}
    </div>
  );
}
