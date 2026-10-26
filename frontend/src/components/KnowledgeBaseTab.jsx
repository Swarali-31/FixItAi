import React from "react";
import { ArrowDownRight, ArrowUpRight, Database, FileText, LoaderCircle, Search, Upload } from "lucide-react";

export function KnowledgeBaseTab({
  librarySearchTerm,
  setLibrarySearchTerm,
  fetchLibraryPassages,
  isLibraryLoading,
  isUploading,
  handleDocumentUpload,
  libraryData,
}) {
  return (
    <div className="tab-pane">
      <div className="page-header">
        <div>
          <span className="header-eyebrow">TECHNICAL REPOSITORY</span>
          <h1>Knowledge Base</h1>
          <p>
            Browse through indexed solutions or upload proprietary guides, runbooks, and documents.
          </p>
        </div>
      </div>

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
        <span>
          <Database size={14} /> {(libraryData?.total || 0).toLocaleString()} total passages
        </span>
        <span>Supported formats: TXT, Markdown, PDF, CSV, JSON</span>
      </div>

      {isLibraryLoading ? (
        <div className="loading-state">
          <LoaderCircle className="spinner" size={28} />
          <p>Loading documentation passages...</p>
        </div>
      ) : (
        <div className="library-grid">
          {libraryData?.items?.map((item) => (
            <article key={item.chunk_id} className="library-card">
              <div className="library-card-icon">
                <FileText size={18} />
              </div>
              <div className="library-card-body">
                <span className="library-category">{item.category || item.source}</span>
                <h4>{item.title}</h4>
                <p>{item.text}</p>
                <div className="tag-chips">
                  {(item.technologies || []).slice(0, 4).map((tech) => (
                    <span className="chip" key={tech}>
                      {tech}
                    </span>
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

      {!isLibraryLoading && libraryData?.items?.length < libraryData?.total && (
        <div className="load-more-row">
          <button className="btn btn-secondary" onClick={() => fetchLibraryPassages(librarySearchTerm, true)}>
            <span>Load more passages</span>
            <ArrowDownRight size={15} />
          </button>
        </div>
      )}
    </div>
  );
}
