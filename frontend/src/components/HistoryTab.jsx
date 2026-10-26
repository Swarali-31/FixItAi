import React, { useState } from "react";
import { Check, History, MessageSquareText, Play, Trash2, X } from "lucide-react";

export function HistoryTab({
  searchHistoryList,
  handleRerunQuery,
  handleFeedback,
  handleClearHistory,
}) {
  const [confirmClear, setConfirmClear] = useState(false);

  return (
    <div className="tab-pane">
      <div className="page-header">
        <div>
          <span className="header-eyebrow">AUDIT TRAIL</span>
          <h1>Query History</h1>
          <p>
            Review past troubleshooting searches, record relevance feedback, or rerun any inquiry.
          </p>
        </div>
        {searchHistoryList.length > 0 && (
          <div className="header-actions">
            {confirmClear ? (
              <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                <span style={{ fontSize: "12px", color: "#dc2626" }}>Confirm clear?</span>
                <button
                  className="btn btn-sm"
                  style={{ background: "#dc2626", color: "#fff" }}
                  onClick={() => {
                    handleClearHistory();
                    setConfirmClear(false);
                  }}
                >
                  Yes, Clear All
                </button>
                <button className="btn btn-sm btn-secondary" onClick={() => setConfirmClear(false)}>
                  Cancel
                </button>
              </div>
            ) : (
              <button
                className="btn btn-sm btn-secondary"
                onClick={() => setConfirmClear(true)}
                title="Clear all query history"
              >
                <Trash2 size={13} />
                <span>Clear History</span>
              </button>
            )}
          </div>
        )}
      </div>

      {searchHistoryList.length > 0 ? (
        <div className="history-list">
          {searchHistoryList.map((item, idx) => (
            <article className="history-row" key={`${item.query}-${idx}`}>
              <div className="history-icon-box">
                <MessageSquareText size={18} />
              </div>
              <div className="history-content">
                <h4>{item.query}</h4>
                <span className="history-meta">
                  {item.result_ids?.length || 0} retrieved passages · Status:{" "}
                  <strong>{item.feedback || "Feedback pending"}</strong>
                  {item.created_at && ` · ${new Date(item.created_at).toLocaleString()}`}
                </span>
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
  );
}
