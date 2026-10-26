import React from "react";
import { Sparkles } from "lucide-react";
import { getUIText } from "../i18n";

export function Navbar({ activeTab, setActiveTab, navigationItems, systemHealth, targetLang = "en", localizeEntirePage = false }) {
  const isModelReady = systemHealth?.status === "ready";
  const deviceBadge = systemHealth?.device ? ` (${systemHealth.device.toUpperCase()})` : "";
  const activeLang = localizeEntirePage && targetLang !== "en" ? targetLang : "en";

  return (
    <header className="navbar">
      <div className="navbar-container">
        <div className="nav-brand" onClick={() => setActiveTab("troubleshoot")} style={{ cursor: "pointer" }}>
          <div className="brand-logo">
            <Sparkles size={20} />
          </div>
          <div className="brand-titles">
            <span className="brand-name">FixItAI</span>
            <span className="brand-tag">Diagnostic Intelligence</span>
          </div>
        </div>

        <nav className="nav-links">
          {navigationItems.map(({ id, label, icon: Icon }) => {
            const localizedLabel = getUIText(activeLang, `nav_${id}`) || label;
            return (
              <button
                key={id}
                className={`nav-button ${activeTab === id ? "active" : ""}`}
                onClick={() => setActiveTab(id)}
              >
                <Icon size={16} />
                <span>{localizedLabel}</span>
              </button>
            );
          })}
        </nav>

        <div className="nav-status">
          <span className={`status-indicator ${isModelReady ? "ready" : "warning"}`} />
          <span>{isModelReady ? `Engine Ready${deviceBadge}` : "Initializing"}</span>
        </div>
      </div>
    </header>
  );
}
