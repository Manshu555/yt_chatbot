const STATUS_PRESENTATION = {
  SUPPORTED: { label: "Supported", icon: "circle-check", className: "status-supported" },
  PARTIALLY_SUPPORTED: { label: "Partially supported", icon: "circle-minus", className: "status-partial" },
  CONTRADICTED: { label: "Contradicted", icon: "circle-x", className: "status-contradicted" },
  UNVERIFIED: { label: "Unverified", icon: "circle-help", className: "status-unverified" }
};

function refreshIcons() {
  globalThis.lucide?.createIcons({ attrs: { "aria-hidden": "true" } });
}

function textElement(tag, text, className = "") {
  const element = document.createElement(tag);
  element.textContent = text ?? "";
  element.className = className;
  return element;
}

function iconElement(name, className = "") {
  const icon = document.createElement("i");
  icon.setAttribute("data-lucide", name);
  icon.className = className;
  return icon;
}

function safeUrl(value) {
  try {
    const url = new URL(value);
    return ["https:", "http:"].includes(url.protocol) ? url.href : null;
  } catch {
    return null;
  }
}

function statusBadge(status) {
  const style = STATUS_PRESENTATION[status] || { label: status || "Unverified", icon: "circle-help", className: "status-unverified" };
  const badge = document.createElement("span");
  badge.className = `status-badge ${style.className}`;
  badge.setAttribute("data-status", status || "UNVERIFIED");
  badge.append(iconElement(style.icon), textElement("span", style.label));
  return badge;
}

function renderAnswer(container, answer) {
  if (!globalThis.marked || !globalThis.DOMPurify) {
    container.textContent = answer;
    container.style.whiteSpace = "pre-wrap";
    return;
  }
  // Treat raw HTML as answer text; sanitize generated Markdown markup separately.
  const renderer = new marked.Renderer();
  renderer.html = ({ text }) => text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  container.innerHTML = DOMPurify.sanitize(marked.parse(answer, { renderer, gfm: true, breaks: true }), {
    ALLOWED_TAGS: ["p", "br", "strong", "em", "del", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "blockquote", "pre", "code", "a", "hr", "table", "thead", "tbody", "tr", "th", "td"],
    ALLOWED_ATTR: ["href", "title", "class", "start"]
  });
  container.querySelectorAll("a").forEach(link => {
    if (safeUrl(link.getAttribute("href"))) {
      link.target = "_blank";
      link.rel = "noopener noreferrer";
    } else {
      link.removeAttribute("href");
    }
  });
}

function sourceLink(title, url, className) {
  const href = safeUrl(url);
  const link = textElement(href ? "a" : "span", title, className);
  if (href) {
    link.href = href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  }
  return link;
}

function renderClaims(claims) {
  const list = document.getElementById("claimList");
  const summary = document.getElementById("validationSummary");
  list.replaceChildren();
  summary.replaceChildren();
  // Counts describe the statuses returned by the backend; no new verdict is computed.
  for (const [status, style] of Object.entries(STATUS_PRESENTATION)) {
    const count = claims.filter(claim => claim.status === status).length;
    if (count) {
      const item = textElement("span", "", style.className);
      item.append(iconElement(style.icon), textElement("span", `${count} ${style.label.toLowerCase()}`));
      summary.appendChild(item);
    }
  }
  if (!claims.length) {
    list.appendChild(textElement("p", "No claim details returned.", "empty-result"));
  }
  claims.forEach(claim => {
    const card = document.createElement("details");
    card.className = "claim-card";
    const heading = document.createElement("summary");
    heading.append(statusBadge(claim.status), textElement("span", claim.claim, "claim-text"), iconElement("chevron-down", "disclosure-icon"));
    const body = document.createElement("div");
    body.className = "claim-body";
    if (claim.explanation) body.appendChild(textElement("p", claim.explanation, "claim-explanation"));
    for (const [label, evidence] of [
      ["Supporting evidence", claim.supporting_evidence || []],
      ["Contradicting evidence", claim.contradicting_evidence || []]
    ]) {
      if (!evidence.length) continue;
      body.appendChild(textElement("h3", label, "evidence-label"));
      evidence.forEach(item => {
        const entry = document.createElement("div");
        entry.className = "claim-evidence";
        const link = sourceLink(item.source_title || item.domain || item.url, item.url, "evidence-link");
        if (link.tagName === "A") link.appendChild(iconElement("arrow-up-right"));
        entry.append(link, textElement("p", item.passage));
        body.appendChild(entry);
      });
    }
    card.append(heading, body);
    list.appendChild(card);
  });
}

function renderSources(sources) {
  const list = document.getElementById("sourcesList");
  list.replaceChildren();
  if (!sources.length) {
    list.appendChild(textElement("p", "No external sources returned.", "empty-result"));
  }
  sources.forEach(source => {
    const item = document.createElement("div");
    item.className = "source-item";
    const copy = document.createElement("div");
    copy.className = "source-copy";
    const href = safeUrl(source.url);
    const domain = source.domain || (href ? new URL(href).hostname : "");
    if (domain) copy.appendChild(textElement("p", domain, "source-domain"));
    copy.appendChild(sourceLink(source.title || source.url, source.url, "source-title"));
    if (source.snippet) copy.appendChild(textElement("p", source.snippet, "source-snippet"));
    item.append(iconElement("arrow-up-right"), copy);
    list.appendChild(item);
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  refreshIcons();
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  let url;
  try {
    url = new URL(tab?.url);
  } catch {
    showContextError("Open a YouTube video to ask a question.");
    return;
  }
  const isYouTube = url.hostname === "youtube.com" || url.hostname.endsWith(".youtube.com");
  if (!isYouTube && url.hostname !== "youtu.be") {
    showContextError("This extension only works on YouTube video pages.");
    return;
  }

  let videoId;
  if (url.hostname === "youtu.be") {
    videoId = url.pathname.split("/")[1];
  } else {
    videoId = new URLSearchParams(url.search).get("v");
  }
  
  if (!videoId) {
    showContextError("Could not extract video ID. Open a YouTube video page.");
    return;
  }

  document.getElementById("videoTitle").textContent = tab.title?.replace(/\s*[-|]\s*YouTube\s*$/i, "") || url.href;
  document.getElementById("videoLink").href = url.href;
  const thumbnail = document.getElementById("videoThumbnail");
  thumbnail.src = `https://i.ytimg.com/vi/${encodeURIComponent(videoId)}/mqdefault.jpg`;
  thumbnail.addEventListener("error", () => thumbnail.classList.add("hidden"));
  document.getElementById("videoContext").classList.remove("hidden");
  document.getElementById("videoStatus").classList.remove("hidden");
  document.querySelectorAll(".suggestion").forEach(button => {
    button.addEventListener("click", () => {
      const input = document.getElementById("queryInput");
      input.value = button.dataset.question;
      input.focus();
    });
  });

  const toggleSourcesBtn = document.getElementById("toggleSourcesBtn");
  const sourcesList = document.getElementById("sourcesList");
  
  toggleSourcesBtn.addEventListener("click", () => {
    if (sourcesList.classList.contains("hidden")) {
      sourcesList.classList.remove("hidden");
      document.getElementById("sourcesToggleLabel").innerText = "Collapse sources";
      toggleSourcesBtn.setAttribute("aria-expanded", "true");
    } else {
      sourcesList.classList.add("hidden");
      document.getElementById("sourcesToggleLabel").innerText = "Expand sources";
      toggleSourcesBtn.setAttribute("aria-expanded", "false");
    }
  });

  document.getElementById("askBtn").addEventListener("click", askQuestion);
  document.getElementById("queryInput").addEventListener("keydown", event => {
    if (event.key !== "Enter" || event.shiftKey || event.isComposing || event.keyCode === 229) return;
    event.preventDefault();
    if (!event.repeat) return askQuestion();
  });

  async function askQuestion() {
    const query = document.getElementById("queryInput").value.trim();
    const validateExternally = document.getElementById("validateToggle").checked;
    
    if (!query || document.getElementById("askBtn").disabled) return;

    // Reset UI
    document.getElementById("errorBox").classList.add("hidden");
    document.getElementById("responseContainer").classList.add("hidden");
    document.getElementById("validationSection").classList.add("hidden");
    sourcesList.classList.add("hidden");
    document.getElementById("sourcesToggleLabel").innerText = "Expand sources";
    toggleSourcesBtn.setAttribute("aria-expanded", "false");
    document.getElementById("emptyState").classList.add("hidden");
    document.getElementById("responseContainer").setAttribute("aria-busy", "true");
    document.getElementById("loadingDetail").innerText = validateExternally ? "Video context and external evidence" : "Video context";
    document.getElementById("askButtonLabel").innerText = "Analyzing...";
    
    document.getElementById("loadingBox").classList.remove("hidden");
    document.getElementById("askBtn").disabled = true;

    const controller = new AbortController();
    // Transcript retrieval and three Gemini calls with retries can exceed 45s.
    const timeoutId = setTimeout(() => controller.abort(), 180000);

    try {
      const res = await fetch("http://127.0.0.1:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, video_id: videoId, validate_externally: validateExternally }),
        signal: controller.signal
      });

      let data;
      try {
        data = await res.json();
      } catch (err) {
        if (err.name === "AbortError") throw err;
        showError(`Backend returned an invalid response (HTTP ${res.status}). Check the backend logs.`);
        return;
      }
      
      if (!res.ok || data.error) {
        showError(data.error || (typeof data.detail === "string" ? data.detail : `Request failed (HTTP ${res.status}). Check the backend logs.`));
        return;
      }

      displayResponse(data);
      
    } catch (err) {
      document.getElementById("loadingBox").classList.add("hidden");
      showError(err.name === "AbortError"
        ? "The request timed out after 3 minutes. Check the backend logs before trying again."
        : "Cannot reach the backend at http://127.0.0.1:8000. Start the server and check extension permissions.");
    } finally {
      clearTimeout(timeoutId);
      document.getElementById("loadingBox").classList.add("hidden");
      document.getElementById("askBtn").disabled = false;
      document.getElementById("askButtonLabel").innerText = "Ask about video";
      document.getElementById("responseContainer").setAttribute("aria-busy", "false");
    }
  }
  
  function showError(msg) {
    const errorBox = document.getElementById("errorBox");
    document.getElementById("errorMessage").innerText = msg;
    errorBox.classList.remove("hidden");
  }

  function showContextError(msg) {
    document.body.classList.add("context-error");
    showError(msg);
  }
  
  function displayResponse(data) {
    const container = document.getElementById("responseContainer");
    container.classList.remove("hidden");
    
    renderAnswer(document.getElementById("answerBox"), data.answer);
    const videoBox = document.getElementById("videoEvidenceBox");
    videoBox.replaceChildren();
    const chunks = data.video_evidence || [];
    document.getElementById("videoEvidenceSection").classList.remove("hidden");
    document.getElementById("videoEvidenceCount").innerText = `${chunks.length} ${chunks.length === 1 ? "passage" : "passages"}`;
    if (chunks.length) {
      chunks.forEach(chunk => videoBox.appendChild(textElement("blockquote", chunk.text)));
    } else {
      videoBox.appendChild(textElement("p", "No video evidence found.", "empty-result"));
    }
    document.getElementById("sourcesSection").classList.toggle("hidden", !data.validation_required);
    
    if (data.validation_required) {
      document.getElementById("validationSection").classList.remove("hidden");
      
      const badge = document.getElementById("validationBadge");
      const status = data.validation_status || "UNVERIFIED";
      badge.replaceChildren(statusBadge(status));
      const sourceCount = (data.sources && data.sources.length) || 0;
      document.getElementById("sourceCount").innerText = `${sourceCount} ${sourceCount === 1 ? "source" : "sources"} returned`;
      renderClaims(data.claims || []);
      renderSources(data.sources || []);
    }
    refreshIcons();
  }
});
