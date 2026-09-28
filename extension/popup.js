document.addEventListener("DOMContentLoaded", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  const url = new URL(tab.url);
  if (!url.hostname.includes("youtube.com") && !url.hostname.includes("youtu.be")) {
    document.body.innerHTML = "<p>This extension only works on YouTube video pages.</p>";
    return;
  }

  let videoId;
  if (url.hostname.includes("youtu.be")) {
    videoId = url.pathname.split("/")[1];
  } else {
    videoId = new URLSearchParams(url.search).get("v");
  }
  
  if (!videoId) {
    document.body.innerHTML = "<p>Could not extract video ID.</p>";
    return;
  }

  const toggleSourcesBtn = document.getElementById("toggleSourcesBtn");
  const sourcesList = document.getElementById("sourcesList");
  
  toggleSourcesBtn.addEventListener("click", () => {
    if (sourcesList.classList.contains("hidden")) {
      sourcesList.classList.remove("hidden");
      toggleSourcesBtn.innerText = "Collapse Sources";
    } else {
      sourcesList.classList.add("hidden");
      toggleSourcesBtn.innerText = "Expand Sources";
    }
  });

  document.getElementById("askBtn").addEventListener("click", async () => {
    const query = document.getElementById("queryInput").value;
    const validateExternally = document.getElementById("validateToggle").checked;
    
    if (!query) return;

    // Reset UI
    document.getElementById("errorBox").classList.add("hidden");
    document.getElementById("responseContainer").classList.add("hidden");
    document.getElementById("validationSection").classList.add("hidden");
    sourcesList.classList.add("hidden");
    toggleSourcesBtn.innerText = "Expand Sources";
    
    document.getElementById("loadingBox").classList.remove("hidden");
    document.getElementById("askBtn").disabled = true;

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 45000); // 45s timeout

      const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, video_id: videoId, validate_externally: validateExternally }),
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      const data = await res.json();
      document.getElementById("loadingBox").classList.add("hidden");
      
      if (!res.ok || data.error) {
        showError(data.error || "An error occurred");
        return;
      }

      displayResponse(data);
      
    } catch (err) {
      document.getElementById("loadingBox").classList.add("hidden");
      showError("Error: Server not running or request timed out.");
      console.error(err);
    } finally {
      document.getElementById("askBtn").disabled = false;
    }
  });
  
  function showError(msg) {
    const errorBox = document.getElementById("errorBox");
    errorBox.innerText = msg;
    errorBox.classList.remove("hidden");
  }
  
  function displayResponse(data) {
    const container = document.getElementById("responseContainer");
    container.classList.remove("hidden");
    
    document.getElementById("answerBox").innerText = data.answer;
    
    if (data.validation_required) {
      document.getElementById("validationSection").classList.remove("hidden");
      
      // Badge
      const badge = document.getElementById("validationBadge");
      const status = data.validation_status || "UNVERIFIED";
      
      if (status === "SUPPORTED") {
        badge.innerHTML = "✓ SUPPORTED";
        badge.className = "status-supported";
      } else if (status === "PARTIALLY_SUPPORTED") {
        badge.innerHTML = "⚠ PARTIALLY SUPPORTED";
        badge.className = "status-partial";
      } else if (status === "CONTRADICTED") {
        badge.innerHTML = "⚠ CONFLICTING INFORMATION";
        badge.className = "status-contradicted";
      } else {
        badge.innerHTML = "? NOT VERIFIED";
        badge.className = "status-unverified";
      }
      
      const sourceCount = (data.sources && data.sources.length) || 0;
      document.getElementById("sourceCount").innerText = `${sourceCount} sources checked`;
      
      // Video Evidence
      const videoBox = document.getElementById("videoEvidenceBox");
      if (data.video_evidence && data.video_evidence.length > 0) {
        videoBox.innerText = `"${data.video_evidence[0].text}..."`;
      } else {
        videoBox.innerText = "No video evidence found.";
      }
      
      // Sources
      const list = document.getElementById("sourcesList");
      list.innerHTML = "";
      if (data.sources && data.sources.length > 0) {
        data.sources.forEach(src => {
          const item = document.createElement("div");
          item.className = "source-item";
          item.innerHTML = `
            <div class="source-title">${src.title}</div>
            <a href="${src.url}" target="_blank" class="source-link">View Source</a>
          `;
          list.appendChild(item);
        });
      } else {
        list.innerHTML = "<div>No external sources found.</div>";
      }
    }
  }
});
