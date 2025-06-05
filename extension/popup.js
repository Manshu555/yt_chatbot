document.addEventListener("DOMContentLoaded", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  // Robust URL validation for YouTube
  const url = new URL(tab.url);
  if (!url.hostname.includes("youtube.com") && !url.hostname.includes("youtu.be")) {
    document.body.innerHTML = "<p>This extension only works on YouTube video pages.</p>";
    return;
  }

  // Extract video ID
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

  document.getElementById("askBtn").addEventListener("click", async () => {
    const query = document.getElementById("queryInput").value;
    const responseBox = document.getElementById("responseBox");

    if (!query) {
      responseBox.innerText = "Please enter a question.";
      return;
    }

    responseBox.innerText = "Thinking...";
    document.getElementById("askBtn").disabled = true;

    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10-second timeout

      const res = await fetch("http://localhost:8000/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          query: query,
          video_id: videoId
        }),
        signal: controller.signal
      });

      clearTimeout(timeoutId); // Cancel the timeout after response is received

      const data = await res.json();
      if (data.error) {
        responseBox.innerText = `Error: ${data.error}`;
      } else {
        responseBox.innerText = data.answer;
      }
    } catch (err) {
      responseBox.innerText = "Error: Ascertain that the server is running and accessible.";
      console.error("Fetch error:", err);
    } finally {
      document.getElementById("askBtn").disabled = false;
    }
  });
});
