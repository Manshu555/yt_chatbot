const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const extensionDir = path.join(__dirname, "..", "extension");
const script = fs.readFileSync(path.join(extensionDir, "popup.js"), "utf8");

async function popup(fetch, tabUrl = "https://www.youtube.com/watch?v=aircAruvnKk") {
  const elements = new Map();
  const timers = new Map();
  const requests = [];
  let ready;
  function makeElement(tag = "div") {
      const classes = new Set(["hidden"]);
      const attributes = new Map();
      return {
        tagName: tag.toUpperCase(), children: [], style: {}, dataset: {}, _text: "",
        value: "What is a neural network?", checked: true, disabled: false,
        innerHTML: "", listeners: {},
        get innerText() { return this._text + this.children.map(child => child.innerText).join(""); },
        set innerText(value) { this._text = value; this.children = []; },
        get textContent() { return this.innerText; },
        set textContent(value) { this.innerText = value; },
        classList: {
          add: (name) => classes.add(name), remove: (name) => classes.delete(name),
          contains: (name) => classes.has(name),
          toggle: (name, force) => force ? classes.add(name) : classes.delete(name),
        },
        setAttribute(name, value) { attributes.set(name, value); },
        getAttribute(name) { return attributes.get(name); },
        querySelectorAll() { return []; },
        replaceChildren(...children) { this._text = ""; this.children = children; },
        append(...children) { this.children.push(...children); },
        addEventListener(name, listener) { this.listeners[name] = listener; },
        appendChild(child) { this.children.push(child); },
        focus() {},
      };
  }
  function element(id) {
    if (!elements.has(id)) elements.set(id, makeElement());
    return elements.get(id);
  }
  const context = vm.createContext({
    URL, URLSearchParams, AbortController,
    chrome: { tabs: { query: async () => tabUrl ? [{ url: tabUrl }] : [] } },
    document: {
      body: makeElement(), getElementById: element, createElement: makeElement,
      querySelectorAll: () => [],
      addEventListener: (name, listener) => { ready = listener; },
    },
    fetch: async (...args) => { requests.push(args); return fetch(...args); },
    setTimeout: (callback, delay) => { timers.set(1, { callback, delay }); return 1; },
    clearTimeout: (id) => timers.delete(id),
  });
  vm.runInContext(script, context);
  await ready();
  return { element, timers, requests, context, ask: () => element("askBtn").listeners.click() };
}

const response = (status, data) => ({ status, ok: status >= 200 && status < 300, json: async () => data });

function keydown(ui, overrides = {}) {
  const event = {
    key: "Enter", shiftKey: false, isComposing: false, repeat: false,
    defaultPrevented: false,
    preventDefault() { this.defaultPrevented = true; },
    ...overrides,
  };
  return { event, pending: ui.element("queryInput").listeners.keydown(event) };
}

test("Enter submits the question through the existing API request", async () => {
  const ui = await popup(async () => response(200, { answer: "Answer", validation_required: false }));
  ui.element("queryInput").value = "  Explain weights and biases  ";
  ui.element("validateToggle").checked = false;
  const { event, pending } = keydown(ui);
  await pending;
  assert.equal(event.defaultPrevented, true);
  assert.equal(ui.requests.length, 1);
  assert.equal(ui.requests[0][0], "http://127.0.0.1:8000/ask");
  assert.deepEqual(JSON.parse(ui.requests[0][1].body), {
    query: "Explain weights and biases", video_id: "aircAruvnKk", validate_externally: false,
  });
  assert.equal(ui.element("answerBox").innerText, "Answer");
});

test("Shift+Enter and composing Enter keep native textarea behavior", async () => {
  const ui = await popup(() => assert.fail("Unexpected request"));
  for (const overrides of [{ shiftKey: true }, { isComposing: true }, { keyCode: 229 }, { key: "a" }]) {
    const { event, pending } = keydown(ui, overrides);
    await pending;
    assert.equal(event.defaultPrevented, false);
  }
  assert.equal(ui.requests.length, 0);
});

test("Enter never submits an empty question or a repeated key event", async () => {
  const ui = await popup(() => assert.fail("Unexpected request"));
  ui.element("queryInput").value = " \n ";
  await keydown(ui).pending;
  ui.element("queryInput").value = "Question";
  const { event, pending } = keydown(ui, { repeat: true });
  await pending;
  assert.equal(event.defaultPrevented, true);
  assert.equal(ui.requests.length, 0);
});

test("Enter and button clicks share the in-flight request guard", async () => {
  let finish;
  const ui = await popup(() => new Promise(resolve => { finish = resolve; }));
  const { pending } = keydown(ui);
  await keydown(ui).pending;
  await ui.ask();
  assert.equal(ui.requests.length, 1);
  finish(response(200, { answer: "Answer", validation_required: false }));
  await pending;
  assert.equal(ui.element("askBtn").disabled, false);
  const next = ui.ask();
  assert.equal(ui.requests.length, 2);
  finish(response(200, { answer: "Next answer", validation_required: false }));
  await next;
});

test("provider errors appear as errors with no answer or validation badge", async () => {
  const ui = await popup(async () => response(429, { error: "Gemini daily quota exceeded." }));
  await ui.ask();
  assert.equal(ui.element("errorMessage").innerText, "Gemini daily quota exceeded.");
  assert.equal(ui.element("errorBox").classList.contains("hidden"), false);
  assert.equal(ui.element("responseContainer").classList.contains("hidden"), true);
  assert.equal(ui.element("validationSection").classList.contains("hidden"), true);
  assert.equal(ui.element("askBtn").disabled, false);
  assert.equal(ui.element("loadingBox").classList.contains("hidden"), true);
  assert.equal(ui.timers.size, 0);
});

test("missing transcript detail is shown to the user", async () => {
  const ui = await popup(async () => response(422, { detail: "Could not retrieve transcript." }));
  await ui.ask();
  assert.equal(ui.element("errorMessage").innerText, "Could not retrieve transcript.");
});

test("non-JSON backend failure reports HTTP status", async () => {
  const ui = await popup(async () => ({ status: 500, ok: false, json: async () => { throw new SyntaxError(); } }));
  await ui.ask();
  assert.match(ui.element("errorMessage").innerText, /HTTP 500/);
  assert.equal(ui.element("askBtn").disabled, false);
  assert.equal(ui.timers.size, 0);
});

test("network failure is distinct from timeout", async () => {
  const ui = await popup(async () => { throw new TypeError("Failed to fetch"); });
  await ui.ask();
  assert.match(ui.element("errorMessage").innerText, /Cannot reach the backend/);
  assert.equal(ui.timers.size, 0);
});

test("timeout is three minutes and never resubmits the request", async () => {
  let ui;
  ui = await popup(async (url, options) => {
    assert.equal(ui.timers.get(1).delay, 180000);
    ui.timers.get(1).callback();
    assert.equal(options.signal.aborted, true);
    const error = new Error();
    error.name = "AbortError";
    throw error;
  });
  await ui.ask();
  assert.match(ui.element("errorMessage").innerText, /timed out after 3 minutes/);
  assert.equal(ui.requests.length, 1);
  assert.equal(ui.timers.size, 0);
});

test("validated success preserves evidence and all four status badges", async () => {
  for (const [status, className] of [
    ["SUPPORTED", "status-supported"], ["PARTIALLY_SUPPORTED", "status-partial"],
    ["CONTRADICTED", "status-contradicted"], ["UNVERIFIED", "status-unverified"],
  ]) {
    const ui = await popup(async () => response(200, {
      answer: "A neural network connects neurons.", validation_required: true,
      validation_status: status, video_evidence: [{ text: "Weights and biases" }], sources: [],
    }));
    await ui.ask();
    assert.equal(ui.element("answerBox").innerText, "A neural network connects neurons.");
    assert.ok(ui.element("validationBadge").children[0].className.includes(className));
    assert.equal(ui.element("validationBadge").children[0].getAttribute("data-status"), status);
    assert.match(ui.element("videoEvidenceBox").innerText, /Weights and biases/);
    assert.equal(ui.element("errorBox").classList.contains("hidden"), true);
    assert.equal(ui.requests[0][0], "http://127.0.0.1:8000/ask");
  }
});

test("duplicate clicks do not submit duplicate pipeline requests", async () => {
  let finish;
  const ui = await popup(() => new Promise((resolve) => { finish = resolve; }));
  const pending = ui.ask();
  await ui.ask();
  assert.equal(ui.requests.length, 1);
  finish(response(200, { answer: "Answer", validation_required: false }));
  await pending;
});

test("missing tabs and non-YouTube hosts never call the backend", async () => {
  for (const url of [null, "https://notyoutube.com/watch?v=aircAruvnKk"]) {
    const ui = await popup(() => assert.fail("Unexpected request"), url);
    assert.match(ui.element("errorMessage").innerText, /YouTube/);
    assert.equal(ui.context.document.body.classList.contains("context-error"), true);
    assert.equal(ui.requests.length, 0);
  }
});

test("question and validation toggle keep the API request contract", async () => {
  for (const validate of [true, false]) {
    const ui = await popup(async () => response(200, { answer: "Answer", validation_required: validate }));
    ui.element("queryInput").value = "  Original question\nwith two lines  ";
    ui.element("validateToggle").checked = validate;
    await ui.ask();
    const [url, options] = ui.requests[0];
    assert.equal(url, "http://127.0.0.1:8000/ask");
    assert.equal(options.method, "POST");
    assert.deepEqual(JSON.parse(options.body), {
      query: "Original question\nwith two lines", video_id: "aircAruvnKk", validate_externally: validate,
    });
  }
});

test("claim disclosures preserve returned verdicts and explanation text", async () => {
  const claims = [
    { claim_id: "claim_1", claim: "Original factual claim", status: "PARTIALLY_SUPPORTED", explanation: "Only part of this claim was supported.", supporting_evidence: [], contradicting_evidence: [] },
    { claim_id: "claim_2", claim: "Another original claim", status: "UNVERIFIED", explanation: "Validation failed due to API error.", supporting_evidence: [], contradicting_evidence: [] },
  ];
  const ui = await popup(async () => response(200, { answer: "Original answer", validation_required: true, validation_status: "PARTIALLY_SUPPORTED", claims }));
  await ui.ask();
  const cards = ui.element("claimList").children;
  assert.equal(cards.length, 2);
  assert.equal(cards[0].children[0].children[0].getAttribute("data-status"), "PARTIALLY_SUPPORTED");
  assert.match(cards[0].innerText, /Original factual claim/);
  assert.match(cards[1].innerText, /Validation failed due to API error/);
  assert.match(ui.element("validationSummary").innerText, /1 partially supported/);
  assert.match(ui.element("validationSummary").innerText, /1 unverified/);
});

test("source markup remains text and unsafe URLs are not clickable", async () => {
  const ui = await popup(async () => response(200, {
    answer: "Answer", validation_required: true,
    sources: [{ title: "<img src=x onerror=alert(1)>", url: "javascript:alert(1)", snippet: "Original snippet" }, { title: "Real source", url: "https://example.com/evidence", domain: "example.com" }],
  }));
  await ui.ask();
  const unsafe = ui.element("sourcesList").children[0].children[1].children[0];
  assert.equal(unsafe.tagName, "SPAN");
  assert.equal(unsafe.textContent, "<img src=x onerror=alert(1)>");
  const safe = ui.element("sourcesList").children[1].children[1].children[1];
  assert.equal(safe.href, "https://example.com/evidence");
  assert.equal(safe.target, "_blank");
  assert.equal(safe.rel, "noopener noreferrer");
});

test("manifest permits the loopback API and only loads the popup in its own page", () => {
  const manifest = JSON.parse(fs.readFileSync(path.join(extensionDir, "manifest.json"), "utf8"));
  assert.ok(manifest.host_permissions.includes("http://127.0.0.1:8000/*"));
  assert.ok(manifest.host_permissions.includes("http://localhost:8000/*"));
  assert.equal(manifest.content_scripts, undefined);
  assert.match(fs.readFileSync(path.join(extensionDir, "popup.html"), "utf8"), /<meta charset="utf-8">/);
});
