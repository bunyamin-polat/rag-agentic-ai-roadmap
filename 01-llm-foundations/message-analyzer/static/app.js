// Message Analyzer front end.
// All model output is inserted with textContent, never innerHTML, so it cannot inject markup.

const messageBox = document.getElementById("message");
const modelsBox = document.getElementById("models");
const analyzeButton = document.getElementById("analyze");
const statusLine = document.getElementById("status");
const resultsBox = document.getElementById("results");
const maxModels = Number(modelsBox.dataset.max);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function setStatus(text, isError = false) {
  statusLine.textContent = text;
  statusLine.classList.toggle("error", isError);
}

function selectedModels() {
  return [...modelsBox.querySelectorAll("input:checked")].map((input) => input.value);
}

async function loadModels() {
  try {
    const response = await fetch("/api/models");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);

    modelsBox.textContent = "";
    for (const name of data.models) {
      const label = el("label");
      const input = el("input");
      input.type = "checkbox";
      input.value = name;
      input.checked = name === data.default;
      label.append(input, name);
      modelsBox.append(label);
    }
    if (data.models.length === 0) {
      modelsBox.textContent = "No chat models found. Pull one first: ollama pull gpt-oss:20b";
    }
  } catch (error) {
    modelsBox.textContent = "";
    setStatus(error.message || "Could not load models.", true);
  }
}

// Keep at most `maxModels` boxes ticked.
modelsBox.addEventListener("change", (event) => {
  if (event.target.checked && selectedModels().length > maxModels) {
    event.target.checked = false;
    setStatus(`You can compare up to ${maxModels} models.`, true);
  }
});

document.querySelectorAll("[data-example]").forEach((button) => {
  button.addEventListener("click", () => {
    messageBox.value = button.dataset.example;
    messageBox.focus();
  });
});

function field(title, text, extraClass = "") {
  const box = el("div", `field ${extraClass}`.trim());
  box.append(el("h3", "", title), el("p", "", text));
  return box;
}

function renderResult(result) {
  const card = el("article", "card");
  const head = el("div", "card-head");
  head.append(el("h2", "", result.model));
  card.append(head);

  if (result.error) {
    card.classList.add("failed");
    card.append(el("p", "", result.error));
    return card;
  }

  head.append(el("span", "meta", `${result.seconds}s · ${result.output_tokens} tokens`));

  const sentiment = el("div", "sentiment");
  const row = el("div", "sentiment-row");
  row.append(el("span", `badge ${result.sentiment}`, result.sentiment), el("span", "meta", `${result.sentiment_score} / 100`));
  const bar = el("div", "bar");
  const fill = el("span", result.sentiment);
  fill.style.width = `${Math.max(result.sentiment_score, 3)}%`;
  bar.append(fill);
  sentiment.append(row, bar);

  const tags = el("div", "tags");
  tags.append(
    el("span", "tag", result.category),
    el("span", result.urgency === "high" ? "tag urgent" : "tag", `${result.urgency} urgency`),
  );

  const copy = el("button", "copy", "Copy reply");
  copy.type = "button";
  copy.addEventListener("click", async () => {
    await navigator.clipboard.writeText(result.reply);
    copy.textContent = "Copied";
    setTimeout(() => (copy.textContent = "Copy reply"), 1500);
  });

  card.append(sentiment, tags, field("Summary", result.summary), field("Suggested reply", result.reply, "reply"), copy);
  return card;
}

async function analyze() {
  const message = messageBox.value.trim();
  const models = selectedModels();
  if (!message) return setStatus("Please enter a message.", true);
  if (models.length === 0) return setStatus("Pick at least one model.", true);

  analyzeButton.disabled = true;
  resultsBox.textContent = "";
  setStatus(`Analyzing with ${models.length} model${models.length > 1 ? "s" : ""}...`);

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, models }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);

    data.results.forEach((result) => resultsBox.append(renderResult(result)));
    setStatus("");
  } catch (error) {
    setStatus(error.message || "Something went wrong.", true);
  } finally {
    analyzeButton.disabled = false;
  }
}

analyzeButton.addEventListener("click", analyze);
messageBox.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) analyze();
});

loadModels();
