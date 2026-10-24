// Loads the fish once, then streams replies back to the page token by token.
import { Tokenizer, Guppy } from "./guppy.js";

let tok, model;

async function load() {
  const [tj, wb] = await Promise.all([
    fetch("model/tokenizer.json").then(r => r.json()),
    fetch("model/guppy.bin").then(r => r.arrayBuffer()),
  ]);
  tok = new Tokenizer(tj);
  model = new Guppy(wb);
  postMessage({ type: "ready", bytes: wb.byteLength });
}

onmessage = ({ data }) => {
  if (data.type !== "chat") return;
  const prompt = tok.chat(data.system, data.turns);
  const ids = [], t0 = performance.now();
  for (const id of model.generate(prompt, { temperature: data.temperature })) {
    ids.push(id);
    postMessage({ type: "token", text: tok.decode(ids) });
  }
  const ms = performance.now() - t0;
  postMessage({ type: "done", text: tok.decode(ids), tokens: ids.length, ms });
};

load().catch(e => postMessage({ type: "error", message: String(e) }));
