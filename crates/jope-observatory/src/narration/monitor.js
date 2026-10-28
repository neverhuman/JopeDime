"use strict";
(() => {
  const get = (id) => document.getElementById(id);
  const state = { run: null, after: "0", socket: null, retry: null, failures: 0, paused: false, stopped: false, count: 0, gap: false, epoch: 0 };
  const setState = (name, detail, tone = "") => {
    get("state").textContent = name; get("detail").textContent = detail; get("light").className = "light " + tone;
  };
  const filter = () => {
    const query = get("filter").value.toLowerCase();
    let shown=0;
    document.querySelectorAll(".event").forEach((node) => { node.hidden = !node.dataset.search.includes(query) || (get("mode").value !== "audit" && node.dataset.salient !== "yes"); if(!node.hidden) shown+=1; });
    get("count").textContent=shown+" shown · "+state.count+" retained in view · up to 500";
  };
  const health = (frame) => {
    get("boundary").textContent = frame.last_sequence;
    const capture=frame.capture || {};
    get("coverage").textContent = state.gap ? "Capture gaps disclosed" : capture.status === "failed" || capture.status === "resource_limit" || capture.status === "incomplete" ? "Capture "+capture.status.replaceAll("_"," ") : frame.partial_tail ? "Uncommitted tail withheld" : capture.status === "complete" ? "Capture complete" : "Committed prefix · closure unknown";
  };
  const append = (event) => {
    if (event.run_id !== state.run || !/^[1-9][0-9]*$/.test(event.sequence)) throw Error("Unexpected run or sequence");
    if (BigInt(event.sequence) <= BigInt(state.after)) return;
    if (BigInt(event.sequence) !== BigInt(state.after) + 1n) throw Error("Sequence gap; reconnect from retained cursor");
    if (!event.narration || typeof event.narration.text !== "string") throw Error("Invalid commentary event");
    get("empty").hidden = true;
    const fact = event.fact || {};
    if (fact.kind === "tap_dropped") { state.gap = true; get("coverage").textContent = "Capture gaps disclosed"; }
    const row = document.createElement("article"); row.className = "event";
    const step = document.createElement("span"); step.className = "step"; step.textContent = event.sequence;
    const card = document.createElement("div"); card.className = "card";
    const meta = document.createElement("div"); meta.className = "meta";
    const category = document.createElement("span"); category.className = "category"; category.textContent = String(event.narration.category || "commentary").replaceAll("_", " ");
    const scope = document.createElement("span"); scope.textContent = fact.kind === "prediction" || fact.kind === "prediction_bound" ? "INFERENCE" : fact.cut != null ? "CUT " + fact.cut : "RUN";
    if (event.narration.category === "limitation" || fact.kind === "tap_dropped") category.classList.add("warning");
    if (fact.status === "failed" || fact.outcome === "operator_failed") category.classList.add("error");
    meta.append(category, scope);
    const candidateId = fact.program_sha256 || fact.model_sha256 || (fact.child && fact.child.sha256);
    if (typeof candidateId === "string") {
      const candidate = document.createElement("span"); candidate.className = "candidate"; candidate.textContent = candidateId.slice(0, 12); candidate.title = candidateId; meta.append(candidate);
    }
    const text = document.createElement("p"); text.textContent = event.narration.text;
    const details = document.createElement("details"); const summary = document.createElement("summary"); summary.textContent = "Inspect evidence & exact event";
    const evidence = document.createElement("pre"); evidence.textContent = JSON.stringify(event, null, 2);
    const brief=document.createElement("pre"); brief.className="evidence-summary";
    const links=Object.entries(event.links || {}).filter(([,v])=>v).map(([k,v])=>k+": "+v);
    const recipes=(Array.isArray(fact.parents)?fact.parents:[]).map((p,i)=>"Parent role "+(i+1)+": "+p.canonical);
    if(fact.child) recipes.push("Child: "+fact.child.canonical);
    const claims=(event.narration.claims || []).map(c=>c.category+": "+c.predicate);
    brief.textContent=[...links,...recipes,...claims].join("\n");
    details.append(summary, brief, evidence); card.append(meta, text, details); row.append(step, card);
    const routine = fact.kind === "started" || (fact.kind === "admitted" && fact.disposition === "admitted") || (fact.kind === "drawn" && Array.isArray(fact.parents) && fact.parents.length > 0 && event.narration.delta && event.narration.delta.edits.length === 0 && fact.outcome !== "operator_failed" && event.narration.category !== "limitation");
    row.dataset.salient = routine ? "no" : "yes";
    row.dataset.search = event.narration.text.toLowerCase() + " " + category.textContent.toLowerCase();
    get("timeline").append(row); state.count += 1;
    if (state.count > 500) { (document.querySelector('.event[data-salient="no"]') || document.querySelector(".event")).remove(); state.count -= 1; }
    state.after = event.sequence; get("cursor").textContent = state.after; get("count").textContent = state.count + " retained in view · up to 500";
    filter();
    if (get("autoscroll").checked && !row.hidden) row.scrollIntoView({ block: "nearest" });
  };
  const disconnect = () => {
    state.epoch += 1;
    clearTimeout(state.retry); state.retry = null;
    if (state.socket) { state.socket.onopen = state.socket.onmessage = state.socket.onerror = state.socket.onclose = null; state.socket.close(); state.socket = null; }
  };
  const retry = () => {
    if(state.paused || state.stopped) return;
    state.failures += 1;
    const delay=Math.min(15000,500*2**Math.min(state.failures,5));
    setState("Reconnecting","Retained cursor "+state.after+" · retry in "+Math.ceil(delay/1000)+"s","error");
    clearTimeout(state.retry); state.retry=setTimeout(connect,delay);
  };
  const connect = async () => {
    disconnect(); if (state.paused) return;
    const epoch=state.epoch;
    state.stopped = false; setState("Connecting", "Resuming after event " + state.after);
    try {
      const response = await fetch("/api/v1/gp/reasoning/status", { cache: "no-store" });
      if (!response.ok) throw Error("Journal unavailable");
      const status = await response.json();
      if(epoch !== state.epoch || state.paused) return;
      if (state.run !== null && status.run_id !== state.run) {state.stopped=true;throw Error("Run changed; reload to select new journal");}
      state.run = status.run_id; get("run").textContent = state.run; get("run").title = state.run; health(status);
      const url = new URL("/api/v1/gp/reasoning", location.href); url.protocol = location.protocol === "https:" ? "wss:" : "ws:";
      url.searchParams.set("run_id", state.run); url.searchParams.set("after", state.after);
      const socket = new WebSocket(url); state.socket = socket;
      socket.onopen = () => { state.failures = 0; setState("Connected", "Following retained journal", "live"); };
      socket.onmessage = ({ data }) => {
        try {
          const frame = JSON.parse(data);
          if (frame.schema === "jope.narration-heartbeat/v1") { health(frame); return; }
          if (frame.schema === "jope.narration-error/v1") {
            if(frame.code === "request_limit") {disconnect();retry();return;}
            throw Error(frame.message || "Journal unavailable");
          }
          if (frame.schema !== "jope.reasoning-event/v1") throw Error("Unsupported event schema");
          append(frame);
        } catch (error) { state.stopped = true; disconnect(); setState("Attention needed", error.message, "error"); }
      };
      socket.onclose = () => {
        state.socket = null; if (state.paused || state.stopped) return;
        retry();
      };
      socket.onerror = () => setState("Disconnected", "Waiting to resume retained cursor", "error");
    } catch (error) {
      if(epoch !== state.epoch || state.paused) return;
      if(state.stopped) setState("Attention needed",error.message,"error");
      else retry();
    }
  };
  get("filter").addEventListener("input", filter);
  get("mode").addEventListener("change", filter);
  get("reconnect").addEventListener("click", () => { get("pause").checked = false; state.paused = false; connect(); });
  get("pause").addEventListener("change", () => {
    state.paused = get("pause").checked;
    if (state.paused) { disconnect(); setState("Paused", "Resume replays everything after event " + state.after, "paused"); }
    else connect();
  });
  connect();
})();
