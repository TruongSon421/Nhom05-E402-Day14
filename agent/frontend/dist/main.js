import"/frontend-static/style.css";const s={activeTab:"result",lastResponse:null,lastMetrics:null},n=e=>{const t=document.getElementById(e);if(!t)throw new Error(`Missing element: ${e}`);return t},d=e=>JSON.stringify(e,null,2),l=e=>e.trim().toLowerCase()==="true";function r(){n("app").innerHTML=`
  <div class="shell">
    <div class="header">
      <div>
        <h1 class="title">Travel Agent Console</h1>
        <p class="subtitle">Vite + TypeScript frontend for LangGraph safety pipeline</p>
      </div>
      <button class="secondary" id="btnMetricsTop">Refresh Metrics</button>
    </div>
    <div class="grid">
      <section class="card">
        <h3>Request</h3>
        <label>API Key</label>
        <input id="apiKey" placeholder="dev-key-change-me-in-production" />
        <label>Travel Prompt</label>
        <textarea id="question" placeholder="Tu van du lich Da Nang 4 ngay budget 6tr, tim ve may bay va khach san"></textarea>
        <div class="row">
          <div>
            <label>HITL Approved (true/false)</label>
            <input id="approved" value="false" />
          </div>
          <div>
            <label>Human Feedback</label>
            <input id="feedback" placeholder="optional" />
          </div>
        </div>
        <div class="actions">
          <button class="primary" id="btnSend">Run Agent</button>
          <button class="secondary" id="btnMetrics">Load Metrics</button>
          <button class="danger" id="btnClear">Clear</button>
        </div>
      </section>
      <section class="card">
        <h3>Response Workspace</h3>
        <div class="tabs">
          <button class="tab-btn active" data-tab="result">Result</button>
          <button class="tab-btn" data-tab="trace">Trace</button>
          <button class="tab-btn" data-tab="metrics">Monitoring</button>
        </div>
        <div id="statusPills"></div>
        <div class="output mono"><pre id="tabContent">No data yet.</pre></div>
      </section>
    </div>
  </div>`}function a(){document.querySelectorAll(".tab-btn").forEach(e=>{const t=e.getAttribute("data-tab");e.classList.toggle("active",t===s.activeTab)})}function o(e){const t=[];t.push(`<span class="pill ${e.blocked_by_guardrails?"danger":"ok"}">${e.blocked_by_guardrails?"Blocked":"Passed"}</span>`),t.push(`<span class="pill ${e.judge_passed?"ok":"warn"}">Judge: ${e.judge_passed}</span>`),t.push(`<span class="pill warn">Latency: ${e.latency_ms}ms</span>`),e.alerts?.length&&t.push(`<span class="pill danger">Alerts: ${e.alerts.join(", ")}</span>`),n("statusPills").innerHTML=t.join("")}function c(){const e=n("tabContent");if(s.activeTab==="metrics"){e.textContent=s.lastMetrics?d(s.lastMetrics):"Metrics not loaded.";return}if(!s.lastResponse){e.textContent="No response yet.";return}s.activeTab==="trace"?e.textContent=d({trace:s.lastResponse.trace,blocked_by:s.lastResponse.blocked_by,judge_scores:s.lastResponse.judge_scores,redactions:s.lastResponse.redactions,audit_id:s.lastResponse.audit_id}):e.textContent=s.lastResponse.answer||"(empty answer)"}async function i(){const e=n("apiKey").value.trim(),t=n("question").value.trim();if(!e||!t){alert("Please fill API key and prompt.");return}const p=await fetch("/ask-multi-agent",{method:"POST",headers:{"Content-Type":"application/json","X-API-Key":e},body:JSON.stringify({question:t,human_approved:l(n("approved").value),human_feedback:n("feedback").value.trim()})}),u=await p.json();p.ok?(s.lastResponse=u,o(s.lastResponse)):s.lastResponse={answer:d(u),blocked_by_guardrails:!0,blocked_by:"http_error",judge_passed:!1,judge_scores:{},redactions:[],trace:[],latency_ms:0,alerts:[],audit_id:""},s.activeTab="result",a(),c()}async function m(){const e=n("apiKey").value.trim();if(!e){alert("Input API key first.");return}const t=await fetch("/metrics",{headers:{"X-API-Key":e}});s.lastMetrics=await t.json(),s.activeTab="metrics",a(),c()}function g(){n("btnSend").addEventListener("click",()=>void i()),n("btnMetrics").addEventListener("click",()=>void m()),n("btnMetricsTop").addEventListener("click",()=>void m()),n("btnClear").addEventListener("click",()=>{s.lastResponse=null,s.lastMetrics=null,s.activeTab="result",n("statusPills").innerHTML="",a(),c()}),document.querySelectorAll(".tab-btn").forEach(e=>{e.addEventListener("click",()=>{const t=e.getAttribute("data-tab");t&&(s.activeTab=t,a(),c())})})}r();g();c();
