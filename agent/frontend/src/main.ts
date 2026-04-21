import "./style.css";

type Message = { role: "user" | "assistant"; text: string; };

type CurrentUser = { id: number; email: string; full_name: string; role: "user" | "admin"; created_at: string; };

type AdminUser = { id: number; email: string; full_name: string; role: string; created_at: string; };

type ViewName = "auth" | "chat" | "admin";

const state: {
  currentView: ViewName;
  authTab: "login" | "register";
  messages: Message[];
  sending: boolean;
  accessToken: string;
  refreshToken: string;
  user: CurrentUser | null;
  adminUsers: AdminUser[];
} = {
  currentView: "auth",
  authTab: "login",
  messages: [],
  sending: false,
  accessToken: localStorage.getItem("accessToken") ?? "",
  refreshToken: localStorage.getItem("refreshToken") ?? "",
  user: null,
  adminUsers: [],
};

const getEl = <T extends HTMLElement>(id: string): T => {
  const node = document.getElementById(id) as T | null;
  if (!node) throw new Error(`Missing element: ${id}`);
  return node;
};

// Utils
function getApiErrorMessage(raw: string, status: number): string {
  if (!raw.trim()) return `API Server Error ${status}`;
  try {
    const parsed = JSON.parse(raw);
    if (typeof parsed?.detail === "string" && parsed.detail.trim()) return `Error: ${parsed.detail}`;
    return `Error ${status}: ${raw}`;
  } catch {
    return `Error ${status}: ${raw}`;
  }
}

function switchView(view: ViewName) {
  if (!state.user && view !== "auth") {
    state.currentView = "auth";
  } else {
    state.currentView = view;
  }
  renderApp();
}

// Layout Renderers
function renderApp() {
  const app = getEl("app");
  if (!state.user) {
    app.innerHTML = renderAuthView();
    bindAuthEvents();
    return;
  }

  app.innerHTML = `
    <div class="sidebar">
      <div class="brand">✨ Travel AI</div>
      <div class="nav-menu">
        <div class="nav-item ${state.currentView === 'chat' ? 'active' : ''}" id="nav-chat">
          <svg style="width:20px" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"></path></svg>
          Chat Assistant
        </div>
        ${state.user.role === 'admin' ? `
        <div class="nav-item ${state.currentView === 'admin' ? 'active' : ''}" id="nav-admin">
          <svg style="width:20px" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z"></path></svg>
          Admin Panel
        </div>
        ` : ''}
      </div>
      <div class="user-profile">
        <div class="user-info">
          <span class="user-name">${state.user.full_name}</span>
          <span class="user-email">${state.user.email}</span>
        </div>
        <button class="btn btn-logout" id="btnLogout">Sign Out</button>
      </div>
    </div>
    
    <div class="main-content">
      <div class="top-bar">
        <div class="page-title" style="font-weight: 600">${state.currentView === 'chat' ? 'AI Assistant' : 'Admin Area'}</div>
        <div class="status-indicator">
          <div class="status-dot" id="statusDot"></div>
          <span id="statusText">Ready</span>
        </div>
      </div>
      <div id="viewContainer" style="flex:1; overflow:hidden;"></div>
    </div>
  `;

  bindNavEvents();
  
  const container = getEl("viewContainer");
  if (state.currentView === "chat") {
    container.innerHTML = renderChatView();
    bindChatEvents();
    scrollToBottom();
  } else if (state.currentView === "admin") {
    container.innerHTML = `<div class="view-admin" id="adminPanelWrapper">Lấy danh sách...</div>`;
    void renderAdminData();
  }
}

// Auth View
function renderAuthView() {
  return `
    <div class="view-auth">
      <div class="auth-card">
        <div class="auth-header">
          <h2>Welcome Back</h2>
          <p>Sign in to continue exploring</p>
        </div>
        <div class="auth-tabs">
          <div class="auth-tab ${state.authTab === 'login' ? 'active' : ''}" id="tabLogin">Sign In</div>
          <div class="auth-tab ${state.authTab === 'register' ? 'active' : ''}" id="tabRegister">Create Account</div>
        </div>
        
        <input class="input-base" id="emailInput" placeholder="Email address" type="email" />
        <input class="input-base" id="passwordInput" placeholder="Password" type="password" />
        ${state.authTab === 'register' ? `<input class="input-base" id="fullNameInput" placeholder="Full name" />` : ''}
        
        <button class="btn btn-primary" id="btnSubmitAuth" style="margin-top: 8px;">
          ${state.authTab === 'login' ? 'Sign In' : 'Sign Up'}
        </button>
      </div>
    </div>
  `;
}

function bindAuthEvents() {
  getEl("tabLogin").addEventListener("click", () => {
    state.authTab = "login";
    renderApp();
  });
  getEl("tabRegister").addEventListener("click", () => {
    state.authTab = "register";
    renderApp();
  });
  getEl("btnSubmitAuth").addEventListener("click", () => {
    if (state.authTab === "login") void login();
    else void register();
  });
}

// Chat View
function renderChatView() {
  return `
    <div class="view-chat">
      <div class="chat-body" id="messages">
        ${state.messages.length === 0 ? `
          <div class="empty-chat">
            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"></path></svg>
            <div>How can I help you plan your trip today?</div>
          </div>
        ` : state.messages.map(msg => `
          <div class="message ${msg.role}">
            <div class="bubble">${msg.text.replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\n/g, "<br>")}</div>
          </div>
        `).join("")}
      </div>
      <div class="chat-input-wrapper">
        <div class="chat-input-box">
          <textarea id="question" placeholder="Ask anything about your destination..." rows="1"></textarea>
          <button class="btn-send" id="btnSend">
             <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"></path></svg>
          </button>
        </div>
      </div>
    </div>
  `;
}

function bindChatEvents() {
  getEl("btnSend").addEventListener("click", () => void sendMessage());
  getEl("question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void sendMessage();
    }
  });
}

function bindNavEvents() {
  const btnLogout = document.getElementById("btnLogout");
  if (btnLogout) btnLogout.addEventListener("click", logout);

  const navChat = document.getElementById("nav-chat");
  if (navChat) navChat.addEventListener("click", () => switchView("chat"));

  const navAdmin = document.getElementById("nav-admin");
  if (navAdmin) navAdmin.addEventListener("click", () => switchView("admin"));
}

function scrollToBottom() {
  const msgs = document.getElementById("messages");
  if (msgs) msgs.scrollTop = msgs.scrollHeight;
}

// Chat Logic
async function sendMessage(): Promise<void> {
  if (state.sending) return;
  const qEl = getEl<HTMLTextAreaElement>("question");
  const question = qEl.value.trim();
  if (!question) return;

  state.messages.push({ role: "user", text: question });
  state.messages.push({ role: "assistant", text: "" });
  const assistantMsgIndex = state.messages.length - 1;

  state.sending = true;
  qEl.value = "";
  
  if(state.currentView !== 'chat') switchView('chat');
  else {
    const msgs = document.getElementById("messages");
    if (msgs) {
      if (state.messages.length <= 2) {
         // It was empty, re-render whole thing
         msgs.innerHTML = state.messages.map(msg => `
          <div class="message ${msg.role}">
            <div class="bubble">${msg.text.replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\n/g, "<br>")}</div>
          </div>
        `).join("");
      } else {
         // Just append to preserve layout/focus
         msgs.innerHTML += `
          <div class="message user">
            <div class="bubble">${question.replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\n/g, "<br>")}</div>
          </div>
          <div class="message assistant">
            <div class="bubble"></div>
          </div>
        `;
      }
      scrollToBottom();
    }
  }
  
  const statusDot = getEl("statusDot");
  const statusText = getEl("statusText");
  statusDot.classList.add("busy");
  statusText.textContent = "Agent thinking...";

  try {
    const response = await fetchWithAuth("/ask-multi-agent/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${state.accessToken}` },
      body: JSON.stringify({ question, human_approved: false, human_feedback: "" }),
    });

    if (!response.ok) {
      const err = await response.text();
      state.messages[assistantMsgIndex].text = `⚠️ ${getApiErrorMessage(err, response.status)}`;
      renderApp();
      return;
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error("No reader");
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n\n");
      buffer = lines.pop() || "";

      for (const block of lines) {
        if (!block.trim()) continue;
        try {
          const parsed = JSON.parse(block);
          if (parsed.event === "metadata") {
            statusText.textContent = parsed.data.status;
          } else if (parsed.event === "token") {
            state.messages[assistantMsgIndex].text += parsed.data;
            const msgs = document.getElementById("messages");
            if (msgs) {
              const lastChild = msgs.lastElementChild;
              if (lastChild) lastChild.querySelector('.bubble')!.innerHTML = state.messages[assistantMsgIndex].text.replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\n/g, "<br>");
              msgs.scrollTop = msgs.scrollHeight;
            }
          } else if (parsed.event === "error") {
             state.messages[assistantMsgIndex].text += `\n⚠️ ${parsed.data}`;
             renderApp();
          } else if (parsed.event === "complete") {
             if (!parsed.data.judge_passed) {
               state.messages[assistantMsgIndex].text = "⚠️ [Tin nhắn đã bị chặn do luật an toàn].";
               renderApp();
             }
          }
        } catch (e) { console.error(e, block); }
      }
    }
  } catch (error) {
    state.messages[assistantMsgIndex].text += `\n⚠️ Error: ${String(error)}`;
    renderApp();
  } finally {
    state.sending = false;
    statusDot.classList.remove("busy");
    statusText.textContent = "Ready";
  }
}

// Admin Logic
async function renderAdminData() {
  const panel = document.getElementById("adminPanelWrapper");
  if (!panel) return;
  
  const res = await fetchWithAuth("/admin/users", { headers: { Authorization: `Bearer ${state.accessToken}` } });
  if (!res.ok) {
    panel.innerHTML = `<div style="color:var(--danger)">Failed to load users</div>`;
    return;
  }
  
  state.adminUsers = await res.json();
  panel.innerHTML = `
    <div class="admin-header">
      <h2>Role Management</h2>
      <p>Secure administrative control center.</p>
    </div>
    <div class="user-table">
      ${state.adminUsers.map(u => `
        <div class="user-row">
          <div class="user-identity">
            <span class="user-list-name">${u.full_name}</span>
            <span class="user-list-email">${u.email}</span>
          </div>
          <div>
            ${u.role === 'admin' 
              ? `<span class="badge-admin">Admin</span>`
              : `<button class="btn btn-danger btn-sm" onclick="window.deleteUser(${u.id})">Remove</button>`
            }
          </div>
        </div>
      `).join('')}
    </div>
  `;
}

;(window as any).deleteUser = async (id: number) => {
  if(!confirm("Are you sure?")) return;
  await fetchWithAuth(`/admin/users/${id}`, { method: "DELETE" });
  renderApp();
};


// Auth Logic
function setTokens(t: string, r: string) {
  state.accessToken = t; state.refreshToken = r;
  localStorage.setItem("accessToken", t); localStorage.setItem("refreshToken", r);
}
function logout() {
  state.user = null; state.messages = [];
  setTokens("", "");
  switchView("auth");
}
async function authRequest(path: string, payload: any) {
  const res = await fetch(path, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload) });
  if (!res.ok) throw new Error(getApiErrorMessage(await res.text(), res.status));
  const data = await res.json();
  setTokens(data.access_token, data.refresh_token);
  await loadCurrentUser();
}
async function login() {
  const email = getEl<HTMLInputElement>("emailInput").value.trim();
  const password = getEl<HTMLInputElement>("passwordInput").value.trim();
  if(!email || !password) return alert("Please fill all fields");
  try { await authRequest("/auth/login", { email, password }); switchView("chat"); }
  catch(e) { alert(String(e)); }
}
async function register() {
  const email = getEl<HTMLInputElement>("emailInput").value.trim();
  const password = getEl<HTMLInputElement>("passwordInput").value.trim();
  const full_name = getEl<HTMLInputElement>("fullNameInput").value.trim();
  if(!email || !password || !full_name) return alert("Please fill all fields");
  try { await authRequest("/auth/register", { email, password, full_name }); switchView("chat"); }
  catch(e) { alert(String(e)); }
}

async function refreshAccessToken() {
  if (!state.refreshToken) return false;
  const res = await fetch("/auth/refresh", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({refresh_token:state.refreshToken})});
  if (!res.ok) { logout(); return false; }
  const data = await res.json(); setTokens(data.access_token, data.refresh_token);
  return true;
}
async function fetchWithAuth(url: string, init: RequestInit) {
  let res = await fetch(url, init);
  if (res.status === 401 && await refreshAccessToken()) {
    res = await fetch(url, { ...init, headers: { ...init.headers, Authorization: `Bearer ${state.accessToken}`} });
  }
  return res;
}
async function loadCurrentUser() {
  if (!state.accessToken) { switchView("auth"); return; }
  const res = await fetchWithAuth("/auth/me", { headers:{ Authorization: `Bearer ${state.accessToken}`} });
  if (!res.ok) { logout(); return; }
  state.user = await res.json();
  switchView("chat");
}

loadCurrentUser();
