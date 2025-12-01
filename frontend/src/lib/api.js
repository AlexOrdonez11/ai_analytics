export const API_BASE = "https://analytics-endpoints-885186858021.us-central1.run.app";

async function request(path, { method = "GET", body, headers = {} } = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers: { "Content-Type": "application/json", ...headers },
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data?.detail || res.statusText || "Request failed");
  return data;
}

export const api = {
  // auth 
  loginEmail: (email, password) => request("/login", { method: "POST", body: { email, password } }),
  register: (payload) => request("/users", { method: "POST", body: payload }),
  meByEmail: (email) => request(`/user?email=${encodeURIComponent(email)}`),

  // projects
  listProjectsByUser: (userId) => request(`/projects/user/${encodeURIComponent(userId)}`),
  getProject: (projectId) => request(`/projects/${encodeURIComponent(projectId)}`),
  createProject: ({ name, description, user_id }) =>
    request("/projects", { method: "POST", body: { name, description, user_id } }),
  updateProject: (projectId, patch) =>
    request(`/projects/${encodeURIComponent(projectId)}`, { method: "PUT", body: patch }),
  deleteProject: (projectId) =>
    request(`/projects/${encodeURIComponent(projectId)}`, { method: "DELETE" }),
  // plots
  listProjectPlots: ({ project_id }) =>
    request(`/projects/${encodeURIComponent(project_id)}/plots`),

  // conversations
  listConversations: ({ project_id, skip = 0, limit = 100, sort_asc = false }) =>
    request(`/conversations/${encodeURIComponent(project_id)}?skip=${skip}&limit=${limit}&sort_asc=${sort_asc}`),
  createConversation: ({ project_id, role, message }) =>
    request("/conversations", { method: "POST", body: { project_id, role, message } }),
  deleteConversation: (conversation_id) =>
    request(`/conversations/${encodeURIComponent(conversation_id)}`, { method: "DELETE" }),

    // analyst
    analystChat: async ({ project_id, message, dataset_id=null }) => {
        const res = await fetch(`${API_BASE}/analyst/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ project_id, message, dataset_id }),
        })
        const data = await res.json().catch(() => ({}))
        if (!res.ok) throw new Error(data?.detail || "Analyst chat failed")
        return data
    },
};

export const datasetsApi = {
  uploadDataset: async ({ project_id, file, user_id }) => {
    const form = new FormData()
    form.append('project_id', project_id)
    if (user_id) form.append('user_id', user_id)
    form.append('file', file)
    const res = await fetch(`${API_BASE}/datasets/upload`, { method: 'POST', body: form })
    const data = await res.json().catch(() => ({}))
    if (!res.ok) throw new Error(data?.detail || 'Upload failed')
    return data
  },
  listProjectDatasets: async (project_id) => {
    const res = await fetch(`${API_BASE}/projects/${encodeURIComponent(project_id)}/datasets`)
    const data = await res.json().catch(() => ({}))
    if (!res.ok) throw new Error(data?.detail || 'List datasets failed')
    return data
  },
  getSignedDownloadUrl: async (dataset_id) => {
    const res = await fetch(`${API_BASE}/datasets/${encodeURIComponent(dataset_id)}/signed-url`)
    const data = await res.json().catch(() => ({}))
    if (!res.ok) throw new Error(data?.detail || 'Signed URL failed')
    return data.url
  },
  deleteDataset: async (dataset_id) => {
    const res = await fetch(`${API_BASE}/datasets/${encodeURIComponent(dataset_id)}`, { method: 'DELETE' })
    const data = await res.json().catch(() => ({}))
    if (!res.ok) throw new Error(data?.detail || 'Delete failed')
    return data
  },
}