// Backend API istemcisi. Vite proxy sayesinde /api -> http://localhost:8000

export async function uploadVideo(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/videos", { method: "POST", body: form });
  if (!res.ok) throw new Error((await res.json()).detail || "Yükleme başarısız");
  return res.json();
}

export async function getJob(jobId) {
  const res = await fetch(`/api/jobs/${jobId}`);
  if (!res.ok) throw new Error("İş durumu alınamadı");
  return res.json();
}

export async function getEvents(jobId) {
  const res = await fetch(`/api/jobs/${jobId}/events`);
  if (!res.ok) throw new Error("Event'ler alınamadı");
  return res.json();
}

export async function getCommentary(jobId) {
  const res = await fetch(`/api/jobs/${jobId}/commentary`);
  if (!res.ok) return null;
  return res.json();
}
