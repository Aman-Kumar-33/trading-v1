let timer;
export function toast(msg, isError = false) {
  const el = document.getElementById("toast");
  if (!el) return;
  el.textContent = msg;
  el.className = isError ? "show error" : "show";
  clearTimeout(timer);
  timer = setTimeout(() => { el.className = ""; }, 3000);
}
