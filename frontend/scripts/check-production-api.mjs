const value = process.env.VITE_API_URL?.trim()
let url
try {
  url = new URL(value)
} catch {
  console.error('Set VITE_API_URL to the public HTTPS origin of the FastAPI server before deploying to Netlify.')
  process.exit(1)
}
if (url.protocol !== 'https:' || url.username || url.password || url.search || url.hash || url.pathname !== '/' || ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname)) {
  console.error('VITE_API_URL must be a public HTTPS origin, without credentials, paths, query strings or fragments.')
  process.exit(1)
}
console.log('Production API address configured. The FastAPI server must be deployed separately.')
