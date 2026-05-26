# Add Hindi Jinnie to your Gatsby portfolio (Netlify)

Netlify hosts **static files only**. The model runs on a separate API server.

## 1. Copy frontend into your Gatsby repo

From this repo:

```bash
cp -r serving/web/* /path/to/your-gatsby-site/static/hindi-jinnie/
```

Or as a Gatsby page — copy assets to `static/hindi-jinnie/` so they deploy at:

`https://projects.tarun-ssharma.com/hindi-jinnie/`

## 2. Set production API URL

Edit `static/hindi-jinnie/config.js` in your Gatsby repo:

```javascript
window.HINDI_JINNIE_API = "https://YOUR-RAILWAY-OR-RUNPOD-API-URL";
```

Do **not** commit secrets — only the public API URL.

## 3. Optional — add a portfolio link

In your Gatsby nav or projects list:

```markdown
[Hindi Jinnie](/hindi-jinnie/)
```

## 4. Deploy

Push to GitHub → Netlify rebuilds as usual.

## CORS

The API must allow your Netlify origin. Default in `serving/api/app.py`:

`https://projects.tarun-ssharma.com`

Set env on API server if you use a different subdomain:

```bash
export CORS_ORIGINS="https://projects.tarun-ssharma.com,https://hindi-jinnie.tarun-ssharma.com"
```
