# Deploying

The app is a Streamlit server. It needs a host that can **upgrade a WebSocket connection**,
which is the one requirement that rules out most quick-preview services — Streamlit's entire
frontend runs over a persistent socket, and a proxy that serves HTTP correctly but refuses
the upgrade produces a page that loads, shows a skeleton, and never connects. That failure
mode is silent from the server's side: health checks return 200 the whole time.

## Recommended: GitHub → Streamlit Community Cloud

Free, and it is the reference host for this framework, so the WebSocket path is not a
question mark.

1. Push this repository to GitHub with its structure intact. **Do not flatten it into a
   single file** — see "Why not one big app.py" below.
2. At [share.streamlit.io](https://share.streamlit.io), create an app from the repo.
3. Set **Main file path** to `app/main.py`.
4. Set **Python version** to **3.12 or 3.13**. Not 3.11: the pinned numpy publishes no 3.11
   wheel, so the install fails outright. This was checked by resolving the dependency set
   against each version rather than inferred from the syntax the code uses.
5. Add secrets (below), then deploy.

The source itself uses no syntax newer than 3.11 — no PEP 695 generics, no 3.12-only
stdlib — so the floor is set by the pinned dependencies, not by the code. If you ever need
3.11, relax `numpy==2.5.2` to `numpy>=1.26,<3` and re-run the suite before trusting it; the
parity constants are numeric and a major numpy change is not a free upgrade.

`requirements.txt` installs `./engine` as a package, so `import scrcae` resolves from
site-packages. This matters more than it looks: Streamlit Cloud gives you no way to set
`PYTHONPATH`, so the local `PYTHONPATH=.:engine/src` habit would fail there. Verified by
building the wheel and importing `scrcae.calibration.elasticity` from a clean target
directory with no source tree on the path.

### Secrets

Set these in the app's **Settings → Secrets** panel, not in the repo.

| Secret | What happens without it |
| --- | --- |
| `SCRCAE_USERS` | **No authentication.** Anyone with the URL is a user. See the warning below. |
| `SCRCAE_MARKET_PROVIDER` | Defaults to `yahoo`, a public but undocumented endpoint with no service commitment. Set to `api-ninjas` for anything real. |
| `SCRCAE_MARKET_API_KEY` | Required by the `api-ninjas` provider. Missing, the panel says the provider is unconfigured rather than showing a held default as telemetry. |
| `SUPABASE_URL`, `SUPABASE_KEY` | Falls back to local SQLite, which on Cloud is erased on every restart. |

Generate a `SCRCAE_USERS` value with `python -m app.auth.hash_password`. Plaintext
passwords are rejected, not warned about.

## Three things that will bite you on Cloud

1. **The filesystem is ephemeral.** `portfolios.db` is written to local disk, and Cloud wipes
   local disk on every restart, redeploy, and idle sleep. Saved portfolios will disappear
   without an error message, which is worse than not having persistence at all. Either wire
   Supabase or tell viewers plainly that it is a demo.
2. **There is no authentication by default.** Unauthenticated is a reasonable default for
   `localhost`; on a public URL it means anyone who has the link can read and write
   portfolios. Set `SCRCAE_USERS` before sharing the link, even for a demo.
3. **The default market provider has no service commitment.** Yahoo's chart endpoint is
   public and undocumented. It is the default only so the app is never dead on arrival
   without a credential. Anything a decision gets made on should run through `api-ninjas`
   with a key.

## Before making the repository public

This repository is private. Two things in [`legacy/`](legacy/) are internal documents rather
than product — the forensic audit of the acquired system and the rebuild blueprint — so read
them again before flipping the repository public and decide whether you want them visible.
Nothing in `legacy/` is imported by the app or the engine, so removing the directory breaks
no code; it only costs the provenance behind the F1–F18 citations in ADR-001.

Streamlit Cloud deploys from private repositories on the free tier, so staying private costs
nothing.

## Why not one big app.py

The tempting shortcut is to paste everything into a single file and run that. The reason not
to is sitting in this repository: [`legacy/acquired-system-v1.py`](legacy/acquired-system-v1.py)
**is** that file, 1,146 lines of it, and the
entire rebuild exists because eighteen separate defects were hiding in it — a solver that
reported infeasible as optimal, a correlation repair that silently rescaled every node, a
market mapping that matched everything when a column was missing. Every one of those is now
pinned by a test, and the tests are only possible because the engine is importable without
Streamlit and the app is a client of it.

Flattening the repo would throw away, concretely:

- the 672-test suite, which cannot import a Streamlit script
- the engine/app separation that lets the solver be exercised headlessly
- the audit ledger's provenance rows, which are computed across module boundaries
- the ability to swap the quote provider, which is what made the feed testable offline

GitHub does not want one file. It wants the repository, which is what this already is.

## Other hosts

If Cloud is ever too limiting, anything that terminates WebSockets works. Fly.io, Render,
and Cloud Run all do, with a container roughly:

```dockerfile
FROM python:3.12-slim
WORKDIR /srv
COPY requirements.txt ./
COPY engine ./engine
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["python", "-m", "streamlit", "run", "app/main.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
```

Leave `--server.enableCORS` and `--server.enableXsrfProtection` at their defaults. They were
disabled once, in a local sandbox, to chase a proxy problem, and that must not follow the
app to a public URL. `.streamlit/config.toml` in this repo leaves both on.

## Verifying a deployment

`tools/seed_demo_portfolio.py` writes the demo portfolio and
`tools/capture_walkthrough.py` drives the whole market-exposure workflow in a real browser,
so a deployment can be checked the same way the screenshots in `app/docs/walkthrough/` were
produced:

```bash
WALKTHROUGH_URL=https://your-app.streamlit.app/ \
  python tools/capture_walkthrough.py
```

Seeding writes to the server's local database, so against a remote deployment you would save
the portfolio through the UI instead.
