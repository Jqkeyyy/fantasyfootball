FROM ghcr.io/astral-sh/uv:python3.11-trixie-slim
WORKDIR /app

# lightgbm's wheel dynamically links libgomp.so.1, which the slim base
# image doesn't include -- without this, `import lightgbm` fails at
# runtime with "libgomp.so.1: cannot open shared object file".
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, cached separately from source changes (uv's own
# documented Docker pattern: https://docs.astral.sh/uv/guides/integration/docker/).
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-dev

COPY . .
RUN uv sync --locked --no-dev
RUN chmod +x weekly-refresh.sh

# Streamlit's first-run behavior with no ~/.streamlit/credentials.toml is to
# print a "Welcome to Streamlit / enter your email" prompt and block on
# stdin forever, with no visible error -- start-app.bat hits and fixes the
# same thing on the Windows path; pre-creating this file is the same fix.
RUN mkdir -p /root/.streamlit && printf '[general]\nemail = ""\n' > /root/.streamlit/credentials.toml
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501
CMD ["uv", "run", "streamlit", "run", "src/ffapp/app/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]
