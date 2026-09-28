# Reproducible environment for fl-aggregation-evidence.
# Python 3.13 matches the interpreter the committed evidence bundles were
# generated under. The version matters: from Python 3.12, comprehensions no
# longer run in their own frame (PEP 709), so the E3 execution record differs
# under 3.11 even though every verdict is unchanged.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Dependencies first, so code changes do not invalidate this layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN chmod +x run.sh

ENTRYPOINT ["./run.sh"]
CMD ["track1"]
