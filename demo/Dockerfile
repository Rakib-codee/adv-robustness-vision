FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# CPU-only torch wheels come from the extra index already in requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Checkpoints are gitignored (see repo .gitignore); fetch them at build time
# from the public HF Hub model repo instead of shipping them in git.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p checkpoints \
    && for m in resnet18_std resnet18_fgsm_at resnet18_pgd_at resnet18_trades_b6; do \
         curl -fsSL "https://huggingface.co/nexgendev/adv-robustness-vision-ckpts/resolve/main/${m}.pt" -o "checkpoints/${m}.pt"; \
       done

COPY . .

ENV PORT=7860
EXPOSE 7860
CMD ["python", "app.py"]
