"""
Live Training Tracker — Telegram & Discord integration.

Sends real-time training updates (epoch summaries, NaN alerts, checkpoint saves,
probe results) directly to your phone/desktop via Telegram or Discord webhooks.

Setup (Telegram):
    1. Message @BotFather on Telegram → /newbot → copy the token
    2. Message your bot, then visit:
       https://api.telegram.org/bot<TOKEN>/getUpdates
       to find your chat_id
    3. Set in config YAML:
       logging:
         telegram_token: "123456:ABC-DEF..."
         telegram_chat_id: "987654321"
    
    OR set environment variables:
       TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
       TELEGRAM_CHAT_ID=987654321

Setup (Discord):
    1. Server Settings → Integrations → Webhooks → New Webhook → Copy URL
    2. Set in config YAML:
       logging:
         discord_webhook: "https://discord.com/api/webhooks/..."

    OR set environment variable:
       DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
"""

import os
import time
import threading
import requests
from datetime import datetime, timedelta
from collections import deque


class LiveTracker:
    """Sends training events to Telegram and/or Discord in real-time.

    All sends are non-blocking (fire-and-forget in a background thread)
    so they never slow down training. Failures are silently logged to
    stderr — training never crashes due to network issues.

    Rate limiting: max 1 message per `min_interval` seconds (default 5s).
    Messages that arrive faster are queued and batched.
    """

    # Rotating emoji pool for visually distinguishing parallel runs
    _RUN_EMOJIS = ["🔵", "🟢", "🟠", "🔴", "🟣", "🟡", "⚪", "🟤"]

    def __init__(self, config=None, min_interval=5):
        # --- Telegram ---
        self.tg_token = None
        self.tg_chat_id = None
        # --- Discord ---
        self.discord_webhook = None
        # --- State ---
        self._enabled = False
        self._last_send = 0
        self._queue = deque(maxlen=50)  # Buffer messages if rate-limited
        self._lock = threading.Lock()
        self.min_interval = min_interval
        self.run_name = "Training"
        self.run_label = "Training"  # Short label for message prefix
        self._start_time = time.time()

        # Load from config or env vars
        if config is not None:
            logging_cfg = config.get('logging', {})
            self.tg_token = logging_cfg.get('telegram_token', None) or os.environ.get('TELEGRAM_BOT_TOKEN')
            self.tg_chat_id = logging_cfg.get('telegram_chat_id', None) or os.environ.get('TELEGRAM_CHAT_ID')
            self.discord_webhook = logging_cfg.get('discord_webhook', None) or os.environ.get('DISCORD_WEBHOOK_URL')
            self.run_name = logging_cfg.get('run_name', config.get('experiment_name', 'Training'))

            # Build a unique label from tags so parallel runs are distinguishable
            # e.g. tags=["mobilenet","tsm","phase1a","scda","dann"] → "DANN"
            #      tags=["mobilenet","tsm","phase1a","scda","mmd"]  → "MMD"
            tags = list(logging_cfg.get('tags', []))
            # Pick the most distinguishing tag (last one, or unique ones)
            # Common tags across runs are filtered out by using the last tag
            if tags:
                # Use last tag as the differentiator (convention: method name goes last)
                method_tag = tags[-1].upper()
                self.run_label = f"{self.run_name}/{method_tag}"
            else:
                self.run_label = self.run_name

            # Assign a deterministic emoji based on the label hash
            emoji_idx = hash(self.run_label) % len(self._RUN_EMOJIS)
            self._emoji = self._RUN_EMOJIS[emoji_idx]
        else:
            self.tg_token = os.environ.get('TELEGRAM_BOT_TOKEN')
            self.tg_chat_id = os.environ.get('TELEGRAM_CHAT_ID')
            self.discord_webhook = os.environ.get('DISCORD_WEBHOOK_URL')
            self._emoji = self._RUN_EMOJIS[0]

        if self.tg_token and self.tg_chat_id:
            self._enabled = True
            print(f"[LiveTracker] Telegram enabled — run: {self._emoji} {self.run_label}")
        if self.discord_webhook:
            self._enabled = True
            print(f"[LiveTracker] Discord webhook enabled — run: {self._emoji} {self.run_label}")

        if not self._enabled:
            print("[LiveTracker] No Telegram/Discord credentials found — live tracking disabled.")
            print("  Set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID env vars, or add to config YAML.")

    def _prefix(self, msg):
        """Prefix a message with the run label for multi-run identification."""
        return f"{self._emoji} *[{self.run_label}]*\n{msg}"

    @property
    def enabled(self):
        return self._enabled

    def _elapsed(self):
        """Human-readable elapsed time since training started."""
        secs = int(time.time() - self._start_time)
        return str(timedelta(seconds=secs))

    # ──────────────── Low-Level Send ──────────────── #

    def _send_telegram(self, text, parse_mode="Markdown"):
        """Send a message via Telegram Bot API."""
        try:
            url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
            payload = {
                "chat_id": self.tg_chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            }
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code != 200:
                print(f"[LiveTracker] Telegram error {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"[LiveTracker] Telegram send failed: {e}")

    def _send_telegram_image(self, image_path, caption=""):
        """Send an image via Telegram Bot API."""
        try:
            url = f"https://api.telegram.org/bot{self.tg_token}/sendPhoto"
            with open(image_path, 'rb') as f:
                files = {'photo': f}
                data = {
                    "chat_id": self.tg_chat_id,
                    "caption": caption[:1024],  # Telegram caption limit
                }
                requests.post(url, data=data, files=files, timeout=30)
        except Exception as e:
            print(f"[LiveTracker] Telegram image send failed: {e}")

    def _send_discord(self, text):
        """Send a message via Discord webhook."""
        try:
            # Discord has 2000 char limit
            payload = {"content": text[:2000]}
            resp = requests.post(self.discord_webhook, json=payload, timeout=10)
            if resp.status_code not in (200, 204):
                print(f"[LiveTracker] Discord error {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            print(f"[LiveTracker] Discord send failed: {e}")

    def _dispatch(self, text, urgent=False):
        """Non-blocking dispatch to all enabled channels.
        
        Respects rate limiting unless `urgent` is True.
        """
        if not self._enabled:
            return

        now = time.time()
        with self._lock:
            if not urgent and (now - self._last_send) < self.min_interval:
                self._queue.append(text)
                return
            
            # Flush queue if we have buffered messages
            if self._queue:
                batched = "\n\n".join(self._queue)
                self._queue.clear()
                text = batched + "\n\n" + text

            self._last_send = now

        # Fire-and-forget in background thread
        def _send():
            if self.tg_token and self.tg_chat_id:
                self._send_telegram(text)
            if self.discord_webhook:
                self._send_discord(text)

        t = threading.Thread(target=_send, daemon=True)
        t.start()

    def send_image(self, image_path, caption=""):
        """Send an image (UMAP plot, loss curve, etc.) to all channels."""
        if not self._enabled or not os.path.exists(image_path):
            return

        def _send():
            if self.tg_token and self.tg_chat_id:
                self._send_telegram_image(image_path, caption)
            # Discord webhook image sending requires multipart form
            if self.discord_webhook:
                try:
                    with open(image_path, 'rb') as f:
                        files = {'file': (os.path.basename(image_path), f)}
                        payload = {"content": caption[:2000]} if caption else {}
                        requests.post(self.discord_webhook, data=payload, files=files, timeout=30)
                except Exception as e:
                    print(f"[LiveTracker] Discord image send failed: {e}")

        t = threading.Thread(target=_send, daemon=True)
        t.start()

    # ──────────────── High-Level Events ──────────────── #

    def training_started(self, phase, total_epochs, device_info=""):
        """Call at the very start of a training phase."""
        self._start_time = time.time()
        msg = self._prefix(
            f"🚀 Phase *{phase}* started\n"
            f"📊 {total_epochs} epochs\n"
            f"🖥️ {device_info}\n"
            f"⏰ {datetime.now().strftime('%H:%M:%S')}"
        )
        self._dispatch(msg, urgent=True)

    def epoch_summary(self, phase, epoch, total_epochs, metrics):
        """Send epoch-level metrics summary.
        
        Args:
            metrics: dict of metric_name -> value, e.g.
                     {"train_loss": 0.42, "val_acc": 0.85, "lr": 1e-3}
        """
        elapsed = self._elapsed()
        progress = f"{epoch+1}/{total_epochs}"
        pct = int(100 * (epoch + 1) / total_epochs)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)

        lines = [f"📈 *Epoch {progress}* ({pct}%) `[{bar}]`"]
        lines.append(f"⏱️ {elapsed}")

        for key, val in metrics.items():
            if isinstance(val, float):
                if abs(val) < 0.01 or abs(val) > 1000:
                    lines.append(f"  • {key}: `{val:.2e}`")
                else:
                    lines.append(f"  • {key}: `{val:.4f}`")
            else:
                lines.append(f"  • {key}: `{val}`")

        self._dispatch(self._prefix("\n".join(lines)))

    def checkpoint_saved(self, path, metric_name="", metric_value=0.0):
        """Notify when a new best checkpoint is saved."""
        filename = os.path.basename(path)
        msg = self._prefix(
            f"💾 *Checkpoint saved*\n"
            f"  `{filename}`\n"
            f"  {metric_name}: `{metric_value:.4f}`"
        )
        self._dispatch(msg, urgent=True)

    def probe_results(self, phase, epoch, probes):
        """Send offline probe results.
        
        Args:
            probes: dict like {"train_cv": 0.45, "val_linear": 0.38, ...}
        """
        lines = [f"🔬 *Probes* (Epoch {epoch})"]
        for key, val in probes.items():
            if isinstance(val, float):
                lines.append(f"  • {key}: `{val:.4f}`")
            else:
                lines.append(f"  • {key}: `{val}`")
        self._dispatch(self._prefix("\n".join(lines)))

    def nan_alert(self, location, diagnosis, details=""):
        """URGENT: NaN/Inf detected. Sends immediately."""
        msg = (
            f"🚨 *NaN/Inf ALERT*\n"
            f"📍 {location}\n"
            f"🔍 {diagnosis}\n"
        )
        if details:
            # Truncate for Telegram (4096 char limit)
            msg += f"```\n{details[:500]}\n```"
        self._dispatch(self._prefix(msg), urgent=True)

    def dann_status(self, epoch, batch_idx, alpha, acc, loss):
        """Report DANN training status."""
        msg = self._prefix(
            f"🎭 *DANN* (Ep {epoch} B{batch_idx})\n"
            f"  α={alpha:.4f} | Acc={acc:.4f} | Loss={loss:.4f}"
        )
        self._dispatch(msg)

    def training_finished(self, phase, best_metric_name="", best_metric_value=0.0):
        """Call when training phase completes."""
        elapsed = self._elapsed()
        msg = self._prefix(
            f"✅ Phase *{phase}* complete!\n"
            f"⏱️ Total time: {elapsed}\n"
            f"🏆 Best {best_metric_name}: `{best_metric_value:.4f}`"
        )
        self._dispatch(msg, urgent=True)

    def training_crashed(self, phase, error_msg):
        """Call when training crashes with an exception."""
        elapsed = self._elapsed()
        msg = self._prefix(
            f"💥 *CRASHED*\n"
            f"Phase *{phase}*\n"
            f"⏱️ After: {elapsed}\n"
            f"```\n{str(error_msg)[:800]}\n```"
        )
        self._dispatch(msg, urgent=True)

    def custom(self, message):
        """Send a custom freeform message."""
        self._dispatch(self._prefix(message))

    def flush(self):
        """Flush any buffered messages. Call before exit."""
        if not self._enabled:
            return
        with self._lock:
            if self._queue:
                batched = "\n\n".join(self._queue)
                self._queue.clear()
                if self.tg_token and self.tg_chat_id:
                    self._send_telegram(batched)
                if self.discord_webhook:
                    self._send_discord(batched)
