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
import json
import tempfile


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
        else:
            if self.tg_token:
                threading.Thread(target=self._poll_telegram_commands, daemon=True).start()

    def _prefix(self, msg):
        """Prefix a message with the run label for multi-run identification."""
        return f"{self._emoji} <b>[{self.run_label}]</b>\n{msg}"

    @property
    def enabled(self):
        return self._enabled

    def _elapsed(self):
        """Human-readable elapsed time since training started."""
        secs = int(time.time() - self._start_time)
        return str(timedelta(seconds=secs))

    # ──────────────── Shared State & Interactive Bot ──────────────── #

    def _update_state(self, updates_dict):
        """Update shared state file with our latest metrics so the interactive bot can read it."""
        if not self.tg_token: return
        state_file = os.path.join(tempfile.gettempdir(), "snn_tracker_state.json")
        try:
            state = {}
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
            
            if self.run_label not in state:
                state[self.run_label] = {"emoji": self._emoji, "start_time": self._start_time}
            
            state[self.run_label].update(updates_dict)
            state[self.run_label]["last_updated"] = time.time()
            
            # Clean up old runs (e.g. haven't updated in 12 hours)
            now = time.time()
            state = {k: v for k, v in state.items() if (now - v.get("last_updated", 0)) < 43200}
            
            with open(state_file, 'w') as f:
                json.dump(state, f)
        except Exception:
            pass

    def _poll_telegram_commands(self):
        """Daemon thread that acts as a simple Telegram interactive bot. 
        Uses a lock so only one running script polls the Telegram API."""
        lock_path = os.path.join(tempfile.gettempdir(), "snn_telegram_poller.lock")
        offset = 0
        while True:
            is_master = False
            try:
                if os.path.exists(lock_path):
                    if time.time() - os.path.getmtime(lock_path) > 15:
                        os.remove(lock_path)
                
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                is_master = True
            except (FileExistsError, OSError):
                pass
            
            if not is_master:
                time.sleep(10)
                continue
                
            try:
                # We are the master poller!
                while True:
                    os.utime(lock_path, None) # keep lock alive
                    url = f"https://api.telegram.org/bot{self.tg_token}/getUpdates?timeout=5&offset={offset}"
                    resp = requests.get(url, timeout=10)
                    if resp.status_code == 200:
                        updates = resp.json().get('result', [])
                        for u in updates:
                            offset = max(offset, u['update_id'] + 1)
                            self._handle_telegram_update(u)
                    else:
                        time.sleep(2)
            except Exception:
                try:
                    os.remove(lock_path)
                except Exception: pass
                time.sleep(5)

    def _handle_telegram_update(self, update):
        if 'message' in update and 'text' in update['message']:
            text = update['message']['text']
            chat_id = update['message']['chat']['id']
            if str(chat_id) != str(self.tg_chat_id): return
            
            if text.startswith('/status') or text.startswith('/runs'):
                self._send_status_dashboard()
                
        elif 'callback_query' in update:
            cb = update['callback_query']
            chat_id = cb['message']['chat']['id']
            msg_id = cb['message']['message_id']
            if str(chat_id) != str(self.tg_chat_id): return
            
            data = cb['data']
            if data.startswith("status:"):
                run_id = data.split("status:")[1]
                self._send_detailed_status(run_id, cb['id'], msg_id)
            elif data.startswith("warnings:"):
                run_id = data.split("warnings:")[1]
                self._send_warnings(run_id, cb['id'])

    def _send_status_dashboard(self):
        state_file = os.path.join(tempfile.gettempdir(), "snn_tracker_state.json")
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
        except Exception:
            state = {}
            
        if not state:
            self._send_telegram("No active runs found.")
            return
            
        old_msg_id = state.get('_meta', {}).get('status_msg_id')
        if old_msg_id:
            try:
                requests.post(f"https://api.telegram.org/bot{self.tg_token}/deleteMessage",
                              json={"chat_id": self.tg_chat_id, "message_id": old_msg_id}, timeout=3)
            except Exception: pass
            
        buttons = []
        for run_id, run_data in state.items():
            if run_id == '_meta': continue
            emoji = run_data.get('emoji', '🏃')
            btn = {"text": f"{emoji} {run_id}", "callback_data": f"status:{run_id}"}
            buttons.append([btn])
            
        reply_markup = {"inline_keyboard": buttons}
        
        url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
        payload = {
            "chat_id": self.tg_chat_id,
            "text": "📊 <b>Active Training Runs</b>\nClick a run to fetch its current live state:",
            "parse_mode": "HTML",
            "reply_markup": reply_markup
        }
        try:
            resp = requests.post(url, json=payload, timeout=5).json()
            if resp.get('ok'):
                state.setdefault('_meta', {})['status_msg_id'] = resp['result']['message_id']
                with open(state_file, 'w') as f:
                    json.dump(state, f)
        except Exception: pass

    def _send_detailed_status(self, run_id, callback_id, callback_msg_id=None):
        # Acknowledge callback
        try:
            requests.post(f"https://api.telegram.org/bot{self.tg_token}/answerCallbackQuery", 
                          json={"callback_query_id": callback_id}, timeout=5)
        except Exception: pass
        
        state_file = os.path.join(tempfile.gettempdir(), "snn_tracker_state.json")
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
        except Exception:
            state = {}
            
        if run_id not in state:
            return
            
        rd = state[run_id]
        emoji = rd.get('emoji', '🏃')
        phase = rd.get('phase', 'Unknown')
        epoch = rd.get('epoch', '?')
        total_epochs = rd.get('total_epochs', '?')
        batch = rd.get('batch', '?')
        total_batches = rd.get('total_batches', '?')
        loss = rd.get('loss', '?')
        
        lines = [f"{emoji} <b>[{run_id}] Live Status</b>"]
        lines.append(f"Phase: <code>{phase}</code>")
        if epoch != '?':
            lines.append(f"Epoch: <code>{epoch}/{total_epochs}</code> (Batch <code>{batch}/{total_batches}</code>)")
        if loss != '?':
            lines.append(f"Current Loss: <code>{loss:.4f}</code>")
            
        # Display any extra metrics passed via update_batch_state (like EMA dots)
        ignore_keys = {'emoji', 'phase', 'epoch', 'total_epochs', 'batch', 'total_batches', 'loss', 'last_updated', 'warnings', 'status', 'start_time'}
        for k, v in rd.items():
            if k not in ignore_keys:
                if isinstance(v, float):
                    lines.append(f"{k}: <code>{v:.4f}</code>")
                else:
                    lines.append(f"{k}: <code>{v}</code>")
            
        warnings = rd.get('warnings', [])
        if warnings:
            lines.append(f"⚠️ <b>Warnings</b>: <code>{len(warnings)}</code> (Collapse, etc.)")
                
        last_upd = time.time() - rd.get('last_updated', time.time())
        if last_upd > 120:
            lines.append(f"\n⚠️ <i>Last update was {int(last_upd)}s ago (might be frozen/evaluating)</i>")
        else:
            lines.append(f"\n<i>Updated {int(last_upd)}s ago</i>")
            
        # Extract existing buttons to keep the dashboard usable!
        buttons = []
        for rid, rdata in state.items():
            if rid == '_meta': continue
            e = rdata.get('emoji', '🏃')
            buttons.append([{"text": f"{e} {rid}", "callback_data": f"status:{rid}"}])
            
        if warnings:
            buttons.insert(0, [{"text": "⚠️ Fetch Warnings", "callback_data": f"warnings:{run_id}"}])
            
        payload = {
            "chat_id": self.tg_chat_id,
            "message_id": callback_msg_id,
            "text": "\n".join(lines),
            "parse_mode": "HTML",
            "reply_markup": {"inline_keyboard": buttons}
        }
        try:
            requests.post(f"https://api.telegram.org/bot{self.tg_token}/editMessageText", json=payload, timeout=5)
        except Exception: pass

    def _send_warnings(self, run_id, callback_id):
        try:
            requests.post(f"https://api.telegram.org/bot{self.tg_token}/answerCallbackQuery", 
                          json={"callback_query_id": callback_id}, timeout=5)
        except Exception: pass
        
        state_file = os.path.join(tempfile.gettempdir(), "snn_tracker_state.json")
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
        except Exception: return
        
        if run_id not in state: return
        
        warnings = state[run_id].get('warnings', [])
        if not warnings:
            self._send_telegram(f"No warnings for `{run_id}`.")
            return
            
        # Send last 10 warnings
        msg = f"⚠️ <b>[{run_id}] Recent Warnings</b>\n\n"
        for w in warnings[-10:]:
            msg += f"• {w}\n"
            
        self._send_telegram(msg)

    # ──────────────── Low-Level Send ──────────────── #

    def _send_telegram(self, text, parse_mode="HTML"):
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
                err_msg = resp.text
                if "can't parse entities" in err_msg and parse_mode:
                    # Ultimate fallback: strip all formatting, send as plain text
                    import re
                    clean = re.sub(r'<[^>]+>', '', text)
                    payload['text'] = clean
                    payload.pop('parse_mode', None)
                    requests.post(url, json=payload, timeout=10)
                else:
                    print(f"[LiveTracker] Telegram error {resp.status_code}: {err_msg[:200]}")
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
        self._update_state({"status": "running", "phase": phase, "total_epochs": total_epochs})
        msg = self._prefix(
            f"🚀 Phase <b>{phase}</b> started\n"
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

        lines = [f"📈 <b>Epoch {progress}</b> ({pct}%) <code>[{bar}]</code>"]
        lines.append(f"⏱️ {elapsed}")

        for key, val in metrics.items():
            if isinstance(val, float):
                if abs(val) < 0.01 or abs(val) > 1000:
                    lines.append(f"  • {key}: <code>{val:.2e}</code>")
                else:
                    lines.append(f"  • {key}: <code>{val:.4f}</code>")
            else:
                lines.append(f"  • {key}: <code>{val}</code>")

        self._dispatch(self._prefix("\n".join(lines)))

    def update_batch_state(self, epoch, batch, total_batches, loss, **kwargs):
        """Silent update of current batch state for the interactive bot. Does not dispatch a message."""
        state_dict = {
            "epoch": epoch,
            "batch": batch, 
            "total_batches": total_batches,
            "loss": loss
        }
        state_dict.update(kwargs)
        self._update_state(state_dict)

    def add_warning(self, text):
        """Silently add a warning to the state for fetching via the dashboard."""
        if not self.tg_token: return
        state_file = os.path.join(tempfile.gettempdir(), "snn_tracker_state.json")
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            if self.run_label not in state: return
            warnings = state[self.run_label].get("warnings", [])
            warnings.append(text)
            
            # Keep only last 50 warnings to prevent file bloat
            state[self.run_label]["warnings"] = warnings[-50:]
            with open(state_file, 'w') as f:
                json.dump(state, f)
        except Exception: pass

    def checkpoint_saved(self, path, metric_name="", metric_value=0.0):
        """Notify when a new best checkpoint is saved."""
        filename = os.path.basename(path)
        msg = self._prefix(
            f"💾 <b>Checkpoint saved</b>\n"
            f"  <code>{filename}</code>\n"
            f"  {metric_name}: <code>{metric_value:.4f}</code>"
        )
        self._dispatch(msg, urgent=True)

    def probe_results(self, phase, epoch, probes):
        """Send offline probe results.
        
        Args:
            probes: dict like {"train_cv": 0.45, "val_linear": 0.38, ...}
        """
        lines = [f"🔬 <b>Probes</b> (Epoch {epoch})"]
        for key, val in probes.items():
            if isinstance(val, float):
                lines.append(f"  • {key}: <code>{val:.4f}</code>")
            else:
                lines.append(f"  • {key}: <code>{val}</code>")
        self._dispatch(self._prefix("\n".join(lines)))

    def nan_alert(self, location, diagnosis, details=""):
        """URGENT: NaN/Inf detected. Sends immediately."""
        msg = (
            f"🚨 <b>NaN/Inf ALERT</b>\n"
            f"📍 {location}\n"
            f"🔍 {diagnosis}\n"
        )
        if details:
            # Truncate for Telegram (4096 char limit)
            msg += f"<pre>\n{details[:3000]}\n</pre>"
        self._dispatch(self._prefix(msg), urgent=True)

    def dann_status(self, epoch, batch_idx, alpha, acc, loss):
        """Report DANN training status."""
        msg = self._prefix(
            f"🎭 <b>DANN</b> (Ep {epoch} B{batch_idx})\n"
            f"  α={alpha:.4f} | Acc={acc:.4f} | Loss={loss:.4f}"
        )
        self._dispatch(msg)

    def training_finished(self, phase, best_metric_name="", best_metric_value=0.0):
        """Call when training phase completes."""
        elapsed = self._elapsed()
        msg = self._prefix(
            f"✅ Phase <b>{phase}</b> complete!\n"
            f"⏱️ Total time: {elapsed}\n"
            f"🏆 Best {best_metric_name}: <code>{best_metric_value:.4f}</code>"
        )
        self._dispatch(msg, urgent=True)

    def training_crashed(self, phase, error_msg):
        """Call when training crashes with an exception."""
        elapsed = self._elapsed()
        msg = self._prefix(
            f"💥 <b>CRASHED</b>\n"
            f"Phase <b>{phase}</b>\n"
            f"⏱️ After: {elapsed}\n"
            f"<pre>\n{str(error_msg)[:800]}\n</pre>"
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
