"""Row counts of every table in the SuperLink state after a completed run."""
import os, sqlite3

db = os.path.expanduser("~/.flwr/local-superlink/state.db")
con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
cur = con.cursor()

tables = [r[0] for r in cur.execute(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]

print(f"{'table':34}{'rows':>6}")
print("-" * 48)
for t in tables:
    n = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    mark = "" if n else "   <-- EMPTY"
    print(f"{t:34}{n:>6}{mark}")

print()
print("=== the four that matter ===")
labels = {
    "message_ins":  "instruction messages sent to clients",
    "message_res":  "reply messages from clients  <-- the client submissions live here",
    "task_message": "task-level messages",
    "objects":      "object store (where ArrayRecords would be held)",
}
for t, desc in labels.items():
    n = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"{t:14} {n:>4} rows   {desc}")
