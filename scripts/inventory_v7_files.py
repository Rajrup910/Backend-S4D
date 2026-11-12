import os
import hashlib
import datetime

files = [
    'docs/V7_MASTER_RESEARCH_PLAN.md',
    '.relay/HANDOFF.md',
    '.relay/GEMINI_NOTES.md',
    'CHANGELOG.md',
    'scripts/audit_v7_final_verification.py'
]
if os.path.exists('docs/v7_final_revision'):
    for fname in sorted(os.listdir('docs/v7_final_revision')):
        files.append(os.path.join('docs/v7_final_revision', fname).replace('\\', '/'))

print(f"{'Path':<60} | {'Size (B)':<10} | {'Last Modified':<19} | {'SHA256'}")
print("-" * 160)
for f in files:
    if os.path.exists(f):
        sz = os.path.getsize(f)
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(f)).strftime('%Y-%m-%d %H:%M:%S')
        with open(f, 'rb') as fp:
            h = hashlib.sha256(fp.read()).hexdigest()
        print(f"{f:<60} | {sz:<10} | {mtime:<19} | {h}")
    else:
        print(f"{f:<60} | {'MISSING':<10} | {'MISSING':<19} | {'MISSING'}")

