import subprocess
import difflib
import json
import os
import re

REPOS = {
    'jinja': 'https://github.com/pallets/jinja.git',
    'httpx': 'https://github.com/encode/httpx.git',
    'mypy': 'https://github.com/python/mypy.git',
    'black': 'https://github.com/psf/black.git',
    'django': 'https://github.com/django/django.git',
    'fastapi': 'https://github.com/tiangolo/fastapi.git',
    'pytest': 'https://github.com/pytest-dev/pytest.git',
    'pip': 'https://github.com/pypa/pip.git',
}

BASE_DIR = 'repos'
os.makedirs(BASE_DIR, exist_ok=True)
OUTPUT_JSONL = 'bugfix_prompts.jsonl'
MAX_TOTAL = 8000
MAX_PER_REPO = 1000

GOOD_KEYWORDS = [
    r"\bfix(?:ed)?\b", r"\bbug\b", r"\btypo\b", r"this fixes #\d+", r"\bdecorator\b", r"\bsetup\.py\b",
    r"type hint", r"mypy", r"\btest\b", r"\bcontext\b", r"\bimport\b", r"\bapp\b", r"\bstatic_url_path\b"
]

BAD_KEYWORDS = [
    r"\bdoc[s]?\b", r"\bchangelog\b", r"\bscreenshot\b", r"\bgrammar\b", r"\breadability\b", r"\bformat\b",
    r"\btypo\b.+docs?", r"\bfixes.*doc", r"\bexample\b", r"\bstyle\b", r"\brst\b", r"\btest.*doc"
]

os.makedirs(BASE_DIR, exist_ok=True)

def is_good_commit(message):
    msg_lower = message.lower()
    if any(re.search(pattern, msg_lower) for pattern in BAD_KEYWORDS):
        return False
    return any(re.search(pattern, msg_lower) for pattern in GOOD_KEYWORDS)

def run(cmd): return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def clone_repo(name, url):
    path = os.path.join(BASE_DIR, name)
    if not os.path.exists(path):
        subprocess.run(["git", "clone", url, path])
    return path


def get_commits(path):
    log = run(["git", "-C", path, "log", "--pretty=format:%H %s", "-n", str(MAX_PER_REPO * 2)])
    return [line.split(maxsplit=1) for line in log.splitlines() if len(line.split()) > 1]


def get_files(repo, commit):
    out = run(["git", "-C", repo, "diff", "--name-only", f"{commit}^", commit])
    return [f for f in out.splitlines() if f.endswith('.py')]


def get_file(repo, commit, path):
    try: return run(["git", "-C", repo, "show", f"{commit}:{path}"]).splitlines()
    except: return []


def buggy_lines(old, new):
    return [i+1 for i, l in enumerate(difflib.ndiff(old, new)) if l.startswith('- ')]


examples, total = [], 0

for name, url in REPOS.items():
    repo = clone_repo(name, url)
    path = os.path.join(BASE_DIR, name)
    count = 0
    for h, msg in get_commits(path):
        if not is_good_commit(msg): continue
        for f in get_files(path, h):
            old, new = get_file(path, f"{h}^", f), get_file(path, h, f)
            if not old or not new: continue
            lines = buggy_lines(old, new)
            if not lines: continue
            examples.append({
                "repo": name,
                "instruction": "Find the buggy line(s) in this code",
                "input": "\n".join(old),
                "output": "Line(s): " + ", ".join(map(str, lines))
            })
            total += 1
            count += 1
            print(f"{total}. {name} – {h[:7]} – {f}")
            if total >= MAX_TOTAL or count >= MAX_PER_REPO: break
        if total >= MAX_TOTAL or count >= MAX_PER_REPO: break
    if total >= MAX_TOTAL: break


with open(OUTPUT_JSONL, "w") as f:
    for e in examples:
        f.write(json.dumps(e) + "\n")

print(f"\nDone: {len(examples)} prompts written to {OUTPUT_JSONL}")
