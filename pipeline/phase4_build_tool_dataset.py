"""
build_tool_dataset.py
=====================
Phase 4: ReAct / Tool-Use Dataset Builder

Generates a structured dataset of agentic tool-use sequences using the
<TOOL: BASH> token and latent pause signatures (===). Each row teaches
the model to:
  1. Receive a task
  2. Emit dark loops (latent reasoning)
  3. Emit a <TOOL: BASH> call with a command
  4. Emit <RESULT> block (simulated output)
  5. Emit more latent loops (reflection)
  6. Produce a final natural language answer

Format:
  [AGENT] {task}
  ={N times}
  <TOOL: BASH>
  {command}
  </TOOL>
  <RESULT>
  {stdout}
  </RESULT>
  ={M times}
  {final_answer}
"""

import json
import os
import random

OUTPUT_FILE = "data/tool_use_dataset.jsonl"
os.makedirs("data", exist_ok=True)

random.seed(42)

# ── Template library ──────────────────────────────────────────────
TOOL_EXAMPLES = [
    # File system tasks
    {
        "task": "List all Python files in the current directory.",
        "loops_pre": 3, "loops_post": 2,
        "command": "find . -name '*.py' -maxdepth 2",
        "result": "./train_qlora_7b.py\n./eval_phase2.py\n./build_7b_dataset.py\n./monitor_ui.py",
        "answer": "There are 4 Python files in the current directory: train_qlora_7b.py, eval_phase2.py, build_7b_dataset.py, and monitor_ui.py."
    },
    {
        "task": "Check how much disk space is available.",
        "loops_pre": 2, "loops_post": 2,
        "command": "df -h / | tail -1",
        "result": "/dev/sdb2       457G  329G  105G  76% /",
        "answer": "You have 105GB of free disk space available (76% used out of 457GB total)."
    },
    {
        "task": "Show the last 5 lines of the training log.",
        "loops_pre": 3, "loops_post": 3,
        "command": "tail -5 training_7b.log",
        "result": "{'loss': '10.57', 'grad_norm': '0.6758', 'learning_rate': '2.187e-09'}\n[SYSTEM] Phase 2 Complete.\nWriting model shards: 100%|██| 1/1\n[SYSTEM] Model saved to checkpoints/mamba-2.8b-phase2",
        "answer": "The training log shows Phase 2 completed successfully with a final loss of 10.57. The model has been saved."
    },
    {
        "task": "Count how many lines are in the dataset file.",
        "loops_pre": 3, "loops_post": 2,
        "command": "wc -l data/universal_7b_latent.jsonl",
        "result": "10164 data/universal_7b_latent.jsonl",
        "answer": "The dataset file contains 10,164 lines, one per training example."
    },
    # System monitoring
    {
        "task": "Check the current GPU memory usage.",
        "loops_pre": 4, "loops_post": 3,
        "command": "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader",
        "result": "388 MiB, 12288 MiB",
        "answer": "The GPU is using 388 MiB out of 12,288 MiB total VRAM. Nearly all memory is free."
    },
    {
        "task": "What Python version is installed?",
        "loops_pre": 2, "loops_post": 2,
        "command": "python --version",
        "result": "Python 3.14.3",
        "answer": "Python version 3.14.3 is installed on this system."
    },
    {
        "task": "Check if CUDA is available for PyTorch.",
        "loops_pre": 4, "loops_post": 3,
        "command": "python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\"",
        "result": "True NVIDIA GeForce RTX 3060",
        "answer": "CUDA is available. The system has an NVIDIA GeForce RTX 3060 GPU detected by PyTorch."
    },
    # File operations
    {
        "task": "Create a directory called 'results' and verify it was created.",
        "loops_pre": 3, "loops_post": 2,
        "command": "mkdir -p results && ls -d results/",
        "result": "results/",
        "answer": "The 'results' directory has been created successfully."
    },
    {
        "task": "Search for any file containing the word 'checkpoint' in its name.",
        "loops_pre": 3, "loops_post": 3,
        "command": "find . -name '*checkpoint*' -type d 2>/dev/null",
        "result": "./checkpoints/mamba-2.8b-phase2\n./checkpoints",
        "answer": "Found two checkpoint directories: the main checkpoints folder and the mamba-2.8b-phase2 checkpoint inside it."
    },
    {
        "task": "Show the size of the model checkpoint directory.",
        "loops_pre": 4, "loops_post": 2,
        "command": "du -sh checkpoints/mamba-2.8b-phase2/",
        "result": "5.6G\tcheckpoints/mamba-2.8b-phase2/",
        "answer": "The Phase 2 model checkpoint is 5.6 gigabytes on disk."
    },
    # Process monitoring
    {
        "task": "Check if there are any Python processes currently running.",
        "loops_pre": 3, "loops_post": 2,
        "command": "ps aux | grep python | grep -v grep | wc -l",
        "result": "2",
        "answer": "There are 2 Python processes currently running on the system."
    },
    {
        "task": "Show the top 3 processes using the most CPU.",
        "loops_pre": 4, "loops_post": 3,
        "command": "ps aux --sort=-%cpu | head -4 | tail -3",
        "result": "phil    1234  45.2  2.1  python train_qlora_7b.py\nphil    5678   8.1  0.5  python monitor_ui.py\nroot    7003   0.6  0.2  /usr/bin/python ./kolibri",
        "answer": "The top CPU consuming processes are: the QLoRA training script (45.2%), the monitor UI (8.1%), and the Kolibri server (0.6%)."
    },
    # Math via bash
    {
        "task": "Calculate 2 to the power of 32 using bash.",
        "loops_pre": 5, "loops_post": 3,
        "command": "python -c \"print(2**32)\"",
        "result": "4294967296",
        "answer": "2 to the power of 32 equals 4,294,967,296."
    },
    {
        "task": "Convert 1 gigabyte to megabytes and bytes.",
        "loops_pre": 5, "loops_post": 3,
        "command": "python -c \"gb=1; print(f'{gb}GB = {gb*1024}MB = {gb*1024**3} bytes')\"",
        "result": "1GB = 1024MB = 1073741824 bytes",
        "answer": "1 gigabyte equals 1,024 megabytes or 1,073,741,824 bytes."
    },
    # Multi-step reasoning
    {
        "task": "Find the largest file in the checkpoints directory.",
        "loops_pre": 5, "loops_post": 4,
        "command": "find checkpoints/ -type f -exec ls -s {} \\; | sort -rn | head -1",
        "result": "5734400 checkpoints/mamba-2.8b-phase2/model.safetensors",
        "answer": "The largest file is model.safetensors at approximately 5.7 GB in the mamba-2.8b-phase2 checkpoint directory."
    },
    {
        "task": "Check the current date and time.",
        "loops_pre": 2, "loops_post": 2,
        "command": "date '+%Y-%m-%d %H:%M:%S %Z'",
        "result": "2026-04-01 00:23:55 CDT",
        "answer": "The current date and time is April 1st, 2026 at 12:23:55 AM CDT."
    },
    # Package checks
    {
        "task": "Check which version of PyTorch is installed.",
        "loops_pre": 3, "loops_post": 2,
        "command": "python -c \"import torch; print(torch.__version__)\"",
        "result": "2.9.0+cu126",
        "answer": "PyTorch version 2.9.0 with CUDA 12.6 support is installed."
    },
    {
        "task": "List all installed pip packages related to transformers.",
        "loops_pre": 4, "loops_post": 3,
        "command": "pip list | grep -i 'transform\\|peft\\|trl\\|accelerate'",
        "result": "accelerate              1.6.0\npeft                    0.15.1\ntransformers            4.51.3\ntrl                     0.17.0",
        "answer": "The HuggingFace stack is installed: transformers 4.51.3, peft 0.15.1, trl 0.17.0, and accelerate 1.6.0."
    },
]

# ── Build dataset rows ─────────────────────────────────────────────
rows = []
# Use each example multiple times with slight loop variation
for base in TOOL_EXAMPLES:
    for variation in range(4):  # 4 loop count variations per base
        lp = max(1, base["loops_pre"]  + random.randint(-1, 2))
        la = max(1, base["loops_post"] + random.randint(-1, 2))

        pre_loops  = "=" * lp
        post_loops = "=" * la

        text = (
            f"[AGENT] {base['task']}\n"
            f"{pre_loops}\n"
            f"<TOOL: BASH>\n"
            f"{base['command']}\n"
            f"</TOOL>\n"
            f"<RESULT>\n"
            f"{base['result']}\n"
            f"</RESULT>\n"
            f"{post_loops}\n"
            f"{base['answer']}"
        )
        rows.append({
            "instruction": f"[AGENT] {base['task']}",
            "dark_loops_pre": pre_loops,
            "tool": base["command"],
            "result": base["result"],
            "dark_loops_post": post_loops,
            "response": base["answer"],
            "full_text": text,
            "domain": "tool"
        })

random.shuffle(rows)

with open(OUTPUT_FILE, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print(f"[BUILD] Tool-use dataset written: {len(rows)} rows → {OUTPUT_FILE}")
print(f"  Domains: {len(TOOL_EXAMPLES)} base examples × 4 loop variations")
print(f"  Format: [AGENT] → loops → <TOOL: BASH> → <RESULT> → loops → answer")
