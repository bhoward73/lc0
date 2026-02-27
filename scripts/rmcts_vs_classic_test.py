#!/usr/bin/env python3
import subprocess
import time
import datetime

# Settings
total_hours = 8
play_minutes = 10
idle_minutes = 5
time_limits = [200, 300, 500]
logfile = "rmcts_vs_classic_test.log"
script = "scripts/bot_match.py"

# Engine args
engine_path = "build/onnxtrt/lc0"
shared_args = "--backend=onnx-trt --backend-opts=batch=128,steps=1 --weights weights/BT4-1024x15x32h-swa-6147500-policytune-332.pb.gz"
a_args = "rmcts"
b_args = "classic --minibatch-size=128 --max-prefetch=128"
label_a = "rmcts"
label_b = "classic"

# Prepare log file
with open(logfile, "a", encoding="utf-8") as f:
    f.write(f"Test started: {datetime.datetime.now()}\n")

end_time = time.time() + total_hours * 3600
cycle = 0
while time.time() < end_time:
    cycle += 1
    for movetime in time_limits:
        start = datetime.datetime.now()
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(f"Cycle {cycle}, movetime {movetime}ms, started {start}\n")
            f.flush()
        # Run bot_match.py for play_minutes
        proc = subprocess.Popen([
            "python3", script,
            "--engine", engine_path,
            "--shared-args", shared_args,
            "--a-args", a_args,
            "--b-args", b_args,
            "--movetime-ms", str(movetime),
            "--label-a", label_a,
            "--label-b", label_b,
            "--positions", "999999",
        ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        play_end = time.time() + play_minutes * 60
        while time.time() < play_end:
            line = proc.stdout.readline()
            if not line:
                break
            with open(logfile, "a", encoding="utf-8") as f:
                f.write(line.decode())
                f.flush()
        proc.terminate()
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(f"Cycle {cycle}, movetime {movetime}ms, finished {datetime.datetime.now()}\n")
            f.flush()
        # Idle for cooldown
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(f"Cycle {cycle}, cooldown started {datetime.datetime.now()}\n")
            f.flush()
        time.sleep(idle_minutes * 60)
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(f"Cycle {cycle}, cooldown finished {datetime.datetime.now()}\n")
            f.flush()

with open(logfile, "a", encoding="utf-8") as f:
    f.write(f"Test finished: {datetime.datetime.now()}\n")
