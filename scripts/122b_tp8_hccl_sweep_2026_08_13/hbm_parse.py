import subprocess, re
out = subprocess.check_output(["npu-smi", "info"], stderr=subprocess.STDOUT).decode()
rows = []
lines = out.splitlines()
for i, line in enumerate(lines):
    m = re.match(r"\|\s+(\d+)\s+(\d+)\s+\|\s+0000:", line)
    if m:
        chip_offset = int(m.group(1))
        phy_id = int(m.group(2))
        m2 = re.search(r"(\d+)/\s*(\d+)\s+\|\s*$", line)
        if m2:
            hbm_used = int(m2.group(1))
            hbm_total = int(m2.group(2))
            rows.append({"chip": phy_id, "hbm_used_mb": hbm_used, "hbm_total_mb": hbm_total})
for r in rows:
    print("chip {:2d}: hbm={:>6}/{:>6}MB".format(r["chip"], r["hbm_used_mb"], r["hbm_total_mb"]))
print("Total HBM used: {}MB, per-chip avg: {:.0f}MB".format(
    sum(r["hbm_used_mb"] for r in rows), sum(r["hbm_used_mb"] for r in rows)/max(len(rows),1)))
