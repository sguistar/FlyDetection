import csv
from collections import defaultdict

REID_CSV_PATH = r"D:\fly\coords\output_tracks_reid.csv"

ids = set()
frames_by_id = defaultdict(set)

with open(REID_CSV_PATH, newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    fi = header.index("frame")
    idi = header.index("id")
    xi = header.index("x")
    yi = header.index("y")

    for row in reader:
        frame = int(row[fi])
        tid = int(row[idi])
        ids.add(tid)
        frames_by_id[tid].add(frame)

print("Total track IDs:", len(ids))
for tid in sorted(ids):
    fs = sorted(frames_by_id[tid])
    print(
        f"ID {tid}: frames = {len(fs)}, "
        f"first = {fs[0]}, last = {fs[-1]}, "
        f"difference={fs[-1] - fs[0]}"
    )
