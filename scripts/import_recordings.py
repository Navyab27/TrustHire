"""Import a folder-labelled set of recordings into the custom test-set layout (docs/DATA_COLLECTION.md).

    python scripts/import_recordings.py "<path>/Training Data"

Folder -> attack label mapping below. Every video gets a single whole-video label segment. Re-encoded copies
(detected by matching duration within 0.1 s) are put in the same `participant_id` group as their original, so
grouped cross-validation never trains on one copy and tests on the other.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from trusthire.common.media import probe
from trusthire.evaluation.dataset import MANIFEST_COLUMNS, custom_root, validate

FOLDER_LABELS = {
    "RealVideos": ("genuine", "real participant, webcam, room A"),
    "Recompression": ("genuine", "re-encoded copy of a RealVideos clip"),
    "ScriptedVideo": ("scripted_reading", "participant reading prepared answers, room B"),
    "AudioMissMatch": ("lip_sync", "real face, swapped/misaligned audio (edited)"),
    "AIVideos": ("face_and_voice", "fully AI-generated talking head (text-to-video)"),
}


def main(src: Path) -> None:
    root = custom_root()
    (root / "recordings").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    durations: dict[str, float] = {}
    for folder, (attack, note) in FOLDER_LABELS.items():
        for p in sorted((src / folder).glob("*.mp4")):
            info = probe(p)
            sid = f"{folder}_{p.stem}".lower()
            dst = root / "recordings" / f"{sid}.mp4"
            if not dst.exists():
                shutil.copy2(p, dst)
            group = sid
            if folder == "Recompression":  # pair with the original real clip of identical duration
                match = [
                    k
                    for k, d in durations.items()
                    if k.startswith("realvideos") and abs(d - info.duration) < 0.1
                ]
                group = match[0] if match else sid
            if folder == "RealVideos":
                durations[sid] = info.duration
            rows.append(
                {
                    "session_id": sid,
                    "participant_id": group,
                    "recording": f"recordings/{sid}.mp4",
                    "attack": attack,
                    "id_photo": "",
                    "voice_sample": "",
                    "resume": "",
                    "lighting": "room B"
                    if folder == "ScriptedVideo"
                    else ("synthetic" if folder == "AIVideos" else "room A"),
                    "accent": "indian-english",
                    "webcam": f"{info.width}x{info.height}",
                    "bitrate_kbps": str(round(p.stat().st_size * 8 / 1000 / max(info.duration, 1e-3))),
                    "gender": "",
                    "skin_tone": "",
                    "glasses": "no",
                    "notes": note,
                }
            )
            (root / "labels" / f"{sid}.json").write_text(
                json.dumps(
                    {
                        "session_id": sid,
                        "segments": [
                            {
                                "start": 0.0,
                                "end": round(info.duration + 0.5, 2),
                                "label": attack,
                                "notes": note,
                            }
                        ],
                    },
                    indent=2,
                )
            )
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(root / "manifest.csv", index=False)
    problems = validate(root)
    print(f"{len(rows)} recordings -> {root}")
    print("\n".join(problems) if problems else "dataset OK")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
