"""
  Verifies that a given challenge submision is properly constructed.

  Usage:

        python verify_submission.py challenge_set.json submission.csv
"""
import sys
import json

NTRACKS = 500


def verify_submission(challenge_path, submission_path):
    has_team_info = False
    error_count = 0

    try:
        f = open(challenge_path)
        js = f.read()
        f.close()
        challenge = json.loads(js)
    except FileNotFoundError:
        error_count += 1
        print("Can't read the challenge set")
        return error_count

    pids = set([playlist["pid"] for playlist in challenge["playlists"]])
    if len(challenge["playlists"]) != 10000:
        print("Bad challenge set")
        error_count += 1

    # seed_tracks contains seed tracks for each challenge playlist
    seed_tracks = {}
    for playlist in challenge["playlists"]:
        track_uris = [track["track_uri"] for track in playlist["tracks"]]
        seed_tracks[playlist["pid"]] = set(track_uris)

    found_pids = set()

    if error_count > 0:
        return error_count

    f = open(submission_path)
    for line_no, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        if line[0] == "#":
            continue

        if not has_team_info:
            if line.startswith("team_info"):
                has_team_info = True
                tinfo = line.split(",")
            else:
                print("missing team_info at line", line_no)
                error_count += 1

        else:
            fields = line.split(",")
            fields = [f.strip() for f in fields]
            try:
                pid = int(fields[0])
            except ValueError:
                print("bad pid (should be an integer)", fields[0], "at line", line_no)
                error_count += 1
                continue
            tracks = fields[1:]
            found_pids.add(pid)
            if not pid in pids:
                print("bad pid", pid, "at line", line_no)
                error_count += 1
            if len(tracks) != NTRACKS:
                print(
                    "wrong number of tracks, found",
                    len(tracks),
                    "should have",
                    NTRACKS,
                    "at",
                    line_no,
                )
                error_count += 1
            if len(set(tracks)) != NTRACKS:
                print(
                    "wrong number of unique tracks, found",
                    len(set(tracks)),
                    "should have",
                    NTRACKS,
                    "at",
                    line_no,
                )
                error_count += 1

            if seed_tracks[pid].intersection(set(tracks)):
                print(
                    "found seed tracks in the submission for playlist",
                    pid,
                    "at",
                    line_no,
                )
                error_count += 1

            for uri in tracks:
                if not is_track_uri(uri):
                    print("bad track uri", uri, "at", line_no)
                    error_count += 1

    if len(found_pids) != len(pids):
        print(
            "wrong number of playlists, found", len(found_pids), "expected", len(pids)
        )
        error_count += 1

    return error_count


def is_track_uri(uri):
    fields = uri.split(":")
    return (
        len(fields) == 3
        and fields[0] == "spotify"
        and fields[1] == "track"
        and len(fields[2]) == 22
    )


if __name__ == "__main__":
    errors = verify_submission('./challenge_data/challenge_set.json', './sub.csv')
    if errors == 0:
        print(
            "Submission is OK! Remember to gzip your submission before submitting it to the challenge."
        )
    else:
        print(
            "Your submission has",
            errors,
            "errors. If you submit it, it will be rejected.",
        )
