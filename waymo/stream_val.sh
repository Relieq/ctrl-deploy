#!/usr/bin/env bash
# Stream Waymo validation tfrecords: download -> convert (convert_segment.py) -> delete, with P parallel workers.
# Only converted outputs stay on disk (~0.75 GB per segment), never more than P tfrecords at once.
#
#   NUM_SEGMENTS=20 bash stream_val.sh      # first 20 validation segments (by CTRL segment index)
#   bash stream_val.sh                      # all 202
# Needs: gcloud auth login with a Google account registered at https://waymo.com/open
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../scripts/env.sh"
BUCKET="${WAYMO_BUCKET:-gs://waymo_open_dataset_v_1_4_3/individual_files/validation}"
NUM_SEGMENTS="${NUM_SEGMENTS:-0}"
P="${P:-3}"
SST="$CTRL_ROOT/SST"
export SST_DATA="${SST_DATA:-$SST/data}"
export TRK_DATA="${TRK_DATA:-$CTRL_ROOT/ImmortalTracker-for-CTRL/data}"
export TMP_DIR="${TMP_DIR:-$CTRL_ROOT/tmp_tfrecord}"
export DONE_DIR="$SST_DATA/waymo/waymo_format/gt_parts"
export SST HERE TF_CPP_MIN_LOG_LEVEL=3
mkdir -p "$TMP_DIR" "$DONE_DIR"
rm -f "$TMP_DIR"/*  # leftovers of an interrupted run (partial .gstmp downloads)

LIST="$CTRL_ROOT/val_tfrecords.txt"
gsutil ls "$BUCKET/*.tfrecord" > "$LIST"
# order by CTRL validation segment index (idx2contextname.pkl), optionally keep the first N
python - "$LIST" "$SST/tools/idx2contextname.pkl" "$NUM_SEGMENTS" <<'PY' > "$LIST.todo"
import pickle, sys
urls = [l.strip() for l in open(sys.argv[1]) if l.strip()]
ctx2seg = {c: int(i[1:4]) for i, c in pickle.load(open(sys.argv[2], 'rb')).items() if i[0] == '1'}
key = lambda u: ctx2seg.get(u.split('segment-')[-1].replace('_with_camera_labels.tfrecord', ''), 10**6)
urls = sorted(urls, key=key)
n = int(sys.argv[3])
print('\n'.join(urls[:n] if n > 0 else urls))
PY
echo "bucket has $(wc -l < "$LIST") tfrecords; processing $(wc -l < "$LIST.todo") with $P workers"

one() {
  url="$1"; name="$(basename "$url")"
  ctx="${name#segment-}"; ctx="${ctx%_with_camera_labels.tfrecord}"
  [ -s "$DONE_DIR/$ctx.bin" ] && { echo "skip $ctx"; return 0; }
  for try in 1 2 3; do
    gsutil -q cp "$url" "$TMP_DIR/$name" && break
    rm -f "$TMP_DIR/$name" "$TMP_DIR/$name"_.gstmp
    echo "download retry $try for $ctx"; sleep 10
  done
  [ -s "$TMP_DIR/$name" ] || { echo "DOWNLOAD FAILED $ctx"; return 0; }
  python "$HERE/convert_segment.py" "$TMP_DIR/$name" --sst "$SST" --sst-data "$SST_DATA" --trk-data "$TRK_DATA" \
    || echo "FAILED $ctx"
  rm -f "$TMP_DIR/$name"
}
export -f one
xargs -P "$P" -I{} bash -c 'one "$@"' _ {} < "$LIST.todo"
echo "converted segments: $(ls "$DONE_DIR" | wc -l)"
