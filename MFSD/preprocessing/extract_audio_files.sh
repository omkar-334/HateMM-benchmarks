#!/bin/bash
set -e

LOGFILE=test.log
(
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
pushd "${SCRIPT_DIR}/.." > /dev/null

videos_folder_path="./videos"
audios_folder_path="./audios"
ext="mp4"
mkdir -p "${audios_folder_path}"

shopt -s nullglob

# Process each video file
for video_file_path in "${videos_folder_path}"/*.${ext}; do
    # Extract file name and construct paths
    video_file_name=$(basename "${video_file_path}")
    video_file_name_without_extension="${video_file_name%.*}"
    audio_file_path="${audios_folder_path}/${video_file_name_without_extension}.mp3"

    echo "Processing: ${video_file_path}"
    if ffmpeg -i "${video_file_path}" -vn -codec:a libmp3lame -qscale:a 2 "${audio_file_path}"; then
        echo "Successfully converted: ${video_file_name}"
    else
        echo "Failed to convert: ${video_file_name}" >&2
        echo "Skipping..."
        continue
    fi
done

popd > /dev/null
) 2>&1 | tee $LOGFILE
