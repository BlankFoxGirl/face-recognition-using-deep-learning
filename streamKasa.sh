#!/bin/bash
MY_IP="10.2.1.143"
CAM_IP="10.2.1.173"
CAM_NAME="lab"

if [[ ! -z "$1" ]]; then
    CAM_IP=$1
fi

if [[ ! -z "$2" ]]; then
    MY_IP=$2
fi

if [[ ! -z "$3" ]]; then
    CAM_NAME=$3
fi

trap printout SIGINT
printout() {
    echo ""
    echo "Finished with count=$count"
    exit
}

# curl -k -s -u $USER_VALUE --ignore-content-length "https://$CAM_IP:19443/https/stream/mixed?video=h264&audio=g711&resolution=hd&deviceId=ASDASDASD" --output - | ffmpeg -hide_banner -i -nostats - -c copy -avoid_negative_ts 1 -r 25 -pre veryfast -f flv -muxdelay 1 -flvflags no_duration_filesize -listen 1 rtmp://$MY_IP:1935/live/$CAM_NAME
curl -k -s -u $USER_VALUE --ignore-content-length "https://$CAM_IP:19443/https/stream/mixed?video=h264&audio=g711&resolution=hd&deviceId=ASDASDASD" --output - | ffmpeg -hide_banner -re -i - -c copy -vsync 1 -avoid_negative_ts 1 -r 25 -ss 00:00:00 -f flv -flvflags no_duration_filesize -rtsp_transport tcp -preset ultrafast -muxdelay 1 -listen 1 rtmp://$MY_IP:1935/live/$CAM_NAME -nostats