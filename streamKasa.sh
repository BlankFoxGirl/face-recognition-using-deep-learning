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

curl -k -s -u $USER_VALUE --ignore-content-length "https://$CAM_IP:19443/https/stream/mixed?video=h264&audio=g711&resolution=hd&deviceId=ASDASDASD" --output - | ffmpeg -stream_loop -1 -hide_banner -i - -c copy -tune zerolatency -f flv -flvflags no_duration_filesize -listen 1 rtmp://$MY_IP:1935/live/$CAM_NAME