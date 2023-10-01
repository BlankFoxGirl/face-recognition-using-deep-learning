FROM ubuntu:22.04

# RUN add-apt-repository ppa:savoury1/ffmpeg4
RUN apt-get update
RUN apt-get install -y curl ffmpeg
RUN mkdir /app
WORKDIR /app
CMD ["bash", "/app/streamKasa.sh"]