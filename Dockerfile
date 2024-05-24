FROM intel/intel-optimized-ffmpeg
# ubuntu 22.04 ffmpeg container

RUN apt-get update && apt-get install ffmpeg python3 python3-pip git -y
RUN pip3 install 'torch==2.1.0'
RUN pip3 install -U audiocraft  # stable release
RUN pip3 install -U git+https://git@github.com/facebookresearch/audiocraft#egg=audiocraft 

RUN mkdir -p /code
COPY . /code
# sudo docker build -t audiocraft .
# sudo docker run -it -v $PWD:/code audiocraft:latest 