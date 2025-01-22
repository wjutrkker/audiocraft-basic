# audiocraft-basic
* A basic implementation of docker images for audiocraft and how to run the audio-craft code

```    
git clone git@github.com:wjutrkker/audiocraft-basic.git
cd audiocraft-basic
git clone https://github.com/wjutrkker/audiocraft-basic
docker build -t audiocraft:latest -f ./audiocraft-basic/Dockerfile .
docker run -it -v $PWD:/code audiocraft:latest bash
```

* If you want to jump right to samples to see the quality of the output. This link should work. 
https://ai.honu.io/papers/mbd/
