docker run --runtime=nvidia -it -v /mnt:/mnt \
         -p 8080:8080 deepjavalibrary/djl-serving:deepspeed-nightly \
         djl-serving -m /mnt/project/djl-serving/engines/python/src/test/resources/gpt2
