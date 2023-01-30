# Speaker-Verification-task3

## How to launch
```
cd Speaker-Verification/task3
docker build . -t speaker_verification:0.0
docker run -d --cpuset-cpus 128-255 --gpus '"device=4,5,6,7"' -v /raid/madina_abdrakhmanova/datasets/sf_pv/data_v2:/workdir/data_v2 --name sv_task3 speaker_verification:0.0
docker exec -it sv_task3 /bin/bash
```
