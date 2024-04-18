# UNNDevelopmentOfDeepLearningSystemsCoursePart2
## How to use
Folder `/Data/` contain two subfolders with images and their promts. Long-CLIP will be use `test_image.jpg` to inference. If you want use your own pictures, save it to `/Data/img/` as `test_image.jpg`, then add new prompt to `/Data/text/test_text.txt`

## How build and run with docker
1) Clone this repository
2) Download checkpoint [LongCLIP-B](https://huggingface.co/BeichenZhang/LongCLIP-B) and place it under `Long-CLIP/checkpoints/`
3) Build docker image
4) Mount folder /Data to /app_volume and run image

## Original repository
https://github.com/beichenzbc/long-clip
