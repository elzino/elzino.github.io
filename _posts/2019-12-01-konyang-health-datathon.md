---
layout: post
title:  "2019 건양 헬스 데이터톤 참가 후기"
categories: "projects"
tags:
  - "hackathon"
  - "deep learning"
  - "medical image classification"
comments: true
---

지난 11월 29일 부터 11월 30일까지 무박 2일로 진행된 [건양 헬스 데이터톤](https://github.com/khd2019/khd2019)(Konyang Health Datathon 2019)에 참가했습니다! 이번에도 친구인 승일이와 '행복코딩'팀으로 참여했는데요. 이번 대회는 안저 이미지를 정상 안저, 황반 변성, 당뇨성 망막 병증, 망막 정맥 폐쇄 이미지로 분류하는 A조와 유방 촬영 이미지를 양성과 악성으로 분류하는 B조로 나뉘어서 진행되었습니다. 저희는 A조에 참여해 당당히 1등을 하였습니다!! :clap::clap::clap: 대회를 어떻게 준비했는지, 또 어떤걸 느꼈는지 간단히 기록해보려고 합니다.

![리더보드](https://www.dropbox.com/s/jvg9rfs336liz4z/leaderboard.png?raw=1)

## 사전 조사
안저 이미지 분류는 원래 잘 알지 못하던 분야였습니다. 그런데 조사를 해보니 4년전에 [Kaggle에서 주관한 대회](https://www.kaggle.com/c/diabetic-retinopathy-detection)도 열렸었고 구글도 [당뇨성 망막 병증을 분류하는 논문](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/45732.pdf)을 냈었습니다. 그래서 대회전에 미리 Kaggle 대회에서 상위팀들의 솔루션 및 관련 논문들을 읽어봤습니다. 이 중에서 Kaggle 1등팀인 Ben Graham의 [Gaussian Blur를 이용해 local average를 빼주는 방법](http://blog.kaggle.com/2015/09/09/diabetic-retinopathy-winners-interview-1st-place-ben-graham/)이 인상 깊었습니다. 

## 대회
대회를 가서 당황한 부분은 다시 nsml이였습니다. 대회전에 Pytorch를 이용해 베이스 코드를 미리 짜서 갔었는데 nsml에서 도커 이미지 생성에 삼십분 가량이 걸렸습니다. 반면 기본으로 제공하는 Keras를 이용한 코드를 돌려보니 도커 생성이 약 10초만에 되서 대회장에서 Keras를 쓰기로 방향을 틀었습니다.  
대회를 참여하면서 성능을 올리기 위해 여러가지를 시도했습니다. 먼저 저번 NAVER 대회에서 배운 교훈대로 train set과 valid set을 나누는 데 신경을 썼습니다. Category 별로 비율을 맞춰서 train set과 valid set을 나누었습니다. 모델은 VGG-network를 변형해서 사용했고 Keras에서 제공하는 ImageDataGenerator를 이용해 rotation, zoom, width shift, height shift, flip을 시켜서 Augmentation 했습니다. Optimizer는 RAdam을 사용했고 Label smooth도 적용했습니다. 또한 학습 후반부에는 Augmentation을 거의 주지 않고 learning rate와 label smooth 비율을 낮춰서 fine tuning 한 점이 성능 향상에 큰 도움을 주었습니다.

## 느낀점
큰 기대를 하지 않고 참여했던 대회인데 좋은 결과가 있어서 너무 기분이 좋았습니다. Keras도 거의 사용해보지 않았었는데 이번 기회에 다루면서 짧은 시간에 모델을 짤때는 확실히 강점이 있음을 느꼈습니다. 다만 Image classification의 경험이 거의 없어서 여러가지를 시도할 때 과연 이게 성능에 도움이 될까? 라는 의구심이 계속 들었습니다. 이론적 기반 없이 동전 던지기를 잘해 성능이 잘 나온 거 같아 좀 더 경험을 쌓아야 할 것 같습니다. 외부 대회에서 처음으로 1등을 해봐서 잊지 못할 추억이 될 것 같습니다. :+1:
