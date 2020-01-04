---
layout: post
title:  "[번역] 딥러닝 분산 학습을 알아보자(Intro to Distributed Deep Learning Systems)"
categories: "papers"
tags:
  - "deep learning"
  - "distributed deep learning"
comments: true
---


> [Intro to Distributed Deep Learning Systems](https://medium.com/@Petuum/intro-to-distributed-deep-learning-systems-a2e45c6b8e7)를 번역한 문서입니다.

번역이 익숙치 않아 글이 어색할 수 있는 점 이해해주세요.

## 머신러닝 분산 학습이란?
일반적으로 머신러닝 분산 학습(distributed machine learning, DML)은 컴퓨터 사이언스 전반의 여러 학문이 포함된 분야입니다. 이론적 영역(통계, [학습이론](https://en.wikipedia.org/wiki/Computational_learning_theory), [최적화](https://en.wikipedia.org/wiki/Mathematical_optimization)), 알고리즘, [머신러닝](https://en.wikipedia.org/wiki/Machine_learning)([딥러닝](https://en.wikipedia.org/wiki/Deep_learning), [그래프 모델](https://en.wikipedia.org/wiki/Graphical_model), [커널](https://en.wikipedia.org/wiki/Kernel_method)), [분산 처리](https://en.wikipedia.org/wiki/Distributed_computing), [기억 장치](https://en.wikipedia.org/wiki/Computer_data_storage) 등이 포함됩니다. 이러한 각각의 분야들에서 수많은 주제들이 연구되고 있습니다. 또한 DML(머신러닝 분산 학습)은 빅데이터에 대한 처리 능력때문에 산업에서 폭 넓게 사용되고 있습니다.

## 머신러닝 분산 학습으로 해결하려는 문제가 무엇일까?
DML(머신러닝 분산 학습)을 이해하는 가장 쉬운 방법은 연구분야를 4개로 쪼개서 살펴보는 것입니다. 하지만 각각의 분야가 겹칠 수 있다는 점 미리 이해해 주세요.

### 1. 통계, 최적화, 알고리즘을 어떻게 사용할까?[^1]
대부분의 머신러닝 방법들은 training data에 대한 loss 함수를 최소화 하므로 다음 항목들을 고려해 학습을 진행합니다.

- 최적화 과정이 수렴하는데까지 얼마나 걸리는지, 즉 수렴 속도가 얼마인지
- 수렴된 솔루션이 얼마나 좋은지
- 좋은 솔루션을 위해 얼마나 많은 데이터가 필요한 지

이러한 분야들을 연구하기 위해 연구자들은 [최적화 이론](https://en.wikipedia.org/wiki/Mathematical_optimization)과 [통계적 학습 이론](https://en.wikipedia.org/wiki/Statistical_learning_theory) 같은 이론적 분석을 이용합니다. 하지만 많은 컴퓨팅 자원이 주어지고 병렬, 분산 학습등을 이용해 학습 속도를 올리는 것이 목표인 large-scale의 머신러닝 관점에서 보면 위와 비슷해 보이지만 다른 항목들을 고려하게 됩니다.

- 분산, 병렬 학습을 사용했을 때 우리의 모델이 원래 수렴했던 것과 똑같이 수렴하는 것이 보장되는 지
- 만약 아니라면 원래의 솔루션과 얼마나 다른지, 그리고 본질적인 최적해와 얼마나 다른지
- 좋은 수렴을 위해 다른 가정이나 조건들이 필요한 지
- 분산 학습을 사용하지 않았을 때와 비교해서 얼마나 빠른지, 그리고 어떻게 평가할 것인지
- 좋은 scailabilty와 좋은 수렴을 둘 다 만족하기 위해 어떻게 학습 과정을 디자인 할 것인지

### 2. 분산 학습에 더 적합한 머신러닝 모델 혹은 알고리즘을 어떻게 사용할까?[^2]
이 분야는 새로운 머신러닝 모델을 개발하거나 이미 존재하는 모델을 큰 데이터에 잘 학습하도록 조정하는 것들을 연구합니다.

### 3. 분산 머신러닝 모델을 어떻게 실생활에 적용할까?
[이미지 분류](https://en.wikipedia.org/wiki/Computer_vision#Recognition)와 같이 특정 모델이나 알고리즘의 scale-up을 요구하는 세부적인 적용 분야가 있습니다. 이러한 문제들에 솔루션들은 대부분 바로 제품에 쓰이기도 합니다.

### 4. 머신러닝의 규모를 키우기 위해서 어떻게 분산, 병렬 컴퓨터 시스템을 개발할까?
이 분야는 오히려 더 직관적입니다. 머신러닝 모델이나 알고리즘이 하나의 노드에서 계산 작업을 다 마치지 못했다면 더 많은 노드를 사용해서 분산 시스템을 개발하는 것을 시도해볼 수 있지요. 하지만 더 많은 자원을 사용하기 위해 고려해야 할 사항들이 많습니다.

- 일관성(Consistency) : 여러개의 노드들이 동시에 학습을 하고 있다면 어떻게 합쳐야 할까? 예를 하나의 문제를 푸는데 노드마다 갖고 있는 데이터 셋이 다르다면 어떻게 해야할까?
- [Fault tolerance](https://en.wikipedia.org/wiki/Fault_tolerance) : 클러스터가 1000개의 노드로 구성되어 있는데 만약 그 중 하나가 망가지면 어떻게 할까? 아예 처음부터 다시 시작하지 말고 고칠 수 있는 방법이 있을까?
- 통신(Communication) : 머신러닝은 많은 I/O(디스크 읽기, 쓰기), 데이터 처리 작업을 포함합니다. 다양한 환경에서(single node local disk, distributed file systems, CPU I/O, GPU I/O 등등) 빠른 I/O와 non-blocking 데이터 처리 과정을 가능하게 하는 저장 시스템을 설계할 수 있을까?
- 자원 관리(Resource management) : 컴퓨터 클러스터를 만드는 것은 엄청나게 비싸기 때문에 많은 유저들에게 공유됩니다. 자원 사용률을 최대화 해서 모두의 요구를 만족시키려면 어떻게 클러스터를 관리하고 자원을 할당해야 할까?
- 프로그래밍 모델(Programming model) : 평소에 프로그래밍 한 것과 같은 방법으로 분산 머신러닝 프로그래밍 해야할까? 코드량을 줄이고 효율성을 향상시키는 새로운 프로그래밍 모델을 만들 수 있을까? 하나의 노드에서 프로그래밍 한 것과 같이 프로그래밍 하면 자동으로 확장시킬 수 있을까?

대부분의 주류 머신러닝 소프트웨어들은 이러한 기술들에 집중하고 있습니다.

## 딥러닝 분산 학습에 대한 이해
딥러닝 분산 학습은 최근 다양한 분야에서 성과를 거두며 굉장히 중요해진 머신러닝 분산 학습의 작은 분야입니다. 딥러닝 분산 학습의 핵심이나 직면하고 있는 문제점들로 들어가기 전에 몇가지 알아야 할 용어들이 있습니다. 바로 데이터 병렬화(data parallelism)과 모델 병렬화(model parallelism) 입니다.[^3]

### 데이터 병렬화(Data parallelism)
데이터 병렬화([Data parallelism](https://en.wikipedia.org/wiki/Data_parallelism))는 데이터를 쪼개서 병렬성을 가능하게 하는 기술입니다. 데이터 병렬화를 사용하면 먼저 데이터를 worker machines(computational node)의 수 만큼 나눕니다. 그 다음 우리는 각 worker가 하나의 독립적인 조각을 갖게 하고 그 데이터에 대해 연산을 하도록 합니다. 우리는 병렬적으로 데이터를 읽는 여러개의 노드를 갖고 있기 때문에 하나의 노드를 사용할 때보다 더 많은 데이터를 읽을 수 있을 것입니다. 데이터 병렬화를 통해 처리량을 증가시킨 것입니다.

여러개의 노드를 사용해 수렴 속도를 높이고자 하는 분산 딥러닝에서, 데이터 병렬화는 직관적입니다. 우리는 각각의 워커가 자신의 데이터 조각에 대해 학습(경사 하강법)을 진행하도록 하고 그것에 대해 파라미터 업데이트(gradient)를 하도록 합니다. 모든 노드들이 네트워크를 통해 파라미터 상태들을 동기화시켜 모두 같은 값을 갖도록 합니다. 동기화를 하는데 시간이 지나치게 오래 걸리지 않는 한 하나의 노드를 사용할 때보다 향상된 결과를 볼 수 있을 것입니다. 이 방법이 구글의 초기 딥러닝 시스템인 [DisBelief](https://en.wikipedia.org/wiki/TensorFlow#DistBelief)가 본질적으로 작동하는 방식입니다.

### 모델 병렬화(Model parallelism)[^4]
데이터 병렬화와 비교했을때, 모델 병렬화는 더 복잡하고 추상적인 개념입니다. 일반적으로, 모델 병렬화에서는 데이터가 아닌 모델을 여러개의 워커에 나눕니다. 예를 들어 우리가 행렬 인수분해(Matrix factorization)를 하려고 할 때, 행렬의 크기가 너무 크고 우리는 이 거대한 행렬의 모든 파라미터를 알고 싶다고 가정해봅시다. 모델 병렬화를 진행하기 위해 우리는 행렬을 작은 단위(부분 행렬)로 나누고 각각의 워커에게 나눠줄 것입니다. 하나의 워커에 있는 [RAM](https://en.wikipedia.org/wiki/Random-access_memory)이 행렬의 파라미터를 담기에 충분하지 않다면 이 방법으로 여러 노드의 추가적인 램을 사용할 수 있게 됩니다. 다양한 노드들이 각각 행렬의 다른 부분들에 해당하는 일을 처리하기 때문에, 병렬적으로 계산할 때 속도 향상을 얻을 수 있게됩니다.

여기서 모델을 어떻게 나눠야 할까? 라는 질문이 떠오르게 됩니다. 너무나도 다양한 머신러닝 모델들이 있고 각 모델들마다 성격과 특징이 또 다르기 때문에, 모델 병렬화를 구현하는 원칙적 방법은 없습니다.

### 분산 딥러닝의 문제들
데이터 병렬화는 학습 데이터의 양이 많아질 때 더 빠르게 데이터를 읽을 수 있어서 상당히 효과적입니다. 모델 병렬화는 모델의 크기가 하나의 노드로 처리하기에 너무 클 때 여러 노드들의 메모리를 사용할 수 있게 해주므로 잘 들어 맞습니다. 

이상적으로 분산 딥러닝에서 우리가 할당한 머신의 수만큼의 속도향상을 얻길 원합니다.(보통 확장성이란 지표로 부릅니다.) 조금 더 구체적으로 K개의 머신을 할당했다면, 우리의 시스템이 하나의 머신에서 작동할 때보다 K배 더 빠르게 작동한다면 K의 확장성 혹은 선형적 확장성을 가지고 있다고 말합니다. 이러한 시스템에서 선형적 확장성은 이상적인 목표입니다.

하지만 동기화로 인한 오버헤드 때문에, 보통 하나의 노드에서 학습할 때보다 분산 컴퓨터 클러스터에서 학습할 때 더 오랜 시간이 걸립니다. 우리는 머신러닝 태스크의 수렴을 위해 계산이 끝날 때마다 여러 노드들에 대해 동기화를 하는데 추가로 시간을 사용해야 합니다. 실제로 동기화는 계산만큰 시간이 걸리거나 더 걸리기도 합니다.

왜일까요? 가장 큰 이유는 다른 노드들보다 느리게 작동하는 특정 노드들이 있기 때문입니다. 그들을 동기화시키기 위해 빠른 노드가 느린 노드들이 일을 끝낼 때까지 기다려야 합니다. 시스템 성능은 항상 더 느린 노드들에 의해 결정됩니다. 이러한 경우에 K개의 머신을 같이 놓는 것은 음의 확장성을 보일 것이고 이것은 돈과 시간 낭비입니다.

-------
## 각주

[^1]: 더 공부해보면 좋을 논문입니다.<br/>[More Effective Distributed ML via a Stale Synchronous Parallel Parameter Server](http://www.cs.cmu.edu/~seunghak/SSPTable_NIPS2013.pdf)
[^2]: 아래 논문들을 추천해드립니다.<br/>[Asymptotically Exact, Embarrassingly Parallel MCMC, by Willie Neiswanger](https://arxiv.org/abs/1311.4780), Chong Wang, Eric P. Xing. UAI 2014.<br/>[LightLDA: Big Topic Models on Modest Compute Clusters](https://arxiv.org/abs/1412.1576), by Jinhui Yuan, Fei Gao, Qirong Ho, Wei Dai, Jinliang Wei, Xun Zheng, Eric P. Xing, Tie-yan Liu, Wei-Ying Ma. WWW 2015.<br/>[SaberLDA: Sparsity-Aware Learning of Topic Models on GPUs](https://arxiv.org/abs/1610.02496), by Kaiwei Li, Jianfei Chen, Wenguang Chen, Jun Zhu. ASPLOS 2017.
[^3]: 두개의 주목할만한 큰 규모의 분산 딥러닝 논문이 NIPS 2012와 2013 ICML에서 출판되었습니다.<br/>첫번째 논문은 구글의 내부 딥러닝 프레임워크의 첫번째 세대에 관해서 설명하고 있습니다. 두번째 논문은 실리콘밸리의 바이두를 이끌고 있는 Adam Coates가 쓴 논문입니다. 이 두 논문의 핵심 아이디어는 더 많은 연산 노드를 사용해 딥러닝 연산의 규모를 키우는 것입니다. 첫번째 논문은 데이터 병렬화 두번째 논문은 모델 병렬화를 사용했습니다.<br/>[Large Scale Distributed Deep Networks](https://static.googleusercontent.com/media/research.google.com/en//archive/large_deep_networks_nips2012.pdf), by Jeffrey Dean, Greg Corrado, Rajat Monga, Kai Chen, Matthieu Devin, Mark Mao, Marc’aurelio Ranzato, Andrew Senior, Paul Tucker, Ke Yang, Quoc V. Le, Andrew Y. Ng.<br/>[Deep Learning with COTS HPC](http://proceedings.mlr.press/v28/coates13.pdf), Adam Coates, Brody Huval, Tao Wang, David Wu, Bryan Catanzaro, Andrew Ng. ICML 2013.
[^4]: 모델 병렬화에 특히 관심 있으신 분들을 다음 두 논문들을 보시면 좋으실 겁니다.<br/>[STRADS: A Distributed Framework for Scheduled Model Parallel Machine Learning](http://www.cs.cmu.edu/~epxing/papers/2016/Kim_etal_EuroSys16.pdf), by Jin Kyu Kim, Qirong Ho, Seunghak Lee, Xun Zheng, Wei Dai, Garth A. Gibson, Eric P. Xing. EuroSys 2016.<br/>[Device Placement Optimization with Reinforcement Learning](https://arxiv.org/abs/1706.04972), by Azalia Mirhoseini Lt ; / RTI