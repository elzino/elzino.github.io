---
layout: post
title:  "Scaling Distributed Machine Learning with the Parameter Server 리뷰"
categories: "papers"
tags:
  - "deep learning"
  - "distributed deep learning"
  - "parameter server"
comments: true
use_math: true
---

논문 링크 : [pdf](https://www.cs.cmu.edu/~muli/file/parameter_server_osdi14.pdf)

최근 딥러닝 분산 학습을 공부하면서 [parameter server framework](https://www.quora.com/What-is-the-Parameter-Server)에 대해 알게 되었습니다. Uber의 [ring-Allreduce](https://towardsdatascience.com/visual-intuition-on-ring-allreduce-for-distributed-deep-learning-d1f34b4911da)와는 다른 방법으로 분산 학습을 지원하고 있었습니다. Parameter Server를 다룬 논문들을 찾아보다가 가장 유명한 이 논문을 읽고 리뷰하게 되었습니다. 14년도에 나온 논문이라 GPU를 본격적으로 사용하기 전인 점을 고려하면서 보시면 조금 더 이해가 잘 될 것 같습니다.

## Abstract
- 이 논문에서는 서버에서 글로벌하게 파라미터들을 관리하는 파라미터 서버 프레임워크를 제안합니다.
- 이 프레임워크는 asynchronous data communication, flexible consistency models, elastic scalability, continuous fault tolerance를 지원합니다.

## Introduction
큰 규모의 머신러닝 학습이 중요해지면서 분산 학습은 점점 필수가 되어가고 있습니다. 하지만 많은 연산량과 큰 규모의 데이터 통신 때문에 세심한 시스템 디자인이 요구됩니다. 그 중 한 예로 워커 노드들간에 파라미터를 공유하고 계산할 때마다 파라미터를 업데이트를 하는 파라미터 서버 방법이 있습니다. 하지만 파라미터를 공유하기 위해서는 막대한 양의 대역폭(network bandwidth), 동기화(synchronization), 10% 정도의 failure rate를 가지고 있는 일반 클라우드에서도 fault tolerance를 갖춰야하기 때문에 어려운 점이 많습니다. 이 논문에서는 이러한 한계들을 극복한 새로운 파라미터 서버 프레임워크를 제안합니다.

### Main Ideas
- Efficient communication: asynchronous communication을 통해 워커노드들이 기다리지 않고 연산 할 수 있게 해줍니다.
- Flexible consistency models: Consistency를 알고리즘, 데이터에 따라서 조절할 수 있게 해주었습니다. 이를 통해 synchronization cost, latency를 줄이고 convergence와 trade-off 할 수 있게 되었습니다.
- Elastic Scalability: 프레임워크를 다시 시작하지 않고 노드를 새롭게 추가하거나 제거할 수 있습니다.
- Fault Tolerance and Durability: 머신에 문제가 생기더라도 1초안에 복구할 수 있습니다. 이를 위해 Vector clock 방법을 이용합니다.
- Ease of Use: 공유된 파라미터들이 벡터나 행렬로 표현되어 머신러닝에 적용하기 쉽게 되어있습니다.

## Architecture
![High level architecture](https://www.dropbox.com/s/8z0wf0gb3858joe/high%20level%20architecture.png?raw=1)
파라미터 서버 노드들은 위 그림과 같이 서버 그룹과 워커 그룹으로 묶을 수 있습니다. 서버 노드들은 공유된 파라미터들을 저장하고 있습니다. 서버 노드들끼리는 파라미터 복제나 이동을 위해 서로 통신합니다. 서버 매니저는 노드의 작동 상태나 파라미터의 할당 등을 관리합니다. 워커 그룹은 각각 application을 돌립니다.  
워커들은 일반적으로 할당 받은 학습 데이터를 저장하고 미분값과 같은 local 통계값을 통신합니다. 워커들은 오직 서버 노드와만 통신하고 워커끼리 통신하지는 않습니다. 각 워커 그룹에는 스케쥴러가 있는데 일을 할당하고 진척상황을 관리합니다. 워커가 추가 되거나 없어지면 스케쥴러가 끝나지 않은 일을 다시 할당합니다.  
파라미터 서버는 namespace를 지원하는데 각 워커 그룹마다 같은 namespace를 사용할 수도 다른 namespace를 사용할 수도 있습니다. 이를 사용하면 여러 개의 워커 그룹이 같은 namespace를 사용해 동시에 학습할 수도 있습니다. 또한 namespace를 달리해 한 워커 그룹은 학습을 담당하고 다른 워커 그룹은 online service에 사용할 수도 있습니다.  

이제 아키텍쳐를 이루는 구성요소들이 뭐가 있는지 하나씩 살펴 보겠습니다.  

### Key-Value Vectors
기본적으로 모델을 이루는 파라미터들을 key, value를 이용해 표현할 수 있습니다. feature ID가 key가 되고 weight가 value가 될 것입니다. 그런데 이 프레임워크는 머신러닝에서 linear algebra object들을 주로 다룬다는 점에 주목합니다. 따라서 key, value를 이용해 파라미터들을 저장하는 동시에 key가 없는 부분들을 0으로 취급해 벡터와 행렬로 표현할 수 있게 합니다. 이를 통해 여러 선형 대수 연산들을 최적화 할 수 있게 합니다.

### Range Push and Pull
노드 간에 데이터를 주고 받을 때 ```push```나 ```pull``` 연산을 사용합니다. 이때 range를 사용하면 효율적으로 network bandwidth를 사용할 수 있습니다. Range 안에 있는 key들에 해당하는 value들을 보내고 받을 수 있습니다. gradient도 파라미터와 같은 key를 사용하므로 ```w.push(R, g, destination)``` 과 같이 range를 사용해서 주고 받을 수 있습니다.

### User-Defined Functions on the Server
워커에서 데이터를 처리하는 것 뿐 아니라, 서버에서도 유저가 정의한 함수를 실행할 수 있습니다. regularizer나 proximal operator등을 계산할 때 유용하게 쓰일 수 있습니다.

### Asynchronous Tasks and Dependency
![async](https://www.dropbox.com/s/zff86wsrds1j5if/async.png?raw=1)
push나 pull, 노드에서 실행되는 user-defined 함수 등을 통틀어 task라 합니다. Task는 기본적으로 asynchronous 하게 작동됩니다. caller는 callee로 부터 끝났다는 신호를 받으면 task가 끝난 것으로 표시합니다. callee는 task의 return 값을 받고 subtask들이 모두 끝나면 끝났다고 표시합니다. Task는 병렬적으로 실행되지만 dependency를 줄 수 있습니다. 위 그림에서 iter 10과 iter 11은 독립적으로 실행되었지만 iter 12는 iter 11이 끝난 후에 실행하게 했습니다. dependency는 알고리즘 및 다음 장에서 설명할 flexible consistency 구현에 사용됩니다.

### Flexible Consistency
![flexible consistency](https://www.dropbox.com/s/jytneivur3vu9e4/flexible%20consistency.png?raw=1)
독립적인 task들은 병렬적으로 CPU, disk, network bandwidth를 사용해 시스템 효율성을 높일 수 있습니다. 하지만 이것은 노드 간에 data inconsistency를 야기해 수렴 속도를 늦출 수 있습니다. inconsistency에 강한 몇몇 알고리즘이 있으므로 여러가지 요소들을 고려해서 consistency model을 정의하는게 최상의 방법일 것입니다.
 - Sequential: 모든 task들이 순차적으로 실행됩니다. single-thread 구현과 동일합니다.
 - Eventual: 모든 task들이 동시에 시작될 수 있습니다. 알고리즘이 delay에 강할 때만 사용하는 것을 추천합니다.
 - Bounded Delay: maximal delay time \\( \tau \\)를 정해 놓고 새로운 task가 \\( \tau \\)전에 실행된 모든 task들이 실행될 때까지 막습니다. \\( \tau = 0 \\)이면 sequential과 같고 \\( \tau=\inf \\)면 Eventual과 같습니다.

### User-defined Filters
파라미터 서버는 key, value 값을 선택적으로 보낼 수 있게 합니다. 예를 들어 기존 파라미터와 비교해서 특정값 이상의 변화가 있을 때만 보내는 significantly modified filter를 사용할 수 있습니다. 이 논문은 뒤의 실험에서 서버의 weight에 영향을 줄 거 같은 gradient만 보내는 KKT filter를 사용합니다.

## Implementation details
구현 과정에서 사용된 기술들이 어떤 것들이 있는지 살펴보겠습니다.
### Vector Clock
복잡한 dependency graph와 machine failure로부터 빠른 복구를 위해 각 key-value값 마다 vector clock을 사용해 시간을 기록합니다. 모든 key-value 값 마다 time을 저장하려면 큰 용량이 필요하지만 range를 사용한다는 점을 이용해 최적화 할 수 있습니다. unique한 range마다 vector clock을 하나씩 주는 방법을 사용합니다. 파라미터 서버가 처음 시작했을 때는 서버 노드마다 하나의 vector clock만 있을 것입니다. range가 쪼개질 때마다 최대 3개의 vector clock이 생기지만 range의 개수가 파라미터의 개수보다 훨씬 적으므로 효율적으로 vector clock을 저장할 수 있게 됩니다.

### Messages
메세지는 다음과 같이 range, key-value 값들, 그리고 vector clock으로 구성됩니다.
\\[ [vc(\mathcal{R}), (k1, v1), ..., (k_p, v_p)]\ k_j\ \in\ \mathcal{R}\ and\ j\ \in\ \\{ 1, ...,p \\} \\]
Range에 있지만 message에 포함되지 않은 key-value도 같은 timestamp를 갖게 업데이트 됩니다. 머신러닝에서 통신량은 굉장히 크므로 메세지를 압축하는 것이 필요합니다. 매번 같은 key list를 보내는 경우가 많을 것이기 때문에 key list를 캐싱해서 사용합니다. 또한 value 값들도 0이 아닌 값들만 보내도록 압축하는 fast Sanppy compression을 사용합니다.

### Consistent Hashing
![consistent hashing](https://www.dropbox.com/s/safvxfqar4yb41k/consistent%20hashing.png?raw=1)
파라미터 서버는 기존의 해쉬 테이블 방식과 비슷하게 key를 나눕니다. Hash ring위에 key와 서버 노드 id를 둬서 각각의 서버 노드가 자신에게 할당된 파리미터를 저장하도록 합니다. load balancing과 recovery를 위해 물리적으로 하나인 서버가 여러 가상의 서버로 나뉘기도 합니다. 서버 매니저가 keyspace의 분할과 분배를 관리합니다.

### Replication and Consistency
![replica generation](https://www.dropbox.com/s/v9e5ews21i2ssnp/replica%20generation.png?raw=1)
각 서버 노드는 k개의 반시계 방향으로 이웃 노드들의 파라미터들을 복제해서 갖고 있습니다. 이때 원래 파라미터의 주인을 master, 복구를 위해 파라미터들을 복제해서 가지고 있는 노드들을 slave라 칭하겠습니다. master의 값이 바뀔 때마다 slave들의 값들도 synchronous하게 바꿉니다. 워커가 server에 push를 하면 slave까지 다 복제가 되어야 task가 완료됩니다. 이 때문에 delay가 생길 수 있습니다. 하지만 위 그림과 같이 여러 노드들에 의해 값이 바꼈을때 그 변화를 aggregate해서 한번에 복제해 communication을 최적화합니다. relaxed consistency 덕분에 delay가 생겨도 큰 영향을 받지 않을 수 있습니다.

### Server Node Management
fault tolerance와 dynamic scaling을 위해 노드들의 추가와 제거를 지원해야합니다. 서버 노드가 추가될 때 다음과 같은 과정을 통해 추가됩니다.
- 서버 매니저가 새로운 노드에 key range를 할당합니다. 다른 노드들의 key range가 쪼개지거나 종료된 노드의 key range가 제거될 수 있습니다.
- 서버 노드가 할당된 데이터를 가져오고 slave로서 k개의 노드에서 복제할 데이터들을 가져옵니다. 데이터 복제는 아래와 같이 두 단계로 진행됩니다.
  -데이터를 미리 복제합니다.
  -데이터를 복제하는 동안 온 메세지들을 새로운 노드에게 보내줍니다.
- 서버 매니저가 노드의 추가를 알립니다.
서버 노드가 제거될 때도 비슷한 과정을 통해 제거합니다. 서버 매니저가 서버노드들을 지켜보면서 비정상적으로 종료된 노드가 있는지 확인합니다.

### Worker Management
워커 노드를 추가하는 것은 서버노드를 추가하는 것보다 더 간단합니다.
- 태스크 스케쥴러가 새로운 워커에게 데이터를 할당합니다.
- 이미 존재하는 워커 노드들이나 네트워크 파일 시스템으로부터 데이터를 받아옵니다. 서버 노드와 달리 학습 데이터는 읽기만 하므로 메세지를 보내는 등의 추가 작업을 하지 않습니다.
- 태스크 스케쥴러가 변화를 알립니다.
프레임워크는 워커 노드를 복구할 지 말지 사용자에게 선택권을 줍니다. 이는 학습 데이터가 클 때 워커 노드를 복구하는 비용이 클 수 있기 때문입니다. 또한 학습 데이터 중 조금을 잃는 것은 모델 학습에 영향을 거의 주지 않기 때문입니다.

## Evaluation
저자는 여러가지 적용 예시들에서 파라미터 서버를 실험해봅니다. 자세한 실험 조건이나 결과를 알고 싶으신 분들은 논문을 참고하는 것을 추천드립니다.
[!convergence of sparse logistic regression](https://www.dropbox.com/s/8fucdcnhnvz7qes/convergence%20of%20sparse%20logistic%20regression.png?raw=1)
위 실험은 세가지 시스템이 같은 objective value에 도달할 때 까지 걸리는 시간을 측정해본 것입니다. System A, System B에 비해 더 빨리 수렴한 것을 확인 할 수 있습니다. 심지어 System B와는 같은 알고리즘을 사용했지만 network traffic을 낮추고 relaxed consistency model을 사용해 더 좋은 성능을 보인 것을 확인할 수 있습니다.
[!time per worker spent on computation and waiting](https://www.dropbox.com/s/pk2xefi7p5hz4c2/time%20per%20worker%20spent%20on%20computation%20and%20waiting.png?raw=1)
위 그래프는 relaxed consistency model 덕분에 워커 노드를 더 효율적으로 사용할 수 있게 됨을 보여줍니다. 전의 task들이 끝나는 것을 기다리지 않고 다음 것을 처리함으로써 delay를 최소화 할 수 있습니다.

## Conclusion
이 논문은 딥러닝 분산 학습 분야에서 사용되는 중요한 개념을 소개합니다. consistent hashing 같이 시스템적인 측면에서 좋은 테크닉들을 조화롭게 사용해 elastic scalability, fault tolerance와 같은 중요한 부분들을 지원합니다.

후에 딥러닝 분산 학습에 큰 영향을 미친 논문을 읽고 요약해봐서 뜻 깊었습니다. 또한 딥러닝 분산 학습 프레임워크를 만들 때 어떤 점들을 고려해야 되는지 알 수 있어 좋았습니다.

## Reference
[Parameter Server for Distributed Machine learning](https://medium.com/coinmonks/parameter-server-for-distributed-machine-learning-fd79d99f84c3#d24d)