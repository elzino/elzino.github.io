---
layout: post
title:  "Collaborative Filtering for Implicit Feedback Datasets 리뷰"
categories: "papers"
tags:
  - "deep learning"
  - "recommendation system"
  - "implicit feedback"
comments: true
use_math: true
---

논문 링크 : [pdf](http://yifanhu.net/PUB/cf.pdf)

최근 추천 시스템에 관심을 갖게 되었는데요. 이 논문을 다들 추천해주셔서 읽고 정리해보게 되었습니다.

논문은 제목에서 나와있듯이 Implicit feedback으로 구성된 dataset에서 어떻게 collaborative filtering을 적용해야하는지 방법론을 제시합니다. 그 과정에서 배경지식, Implicit feedback의 특징 등도 잘 정리해놓아서 처음 추천 시스템을 공부하시는 분들이 읽으시면 많은 도움이 될 것 같습니다.

## Explicit feedback vs Implicit feedback
*Explicit feedback*은 유저가 상품에 대해 직접 평가를 남긴 경우를 의미합니다. 예를들어 쿠팡에서 상품을 산 후 후기를 별점으로 남기는 경우, 넷플릭스에서 영화를 본 후 평점을 남기는 경우 등이 이에 해당합니다.

반면에 *implicit feedback*은 유저가 명시적으로 상품에 대해 평가를 한 것이 아니라 유저의 행동을 관찰해 의견을 간접적으로 반영하는 것을 의미합니다. 예를 들어 구매 기록, 검색 기록 등이 이에 해당됩니다.

이 논문 이전에는 많은 연구가 explicit feedback에 한해서 이루어졌는데요. 이는 explicit feedback이 데이터를 활용하기 더 편하기 때문입니다. 하지만 실제 산업에서 추천 시스템을 적용할 때는 explicit feedback이 존재하지 않거나 implicit feedback에 비해 데이터가 훨씬 적은 경우가 많습니다. 따라서 implicit feedback에서의 추천 시스템 연구가 중요하다고 논문은 강조하고 있습니다.

## Characteristics of implicit feedback
Implicit feedback은 explicit feedback과 비교했을 때 다른 중요한 특징들을 갖고 있습니다.

### No negative feedback
implicit feedback은 유저의 구매 기록등을 관찰함으로써 유저가 좋아할 거 같은 아이템들은 예측할 수 있습니다. 하지만 유저가 좋아하지 않는 아이템들은 예측하기 쉽지 않습니다. 예를들어 유저가 어떤 상품을 사지 않았을 때, 그 상품을 좋아하지 않아서 사지 않았을 수도 있고, 아니면 존재 자체를 몰라서 사지 않았을 수도 있습니다. explicit feedback의 경우 평점을 낮게 줌으로써 어떤 상품을 싫어하는지도 알 수 있는 것에 비교하면 큰 차이입니다. 따라서 explicit feedback은 평점이 있는 유저-상품 relationship만 dataset으로 사용을 합니다. 이에 반해 implicit feedback은 missing data(위 예시에서 구매기록이 없는 경우)도 사용을 해야합니다. 이런 missing data들이 negative feedback일 것이라고 기대를 하고 사용을 하게 됩니다.

### Implicit feedback is inherently noisy
우리가 유저의 행동을 통해 선호를 예측할 수는 있지만, 이것이 유저의 상품에 대한 positive view를 보장하지는 않습니다. 예를 들어 상품을 선물용으로 샀을 수도 있고, 상품을 산 이후에 실망을 했을 수도 있습니다.

### The numerical value of explicit feedback indicates *preference*, whereas the numberical value of implicit feedback indicates *confidence*
explicit feedback은 rating의 값에 따라 선호를 나타냅니다. 예를 들어 5점은 정말 좋아하는 경우를 나타내고, 1점은 정말 싫어하는 경우를 나타내죠. 하지만 implicit feedback의 수치는 어떠한 행동의 빈도를 나타냅니다. 예를 들어 유저가 특정 상품을 몇번 구매했는지를 나타내죠. 여기서는 더 큰 값일 수록 더 큰 선호를 나타내지 않습니다. 가장 좋아하는 영화가 한 번 본 영화일수도 있기 때문이죠. 그럼에도 불구하고 implicit feedback 수치는 유용합니다. 이것은 우리가 그 특정 기록에 대해 얼마나 확신을 갖을 수 있는지 말해주기 때문입니다. 한 번만 일어난 event는 유저의 선호와 관련이 없을 수 있지만, 여러번 반복해서 일어나는 event는 필연적으로 user의 의견을 반영하고 있기 때문입니다.

### Evaluation of implicit-feedback recommender requires appropriate measures
explicit feedback과 같이 유저가 상품에 대한 수치적 점수를 명시하는 경우는 mean square error 등을 통해 추천의 성능을 쉽게 측정할 수 있습니다. 하지만 implicit feedback의 경우 모호한 점이 많습니다.

## Previous work
여기서는 기존에 사용되던 CF(collaborative filtering) 방법에 대해 간략히 요약하고 넘어가겠습니다. CF는 과거의 user와 item간의 관계를 분석함으로써 미래의 user와 item 간의 intercation을 예측합니다.

### Neighborhood models
Negihborhood model은 크게 user-oriented method와 item-oriented method로 구성되어 있습니다. user-oriented method는 비슷한 유저의 기록을 바탕으로 unknown rating을 예측합니다. 반면 item-orietned method는 유저가 이미 평점을 내린 다른 item들과 유사도를 이용해서 unknown rating을 예측합니다. 

item-oriented method를 수식으로 나타내보겠습니다. 현재 우리의 목표는 유저 u가 item i에 얼마만큼의 평점 \\( r_{ui} \\)를 매길지 예측하는 것입니다. 먼저 유저 u가 rating한 item 중 유사도를 이용해서 item i와 가장 유사한 k개를 뽑습니다. 이를 \\( S^k(i;u) \\)라고 하겠습니다. \\( r_ui \\)의 예측값은 유사한 item들의 평점의 weighted average로 계산됩니다. 수식은 다음과 같습니다.
\\[ \hat{r}\_{ui} = \frac{\sum_{j \in S^k(i;u)}{s_{ij}r_{uj}}}{\sum_{j \in S^k(i;u)}{s_{ij}}} \\]

### Latent factor models
Latent factor model은 관측된 rating들을 잘 설명하는 latenet feature들을 찾는 것을 목표로 합니다. 각 user를 \\( x_u \in \mathbb{R}^f \\)에, 각 item을 \\( y_i \in \mathbb{R}^f \\)에 매핑하고 내적을 통해 예측합니다. \\( \hat{r}\_{ui} = x^T_uy_i \\).
\\[ \min_{x,y} \sum_{r_{ui}\ is\ known}{(r_{ui} - x^T_u y_i)^2 + \lambda(\lVert x_u \rVert^2 + \lVert y_i \rVert^2)}  \\]
여기서 lambda는 regularization을 위해 존재합니다. parameter는 보통 stochastic gradient descent를 통해 update 됩니다. 

## Our model

이제 본격적으로 논문에서 제안한 모델을 소개해드리겠습니다. 제일 중요한 부분은 implicit feedback의 특징을 잘 반영하기 위해 preference와 confidence를 도입한 것입니다. 이제 하나씩 살펴보겠습니다.

preference는 user의 선호 여부를 나타내는 binary variable입니다. \\( p_{ui}\\)는 다음과 같이 \\( r_{ui} \\)의 값을 이진화함으로써 얻어집니다.
\\[
  p_{ui} =
    \begin{cases}
      1 & r_{ui} > 0 \cr
      0 & r_{ui} = 0
    \end{cases}
\\]
여기서는 0보다 크면 preference를 1로 설정하였지만 이는 task에 따라 다르게 설정할 수 있습니다. threshold의 개념으로 생각하면 될 것 같습니다.

confidence \\( c_{ui} \\)는 우리가 preference \\( p_{ui} \\)에 얼마나 자신이 있는지를 수치로 나타냅니다. 만약 \\( r_{ui} \\)가 높다면 user가 그 상품에 대해 반복적으로 구매 혹은 시청을 했다는 것입니다. 즉 \\( r_{ui} \\)가 높다는 것은 user가 그 상품을 선호한다고 더 확신할 수 있게 해줍니다. 합리적인 \\( c_{ui} \\) 선택은 다음과 같습니다.
\\[ c_{ui} = 1 + \alpha r_{ui} \\]
여기 alpha값은 hyperparameter로 실험을 통해 잘 작동하는 값으로 설정해주면 됩니다. 후에 서술하겠지만 \\( c_{ui} \\)는 다른 방법으로도 설정해도 됩니다. 저자는 또 다른 방법으로 \\( c_{ui} = 1 + \alpha \log(1 + r_{ui} / \epsilon) \\) 과 같은 방법을 제안합니다.

논문의 모델도 latent factor model과 유사하게 각 user를 \\( x_u \in \mathbb{R}^f \\)에, 각 item을 \\( y_i \in \mathbb{R}^f \\)에 매핑해서 이 둘의 내적이 preference를 나타내도록 하는 것입니다:\\( \hat{p}\_{ui} = x^T_uy_i \\). 이를 다음과 같은 식을 최소화 하는 x, y들을 찾아야 합니다.
\\[ \min_{x,y} \sum_{u, i}{c_{ui}(p_{ui} - x^T_u y_i)^2 + \lambda(\lVert x_u \rVert^2 + \lVert y_i \rVert^2)}  \\]
위 식을 최소화 하기 위해 user-factor나 item-factor를 고정한 후 이차식에 대해 최적화를 진행하는 alternating-least-square 방법으로 문제를 풉니다.

latent factor model과 비교했을때 confidence와 preference의 개념이 추가된것을 확인할 수 있습니다. 또한 rating이 존재하는 relation들만 고려하는 것이 아니라 모든 가능한 u, i pair에 대해 optimization을 진행하게 됩니다. 따라서 연산량이 크게 증가하게 되고 효율적인 최적화 방법이 필요해집니다. 이를 논문에서 효율적으로 최적화 하는 방법 또한 소개합니다. 이 부분은 따로 정리하지 않겠습니다. 관심 있으신 분들은 논문을 보시면 좋을 것 같습니다.

## Explanning recommendation
좋은 추천 모델은 왜 그런 item들을 추천하였는지 설명할 수 있어야 합니다. item-oriented neighborhood model 방법은 상대적으로 설명이 쉬운 반면, latent factor model은 그런 설명을 하기가 어렵습니다. 하지만 저자는 제안한 모델을 설명가능하게 만들고자 식을 적당히 조절을 해서 다음과 같이 나타냅니다.
\\[ \hat{p}\_{ui} = \sum_{j:r_{uj}>0}{s^u_{ij}c_{uj}} \\]
이때 \\( c_{uj} \\)는 유저가 아이템을 얼마나 좋아하는지 나타내는 정도, \\( s^u_{ij} \\)는 아이템간의 유사도로 해석을 합니다.
이 방법을 사용하면 item-oriented neighborhood model과 유사하게 설명을 할 수 있게 됩니다.

## Experimental study
논문에서는 텔레비젼의 시청기록을 dataset으로 이용해 실험을 진행하였습니다. 4주간의 시청기록을 바탕으로 training set을 구성하였고 그 직후의 1주간의 시청기록을 test set으로 사용하였습니다.
유저가 잘 모르는 프로그램을 추천해주는 것이 중요하기 때문에 성능 측정할때 test set에서 training set에서 이미 본 프로그램들은 지운 점이 인상깊었습니다. 이 외에도 \\( r^t_{ui} \\)가 0.5 미만인 것은 제외하고 연달아 시청한 경우 가중치를 곱해 더하는 등 여러가지 추가 조정을 적용합니다. 관심있으신 분들은 논문을 찾아보시는 것을 추천드립니다.
\\[ \bar{rank} = \frac{\sum_{u,i}{r^t_{ui}rank_{ui}}}{\sum_{u,i}{r^t_{ui}}} \\]
위 metric을 이용하여 개인화 추천 없이 인기순 대로 추천을 하는 popularity, neighborhood method와 결과를 비교합니다.
![result](https://www.dropbox.com/s/0ji3u8u19mfyzqz/result.png?raw=1){: width="500px" }
실험 결과 논문의 방법인 factor 모델이 가장 좋은 성능을 보이는 것을 알 수 있습니다. 또한 dimmension을 키울수록 더 성능이 좋아지는 것을 확인할 수 있습니다.

## Conclusion
Implicit feedback 및 추천 시스템의 기초에 대해 잘 정리되어 있는 논문이라고 생각합니다. 논문을 읽고 정리하면서 많이 배울 수 있어 좋았습니다.
