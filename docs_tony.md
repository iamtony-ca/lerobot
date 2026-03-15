# LeRobot 기반 OMX 로봇의 ACT 알고리즘 적용 및 모방 학습(Imitation Learning) 트러블슈팅 리뷰

**목적:** 본 문서는 LeRobot 프레임워크와 실물 OMX 로봇을 활용하여 'Pick and Place' Task를 수행하는 과정에서 발생한 모방 학습(Imitation Learning)의 한계점과 극복 과정을 ACT(Action Chunking with Transformers) 알고리즘의 이론적 배경을 바탕으로 분석합니다.

## 1. 시스템 파이프라인 및 데이터 수집 전략
본 테스트는 Mobile ALOHA 시스템의 엔드투엔드(End-to-End) 모방 학습 파이프라인을 실물 로봇에 적용하여 진행되었습니다.
* **하드웨어 구성:** OMX 로봇(Leader-Follower 양팔 원격 조작 시스템), 2D RGB 카메라 2대(전면, 손목).
* **소프트웨어 스택:** LeRobot 프레임워크, ACT(Action Chunking with Transformers) 정책, 비동기 추론 서버-클라이언트 아키텍처.
* **커리큘럼 기반 데이터 수집 (총 100 Episodes):**
  * **Phase 1 (초기 50개):** Goal Target(흰색 그릇)의 위치는 고정하고, Pick 대상(인형)의 위치만 랜덤하게 변경하여 수집.
  * **Phase 2 (후기 50개):** Goal Target과 인형의 위치를 모두 랜덤하게 변경하며 수집. (이 중 25개는 밝은 조명, 25개는 덜 밝은 조명 환경에서 수집).

## 2. 테스트 케이스 분석: 실패와 극복의 이론적 고찰

### Case 1: 공간적 과적합에 의한 Place 실패 (50 Ep 모델)
* **상황:** 초기 50개 데이터로 학습된 모델. 추론 시 인형의 위치 변경에는 대응해 Pick을 성공했으나, 흰색 그릇(Goal)의 위치를 변경하자 시각적 위치를 무시하고 원래 그릇이 있던 '허공'에 인형을 놓음.
* **이론적 분석:** 전형적인 허위 상관관계(Spurious Correlation) 및 공간적 과적합(Spatial Overfitting) 현상입니다. 데이터셋에서 그릇의 위치 변동성이 없었기 때문에, 모델의 시각적 어텐션(Visual Attention)이 붕괴되었습니다. 즉, 그릇을 눈으로 찾는 대신 "인형을 잡은 후에는 무조건 특정 절대 좌표로 이동한다"는 잘못된 인과관계를 학습한 결과입니다.

### Case 2: 기하학적 모호성에 의한 Pick 조기 실패 (50 Ep 모델)
* **상황:** 평면(X, Y) 위치는 찾아갔으나, 깊이(Z)를 정확히 파악하지 못해 객체보다 약간 위쪽 허공에서 그리퍼를 닫아버림.
* **이론적 분석:** 명시적인 Depth 센서가 없는 2D RGB 비전 전용 모델이 겪는 **기하학적 모호성(Geometric Ambiguity)**의 한계입니다. 단 50개의 데모만으로는 객체의 크기 변화, 그림자 등 암묵적 깊이(Implicit Depth)를 유추할 수 있는 시각적 단서(Visual Cues)를 충분히 일반화하지 못했습니다. 손목 카메라의 원근 왜곡에 모델이 과적합된 상태입니다.

### Case 3: 데이터 다양성을 통한 완전한 Pick & Place 성공 (100 Ep 모델)
* **상황:** 100개 데이터(Target & Object 모두 랜덤 포함)로 학습된 모델. 두 객체의 위치를 모두 변경해도 정확하게 Pick & Place를 성공함.
* **이론적 분석:** Phase 2에서 수집된 랜덤 데이터가 허위 상관관계를 타파했습니다. 모델은 훈련 오차를 줄이기 위해 흰색 그릇의 시각적 좌표를 반드시 추적하도록 강제되었으며, 100개로 누적된 궤적 데이터를 통해 2D 이미지의 픽셀 변화량만으로 Z축 거리를 정밀하게 유추하는 암묵적 3D 지각력을 확보했습니다.
* **해결책 및 시사점 (Best Practice):** 데이터 수집 시, 로봇이 인지하고 상호작용해야 하는 **모든 객체(목표물, 장애물 등)에 물리적 변동성(Randomization)을 부여**해야만 모델이 올바른 시각적 인과관계를 학습합니다.

### Case 4: 폐루프 제어 기반의 Pick 재시도 성공 (100 Ep 모델)
* **상황:** 초기 파지에 실패하여 빈 그리퍼로 닫혔으나, 그대로 허공에 Place하지 않고 다시 그리퍼를 열어 위치를 재조정한 뒤 인형을 완벽하게 파지하여 성공함.
* **이론적 분석:** 로봇이 스크립트 기반 개방 루프(Open-Loop)가 아닌, 초당 30프레임의 완벽한 **시각적 폐루프 제어(Visuomotor Closed-Loop Control)**로 작동하고 있음을 증명합니다. ACT의 CVAE 구조가 인간 시연자의 "실수 후 복구"라는 다중 모달리티(Multi-modality) 패턴을 잠재 공간에 성공적으로 인코딩했으며, **시간적 앙상블(Temporal Ensembling)**이 궤적 충돌 없이 부드러운 재시도를 이끌어냈습니다.
* **해결책 및 시사점 (One-shot 성공을 위한 개선):** 초기 실패 자체를 방지하려면 모델의 State Coverage를 대폭 늘려야 합니다. Isaac Sim과 같은 물리 엔진 환경을 구축하여 무한한 조명, 텍스처, 앵글 변수가 적용된 합성 데이터(Synthetic Data)를 생성하고 이를 실제 데이터와 섞어 학습(Sim-to-Real)시킴으로써 모델의 초기 강건함을 극대화할 수 있습니다.

### Case 5: 인지적 복구를 통한 OOD 극복 (100 Ep 모델)
* **상황:** Pick 성공 후, Place 이동 중 엉뚱한 허공에 인형을 떨어뜨림. 그러나 로봇이 다시 바닥에 떨어진 인형의 위치를 재탐색하여 파지하고 올바른 흰색 그릇 위치에 Place를 성공함.
* **이론적 분석:** Phase 1(고정 Target)의 강력한 공간적 사전 지식(Spatial Prior)과 Phase 2(랜덤 Target)의 시각적 사전 지식(Visual Prior)이 추론 중 충돌하여 발생한 일시적 실수입니다. 하지만 인형이 애매한 중간에 떨어진 분포 밖(OOD: Out-of-Distribution) 상황에서도 로봇은 시각적 상태(빈 그리퍼, 떨어진 인형)를 즉각 수용하여 기존 Action Chunk를 폐기하고 새로운 최적 궤적을 샘플링하여 복구해냈습니다.
* **해결책 및 시사점 (모드 충돌 완화):** 추론(Inference) 단계에서 과거 궤적이 현재의 올바른 예측을 방해하지 않도록 파라미터 미세 튜닝이 필요합니다. `--chunk_size_threshold`를 낮추거나 `aggregate_fn_name` 방식을 최적화하여, 가장 최근 카메라 프레임에서 예측된 신선한 행동(Fresh Action)에 더 높은 가중치를 부여함으로써 중간에 물건을 떨어뜨리는 사전 지식 충돌 현상을 방지할 수 있습니다.

### Case 6: 특징 얽힘(Feature Entanglement)과 조명 강인성 (100 Ep 모델)
* **상황:** 밝은 조명에서 Pick 성공 후 이동 중 로봇이 정지하여 Jittering 발생. 가림막을 씌워 조명을 데이터셋과 유사하게 덜 밝게 만들자 다시 이동하여 Place 성공.
* **이론적 분석:** CNN 백본의 특징 얽힘(Feature Entanglement) 현상입니다. 모델은 "흰색 그릇이 특정 위치에 있을 때는 조명이 덜 밝았다"는 편향을 학습했습니다. 밝은 조명이라는 OOD 상황에 직면하자 모델의 불확실성이 치솟았고, 상충하는 미래 궤적들이 시간적 앙상블을 거치며 제자리에서 떠는 Jittering을 유발했습니다. 가림막으로 시각적 컨텍스트(In-Distribution)를 복구하자 제어가 즉각 정상화되었습니다.
* **해결책 및 시사점 (Data Augmentation):** LeRobot 프레임워크의 학습 설정(Config)에서 이미지 데이터 증강 기법을 적극 활용해야 합니다. `ColorJitter(brightness, contrast, saturation, hue)`를 적용하여 모델이 조명의 절대적인 밝기에 과적합되지 않고 객체의 기하학적 형태에만 집중하도록 유도해야 합니다.

## 3. 결론 및 향후 개선 과제
본 테스트를 통해 엔드투엔드(End-to-End) 모방 학습에서 데이터의 **양(Scale)**뿐만 아니라 **다양성과 구성 방식(Variance & Curriculum)**이 모델의 추론 안정성에 절대적인 영향을 미침을 확인했습니다. ACT 아키텍처는 충분한 시각적 다양성이 보장될 때 강력한 폐루프 제어를 통해 자가 복구(Self-Correction) 능력을 발휘합니다.

**향후 개선 방향 (Action Items):**
* **조명/환경 강인성 확보:** `image_transforms`를 통한 ColorJitter 기반 Data Augmentation 적용.
* **Sim-to-Real 도입 검토:** One-shot 성공률 향상을 위해 합성 데이터를 활용한 Data Scaling.
* **통합 시스템 구축:** 현재 독립적으로 동작하는 Python 기반 추론 클라이언트를 ROS 2 Action Server로 래핑하고, BehaviorTree.CPP와 연동하여 자율 이동 및 조작이 결합된 고도화된 모바일 매니퓰레이션 시스템으로 확장.


##
##
##

# LeRobot 기반 OMX 로봇의 ACT 알고리즘 적용 및 모방 학습(Imitation Learning) 트러블슈팅 리뷰

**목적:** 본 문서는 LeRobot 프레임워크와 실물 OMX 로봇을 활용하여 'Pick and Place' Task를 수행하는 과정에서 발생한 모방 학습(Imitation Learning)의 한계점과 극복 과정을 ACT(Action Chunking with Transformers) 알고리즘의 심층적인 이론적 배경을 바탕으로 분석합니다.

---

## 1. 시스템 파이프라인 및 데이터 수집 전략

본 테스트는 Mobile ALOHA 시스템의 엔드투엔드(End-to-End) 모방 학습 파이프라인을 실물 로봇에 적용하여 진행되었습니다.

* **하드웨어 구성:** OMX 로봇(Leader-Follower 양팔 원격 조작 시스템), 2D RGB 카메라 2대(전면, 손목).
* **소프트웨어 스택:** LeRobot 프레임워크, ACT(Action Chunking with Transformers) 정책, 비동기 추론 서버-클라이언트 아키텍처.
* **커리큘럼 기반 데이터 수집 (총 100 Episodes):**
  * **Phase 1 (초기 50개):** Goal Target(흰색 그릇)의 위치는 고정하고, Pick 대상(인형)의 위치만 랜덤하게 변경하여 수집.
  * **Phase 2 (후기 50개):** Goal Target과 인형의 위치를 모두 랜덤하게 변경하며 수집. (이 중 25개는 밝은 조명, 25개는 덜 밝은 조명 환경에서 수집).

---

## 2. 테스트 케이스 분석: 실패와 극복의 이론적 고찰

### Case 1: 공간적 과적합에 의한 Place 실패 (50 Ep 모델)
* **상황:** 초기 50개 데이터로 학습된 모델. 추론 시 인형의 위치 변경에는 대응해 Pick을 성공했으나, 흰색 그릇(Goal)의 위치를 변경하자 시각적 위치를 무시하고 원래 그릇이 있던 '허공'에 인형을 놓음.
* **심층 이론적 분석:** 딥러닝 모델은 목적 함수(Loss)를 최소화하기 위해 가장 쉬운 지름길(Shortcut Learning)을 택하는 본질적 특성을 갖습니다. 데이터셋 내에서 그릇의 위치 분산(Variance)이 $0$이었기 때문에, ACT의 Transformer 디코더는 카메라이미지 내 '그릇의 형태'에 어텐션(Attention) 가중치를 부여하는 대신, 로봇의 현재 관절 상태(Proprioception)와 시간의 흐름만으로 목표 좌표를 암기해버리는 **공간적 과적합(Spatial Overfitting)**을 일으켰습니다. 이는 인과관계(Causality)가 아닌 **허위 상관관계(Spurious Correlation)**가 학습된 전형적인 사례입니다.
* **시사점:** 데이터 수집 환경을 설계할 때, 로봇이 인지하고 상호작용해야 하는 **모든 타겟 객체에는 반드시 물리적 위치 변동성(Randomization)을 부여**하여 비전 백본(Vision Backbone)이 픽셀의 의미론적(Semantic) 특징을 강제로 추출하도록 유도해야 합니다.

### Case 2: 기하학적 모호성에 의한 Pick 조기 실패 (50 Ep 모델)
* **상황:** 평면($X, Y$) 위치는 찾아갔으나, 깊이($Z$)를 정확히 파악하지 못해 객체보다 약간 위쪽 허공에서 그리퍼를 닫아버림.
* **심층 이론적 분석:** 3D 공간을 2D 이미지로 투영(Perspective Projection)하면 필연적으로 깊이 정보가 손실되는 **불량 조건 문제(Ill-posed Problem)**가 발생합니다. 명시적인 Depth 센서가 없는 RGB 전용 모델은 객체의 스케일(Scale) 변화, 렌즈 왜곡, 조명에 의한 그림자 등 간접적인 단서(Visual Cues)를 통해 $Z$축을 암묵적으로 추정(Implicit Depth Estimation)해야 합니다. 단 50개의 에피소드만으로는 이러한 기하학적 매핑 함수를 일반화하기에 데이터의 다양성이 절대적으로 부족했습니다.
* **시사점:** 이 문제를 근본적으로 해결하기 위해서는 ZED X 등 스테레오 카메라를 활용하여 $Z$축 데이터를 명시적으로 주입(RGB-D)하는 것이 이상적입니다. 단, ACT 네트워크 구조 변경과 초근접 시 발생하는 Depth 센서 노이즈 필터링 등 하드웨어-소프트웨어 통합 레이어의 튜닝이 동반되어야 합니다. ACT Network 아키텍처 변경에 대해서는 Mobile ALOHA의 기본 ACT 백본(ResNet18)은 3채널(RGB) 입력에 맞춰져 있습니다. Depth를 추가하려면 첫 번째 합성곱 레이어(Conv1)를 4채널(RGB-D)로 수정하고 처음부터 가중치를 다시 학습시키거나, 별도의 Depth 인코더를 추가하는 등 PyTorch 코드 레벨의 수정이 필요합니다.

### Case 3: 데이터 다양성을 통한 완전한 Pick & Place 성공 (100 Ep 모델)
* **상황:** 100개 데이터(Target & Object 모두 랜덤 포함)로 학습된 모델. 두 객체의 위치를 모두 변경해도 정확하게 Pick & Place를 성공함.
* **심층 이론적 분석:** Phase 2의 추가 데이터가 모델의 시각적 어텐션을 복구했습니다. 목적지의 위치가 무작위로 변함에 따라, 로봇은 더 이상 관절 좌표의 암기만으로는 훈련 오차를 좁힐 수 없게 되었습니다. 또한, 100번의 다채로운 접근 궤적이 누적되면서 손목 카메라 영상에 나타나는 '객체 픽셀 크기의 비선형적 팽창'을 Transformer가 거리 데이터로 완벽히 치환 해석할 수 있게 되었습니다. 이는 데이터 스케일링이 RGB 모델의 3D 공간 지각력을 어떻게 향상시키는지를 증명합니다.
* **시사점:** 데이터셋을 구성할 때 모든 변수(Pick 위치, Place 위치)를 처음부터 최고 난이도로 무작위화하면 모델이 학습 방향을 잃고 발산(Underfitting)할 가능성이 있어서, 본 테스트처럼 전체 데이터의 절반은 단일 변수(Pick 위치만 변경)로 구성하여 '파지(Grasp)'라는 기본 동작의 특징 추출(Feature Extraction)을 하게 하고, 나머지 절반에 다중 변수(모두 변경)를 섞어 모델이 적절히 수렵할 수 있도록 해줌. 데이터 수집 시, 로봇이 인지하고 상호작용해야 하는 모든 객체(목표물, 장애물 등)에 물리적 변동성(Randomization)을 부여해야만 모델이 올바른 시각적 인과관계를 학습.

### Case 4: 폐루프 제어 기반의 Pick 재시도 성공 (100 Ep 모델)
* **상황:** 초기 파지에 실패하여 빈 그리퍼로 닫혔으나, 그대로 허공에 Place하지 않고 다시 그리퍼를 열어 위치를 재조정한 뒤 인형을 완벽하게 파지하여 성공함.
* **심층 이론적 분석:** 개방 루프(Open-loop) 제어의 한계를 완벽히 벗어난 **시각적 폐루프 제어(Visuomotor Closed-Loop Control)**의 발현입니다. 마르코프 결정 과정(MDP) 관점에서 초기 파지 실패 직후, 카메라에 입력된 "비어 있는 그리퍼와 테이블 위의 인형"이라는 새로운 상태($s_t$)가 현재 정책 $\pi(a_t|s_t)$를 즉각 업데이트했습니다. 특히 ACT 내부의 **CVAE (Conditional Variational Autoencoder)**가 시연자의 '실수 후 궤적 수정'이라는 비선형적 다중 모달리티(Multi-modality)를 잠재 공간(Latent Space)에 인코딩해 두었기 때문에, 에러 상황에서도 동적인 궤적 재생성이 가능했습니다.
* **시사점:** 인간의 시연 데이터는 완벽할 필요가 없습니다. 오히려 미세한 실수와 복구 과정이 포함된 데모가 모델의 대응력을 높입니다. 단, One-shot 성공률 자체를 높이려면 Isaac Sim 기반의 대규모 합성 데이터(Synthetic Data)를 병합하여 Sim-to-Real 방식으로 상태 커버리지(State Coverage)를 극대화하는 접근이 필요합니다.

### Case 5: 인지적 복구를 통한 OOD 극복 (100 Ep 모델)
* **상황:** Pick 성공 후, Place 이동 중 엉뚱한 허공에 인형을 떨어뜨림. 그러나 로봇이 다시 바닥에 떨어진 인형의 위치를 재탐색하여 파지하고 올바른 흰색 그릇 위치에 Place를 성공함.
* **심층 이론적 분석:** Phase 1의 강력한 공간적 사전 지식(Spatial Prior)과 Phase 2의 시각적 사전 지식(Visual Prior)이 신경망 내에서 충돌(Competing Priors)하여 발생한 일시적 모드 붕괴입니다. 그러나 인형이 중간에 떨어져 분포 밖(OOD: Out-of-Distribution) 상태가 되었음에도, ResNet 백본이 학습한 강력한 시각적 특징 추출(Feature Extraction) 능력이 객체를 재인식해냈습니다. 이 과정에서 **시간적 앙상블(Temporal Ensembling)**이 무의미해진 과거의 Place 궤적 가중치를 빠르게 소멸시키고, 새로운 Pick 궤적을 부드럽게 오버레이(Overlay) 하였습니다.
* **시사점:** 모델 내부의 사전 지식 충돌을 최소화하려면 비동기 추론 서버의 파라미터 최적화가 중요합니다. '--chunk_size_threshold'는 높이고, '--actions_per_chunk'는 낮춰서 과거 청크의 유효 수명을 줄이고, 가장 최근 시각적 프레임이 지배적인 권한을 갖도록 제어 로직을 튜닝해야 합니다.

### Case 6: 특징 얽힘(Feature Entanglement)과 조명 강인성 (100 Ep 모델)
* **상황:** 밝은 조명에서 Pick 성공 후 이동 중 로봇이 정지하여 Jittering 발생. 가림막을 씌워 조명을 데이터셋과 유사하게 덜 밝게 만들자 다시 이동하여 Place 성공.
* **심층 이론적 분석:** 합성곱 신경망(CNN)의 전형적인 **특징 얽힘(Feature Entanglement)** 현상입니다. 모델은 객체의 위치 기하학(Geometry)과 픽셀의 절대적 조도(Illumination)를 독립 변수로 분리하지 못하고, "흰색 그릇이 이 위치에 있을 때는 조명이 항상 덜 밝았다"라는 편향을 학습했습니다. 이로 인해 밝은 조명 하에서 OOD 상태로 인식되었고, 극대화된 불확실성(Uncertainty)으로 인해 서로 상충하는 행동 청크들이 앙상블되어 상쇄(Cancellation)되면서 속도가 $0$에 수렴하는 Jittering 현상이 나타났습니다.
* **시사점:** 모델이 조명의 절대값이라는 방해 요소(Distractor)를 무시하고 객체의 형태(Semantic Shape)에만 집중하도록 유도해야 합니다. 학습 과정에서 `ColorJitter` (밝기, 대비, 채도 무작위 변경)와 같은 **데이터 증강(Data Augmentation)** 기법을 필수적으로 적용하여 도메인 이동(Domain Shift)에 대한 모델의 강인성을 확보해야 합니다.

---

## 3. 결론 및 향후 시스템 고도화 방안

본 트러블슈팅 리뷰를 통해 엔드투엔드 모방 학습에서 데이터의 절대적인 양(Scale)뿐만 아니라, **분산의 설계(Variance Design)**와 **커리큘럼(Curriculum)**이 모델의 폐루프 자가 복구(Self-Correction) 능력과 추론 안정성에 결정적인 영향을 미친다는 것을 입증했습니다.

**향후 개선 방향 (Action Items):**
1. **환경 변화에 대한 강인성 확보 (Data Augmentation):** LeRobot 학습 `config`에 `image_transforms` 파이프라인을 구축하여 조명 및 카메라 노이즈에 대한 과적합을 방지합니다.
2. **비동기 추론 파라미터 미세 튜닝:** Jittering 및 지연 보상을 최적화하기 위해, 실제 로봇 모터의 응답성에 맞춰 Action Chunking의 앙상블 가중치를 재조정합니다.

---

## 4. 본 테스트의 종합적 의의 (Significance and Implications)

본 테스트는 단순한 알고리즘의 동작 확인을 넘어, 실물 로봇 기반의 **물리적 AI(Physical AI)**를 실무에 적용하기 위한 핵심적인 통찰을 제공하며 다음과 같은 중대한 의의를 가집니다.

**1. 전통적 제어 파이프라인의 한계 돌파 및 대안 입증**
기존의 매니퓰레이션은 '객체 인식(Vision) $\rightarrow$ 상태 추정(State Estimation) $\rightarrow$ 궤적 계획(Motion Planning) $\rightarrow$ 역기구학(Inverse Kinematics) 제어'라는 복잡하고 직렬적인 파이프라인을 거쳐야 했습니다. 본 테스트는 이러한 중간 과정 없이 2D RGB 이미지와 관절 데이터만으로 이루어진 **엔드투엔드(End-to-End) 비전-행동 매핑**만으로도, 예기치 못한 에러 상황을 스스로 복구하는 수준 높은 폐루프 제어(Closed-Loop Control)가 가능함을 실물 로봇으로 입증했습니다.

**2. 데이터 중심(Data-Centric) 로보틱스의 방법론적 기준 제시**
모델의 네트워크 구조나 알고리즘을 수정하지 않고도, **'데이터의 품질, 분산(Variance) 설계, 그리고 커리큘럼'**만으로 치명적인 에러(공간적 과적합, 특징 얽힘 등)를 치유할 수 있음을 교차 검증했습니다. 이는 향후 새로운 태스크(문 열기, 장애물 치우기 등)를 로봇에게 전이 학습(Transfer Learning)시킬 때, 모델 튜닝보다 **데이터 수집 시나리오 설계**에 엔지니어링 역량을 집중해야 한다는 명확한 실무적 가이드라인을 제공합니다.

---

### * 개인 의견 및 느낀점
기존에 알고있던 이론적인 내용들을 실습을 통해 검증할 수 있었기에 의미가 있었으며, 추후 Physical AI기반 S/W 개발 시 방향성을 제시해줌.  
GPU Resource(VRAM 8.0) 한계로 Data Processing 및 Model 의 Performance 향상에 한계가 있었기 때문에, 추후, 더 좋은 GPU 사용이 가능하다면, Model 성능을 더 향상 시킬수 있을 것으로 기대됨.  
Simple task 에 대해서는 ACT 가 어느 정도 성능 보장이 되나, Complex Task 에 대해서는 Groot N1.X 와 같은 VLA 모델로 넘어가야 될 것으로 보임.  
ACT 는 Vision Backbone이 CNN기반인 Resnet 이므로, 최신 VLA 모델의 Vision Backbone(ViT) 과 비교할 때 Vision 측면에서 성능 상 한계점이 보임.  
Data Scale 과 Quality의 중요성이 보임. 또한, Leader Device 의 사용 편의성이 중요해질 것으로 보임. 더불어, Data Collection의 편의성 중요도 상승. (최근 연구에서는 Action Data 생성을 자동화하려는 방법론들도 나오는 중)  
Data Quality 측면에서는, Expert의 역할이 중요해 보임. 

---

## Reference
.. https://mobile-aloha.github.io/
.. https://github.com/huggingface/lerobot
.. https://research.nvidia.com/labs/gear/dreamgen/
.. https://mimicgen.github.io/
