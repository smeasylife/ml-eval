# ML 모델 평가 프로젝트

## Streamlit 데모 실행 방법
```
streamlit run main.py
```

## 모델 평가 대시보드

### Evaluation Dashboard (`streamlit_evaluation.py`)
**목적**: 내부 개발팀을 위한 하이퍼파라미터 최적화 및 모델 성능 분석

#### 평가 방식
- **LOOCV (Leave-One-Out Cross Validation)**: data1 + data2의 모든 데이터를 하나씩 교차 검증하여 가장 정확한 성능 측정
- **적용 대상**: 소규모 데이터셋의 신뢰성 높은 평가, 모델 한계 분석

#### 주요 평가 지표
- **F1-Score**: 클래스 불균형 환경에서의 정확도 측정 (가장 중요)
- **Accuracy**: 전체 예측 정확률
- **Precision/Recall**: 모델의 정밀도와 재현율
- **Confusion Matrix**: 구체적인 오분류 패턴 분석
- **Prediction Probability**: 정답/오답 시 모델의 확신도 분석

### Presentation Dashboard (`presentation.py`)
**목적**: 실제 발표 및 데모를 위한 모델 성능 증명

#### 평가 방식
- **Train-Test 분리**: data1 + 증강 데이터로 학습, data2로 독립 테스트
- **적용 대상**: 실제 사용 시나리오 기반 성능 발표, 일반화 능력 증명

#### 주요 평가 지표
- **종합 성능**: F1-Score, Accuracy, Precision, Recall를 크게 표시
- **Confusion Matrix**: 실제 데이터셋에서의 분류 성능 시각화
- **Prediction Probability**: 모델의 예측 신뢰도 분석 (정답/오답 확률 분포)

### 실행 방법
```bash
# 평가용 대시보드 (개발/최적화)
streamlit run evaluation/streamlit_evaluation.py

# 발표용 대시보드 (데모/증명)
streamlit run evaluation/presentation.py
```

### 데이터 구조
- **data1**: 기본 학습 데이터
- **augmented_data**: Time Warping, Jittering, Rotation, Scaling, Shifting으로 증강된 데이터
- **data2**: 독립적인 테스트 데이터 (실제 성능 검증용)
