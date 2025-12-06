"""
Streamlit 기반 모델 평가 발표용 대시보드
data1+증강 데이터로 학습하고 data2로 테스트하는 방식
실제 발표용 페이지
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# ML 관련 import
from evaluation.model_evaluator import ModelEvaluator
from dataset import CATEGORY_LIST
from src.features import load_folders_and_build_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

# 페이지 설정
st.set_page_config(
    page_title="ML 모델 평가 발표",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def plot_confusion_matrix_plotly(cm, class_names, title):
    """Plotly로 혼동 행렬을 생성합니다."""
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=class_names,
        y=class_names,
        color_continuous_scale='Blues',
        title=title
    )

    # 숫자 표시
    fig.update_traces(
        text=np.around(cm.astype(int), decimals=0),
        texttemplate="%{text}",
        textfont={"size": 12}
    )

    fig.update_layout(
        width=600,
        height=600,
        xaxis_title="Predicted Label",
        yaxis_title="True Label"
    )

    return fig


def main():
    """메인 애플리케이션 - 발표용 평가 대시보드"""
    st.title("📊 ML 모델 평가 발표")
    st.markdown("### 왼쪽에서 파라미터 설정 후 평가 실행 버튼을 눌러주세요")
    st.markdown("---")

    # 세션 상태 초기화
    if 'presentation_results' not in st.session_state:
        st.session_state.presentation_results = None
    if 'presentation_hyperparams' not in st.session_state:
        st.session_state.presentation_hyperparams = {}

    # 사이드바 - 파라미터 설정
    st.sidebar.title("⚙️ 파라미터 설정")

    # Random Forest 하이퍼파라미터 설정
    st.sidebar.subheader("🌲 Random Forest")

    n_estimators = st.sidebar.slider(
        "n_estimators",
        min_value=10,
        max_value=1000,
        value=300,
        step=10
    )

    max_depth_option = st.sidebar.radio(
        "max_depth",
        options=["제한 없음", "직접 설정"],
        index=1
    )

    if max_depth_option == "직접 설정":
        max_depth = st.sidebar.slider(
            "최대 깊이",
            min_value=1,
            max_value=100,
            value=10,
            step=1
        )
    else:
        max_depth = None

    min_samples_split = st.sidebar.slider(
        "min_samples_split",
        min_value=2,
        max_value=20,
        value=2,
        step=1
    )

    min_samples_leaf = st.sidebar.slider(
        "min_samples_leaf",
        min_value=1,
        max_value=10,
        value=1,
        step=1
    )

    # 모델 평가 실행 버튼
    if st.sidebar.button("🚀 모델 평가 실행", type="primary"):
        # 기존 결과 초기화
        st.session_state.presentation_results = None
        st.session_state.presentation_hyperparams = {}

        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()

        try:
            progress_bar.progress(0.2)
            status_text.text("데이터 로딩 중...")

            # 하이퍼파라미터 설정
            hyperparams = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf
            }

            # 학습 데이터 로드 (data1 + 증강 데이터)
            train_folders = ["data1", "augmented_data"]
            try:
                X_train, y_train = load_folders_and_build_features(train_folders)
            except:
                # 증강 데이터가 없는 경우 fallback
                train_folders = ["data1"]
                X_train, y_train = load_folders_and_build_features(train_folders)

            progress_bar.progress(0.5)
            status_text.text("모델 학습 중...")

            # 모델 생성 및 학습
            model = RandomForestClassifier(
                n_estimators=hyperparams['n_estimators'],
                max_depth=hyperparams['max_depth'],
                min_samples_split=hyperparams['min_samples_split'],
                min_samples_leaf=hyperparams['min_samples_leaf'],
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)

            progress_bar.progress(0.7)
            status_text.text("테스트 평가 중...")

            # 테스트 데이터 로드 (data2)
            test_folders = ["data2"]
            X_test, y_test = load_folders_and_build_features(test_folders)

            # 예측 수행
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            # 성능 지표 계산
            accuracy = accuracy_score(y_test, y_pred)
            precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
                y_test, y_pred, labels=CATEGORY_LIST, zero_division=0
            )

            # 전체 평균 지표 (macro average)
            macro_precision = np.mean(precision_per_class)
            macro_recall = np.mean(recall_per_class)
            macro_f1 = np.mean(f1_per_class)

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred, labels=CATEGORY_LIST)

            # 정답/오답 확률 분리
            correct_probabilities = []
            incorrect_probabilities = []

            for i in range(len(y_test)):
                true_class = y_test.iloc[i]
                pred_class = y_pred[i]
                class_idx = CATEGORY_LIST.index(pred_class)
                prob = y_proba[i][class_idx]

                if true_class == pred_class:
                    correct_probabilities.append(prob)
                else:
                    incorrect_probabilities.append(prob)

            progress_bar.progress(0.9)
            status_text.text("결과 생성 중...")

            # 결과 저장
            presentation_results = {
                'accuracy': accuracy,
                'precision': macro_precision,
                'recall': macro_recall,
                'f1_score': macro_f1,
                'confusion_matrix': cm,
                'y_test': y_test,
                'y_pred': y_pred,
                'correct_probabilities': correct_probabilities,
                'incorrect_probabilities': incorrect_probabilities,
                'precision_per_class': precision_per_class,
                'recall_per_class': recall_per_class,
                'f1_per_class': f1_per_class
            }

            st.session_state.presentation_results = presentation_results
            st.session_state.presentation_hyperparams = hyperparams

        except Exception as e:
            st.error(f"평가 중 오류 발생: {str(e)}")

        progress_bar.progress(1.0)
        status_text.text("✅ 평가 완료!")

    # 메인 콘텐츠 - 평가 결과만 표시
    if st.session_state.presentation_results is not None:
        results = st.session_state.presentation_results
        hyperparams = st.session_state.presentation_hyperparams

        # 1. 종합 성능 지표 (큼게)
        st.markdown("---")
        st.header("🎯 최종 성능 지표")

        # 주요 성능 메트릭
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("### F1-Score", f"{results['f1_score']:.4f}")
        with col2:
            st.metric("### Accuracy", f"{results['accuracy']:.4f}")
        with col3:
            st.metric("### Precision", f"{results['precision']:.4f}")
        with col4:
            st.metric("### Recall", f"{results['recall']:.4f}")

        # 성능 요약 정보
        st.info(f"""
        **모델 평가 결과**
        - **학습 데이터**: data1 + 증강 데이터
        - **테스트 데이터**: data2 (독립 데이터셋)
        - **총 테스트 샘플**: {len(results['y_test'])}개
        - **정답 샘플**: {len(results['correct_probabilities'])}개
        - **오답 샘플**: {len(results['incorrect_probabilities'])}개
        """)

        # 2. Confusion Matrix
        st.markdown("---")
        st.header("🧩 Confusion Matrix")

        cm = results['confusion_matrix']
        fig_cm = plot_confusion_matrix_plotly(cm, CATEGORY_LIST, "혼동 행렬")
        st.plotly_chart(fig_cm, use_container_width=True)

        # 3. Prediction Probability
        st.markdown("---")
        st.header("📊 Prediction Probability")

        # 정답/오답 경우에 따른 예측 확률 시각화
        try:
            correct_probs = results['correct_probabilities']
            incorrect_probs = results['incorrect_probabilities']

            # 두 개의 컬럼으로 나누어 표시
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### ✅ 정답일 때 예측 확률")
                st.markdown("**실제 정답 클래스에 대한 모델의 확신도**")

                if correct_probs:
                    # 정답 확률 분포 히스토그램
                    correct_df = pd.DataFrame(correct_probs, columns=['Probability'])

                    fig_correct = px.histogram(
                        correct_df,
                        x='Probability',
                        nbins=20,
                        title="정답 예측 확률 분포",
                        labels={'Probability': '예측 확률', 'count': '샘플 수'},
                        color_discrete_sequence=['green']
                    )
                    fig_correct.update_layout(height=350)
                    st.plotly_chart(fig_correct, use_container_width=True)

                    # 정답 확률 요약
                    st.markdown("**정답 확률 요약**")
                    col1_1, col1_2 = st.columns(2)
                    with col1_1:
                        st.metric("평균 확률", f"{np.mean(correct_probs):.3f}")
                    with col1_2:
                        st.metric("표준편차", f"{np.std(correct_probs):.3f}")

                    st.info(f"정답 샘플 수: {len(correct_probs)}개")
                else:
                    st.warning("정답인 샘플이 없습니다")

            with col2:
                st.markdown("### ❌ 오답일 때 예측 확률")
                st.markdown("**잘못 예측한 클래스에 대한 모델의 확신도**")

                if incorrect_probs:
                    # 오답 확률 분포 히스토그램
                    incorrect_df = pd.DataFrame(incorrect_probs, columns=['Probability'])

                    fig_incorrect = px.histogram(
                        incorrect_df,
                        x='Probability',
                        nbins=20,
                        title="오답 예측 확률 분포",
                        labels={'Probability': '예측 확률', 'count': '샘플 수'},
                        color_discrete_sequence=['red']
                    )
                    fig_incorrect.update_layout(height=350)
                    st.plotly_chart(fig_incorrect, use_container_width=True)

                    # 오답 확률 요약
                    st.markdown("**오답 확률 요약**")
                    col2_1, col2_2 = st.columns(2)
                    with col2_1:
                        st.metric("평균 확률", f"{np.mean(incorrect_probs):.3f}")
                    with col2_2:
                        st.metric("표준편차", f"{np.std(incorrect_probs):.3f}")

                    st.warning(f"오답 샘플 수: {len(incorrect_probs)}개")
                else:
                    st.success("오답인 샘플이 없습니다")

            # 정답률 메트릭
            total_samples = len(correct_probs) + len(incorrect_probs)
            if total_samples > 0:
                accuracy_rate = len(correct_probs) / total_samples * 100
                st.markdown("---")
                st.markdown("### 📈 예측 정확도 분석")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("정답률", f"{accuracy_rate:.1f}%",
                             delta=f"{len(correct_probs)}/{total_samples} 샘플")
                with col2:
                    st.metric("정답 샘플 수", len(correct_probs))
                with col3:
                    st.metric("오답 샘플 수", len(incorrect_probs))

                # 확률 비교 분석
                st.markdown("**확률 분석**")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("정답 평균 확률", f"{np.mean(correct_probs):.3f}")
                with col_b:
                    st.metric("오답 평균 확률", f"{np.mean(incorrect_probs):.3f}")

        except Exception as e:
            st.warning(f"Prediction probability visualization failed: {str(e)}")

    else:
        # 초기 화면 - 간단한 안내만
        st.info("👈 왼쪽 사이드바에서 파라미터를 설정하고 '모델 평가 실행' 버튼을 클릭하세요.")

        st.markdown("---")
        st.header("📋 평가 방법")

        st.markdown("""
        **🎯 발표용 모델 평가**
        - **학습 데이터**: data1 + 증강 데이터
        - **테스트 데이터**: data2 (독립 데이터셋)
        - **평가 방식**: 실제 사용 시나리오 기반
        - **목표**: 모델의 실제 성능 및 일반화 능력 평가

        **🔄 데이터 증강 효과**
        - **Time Warping**: 시간 축 변형
        - **Jittering**: 노이즈 추가
        - **Rotation**: 회전 변환
        - **Scaling**: 크기 조절
        - **Shifting**: 이동 변환
        """)


if __name__ == "__main__":
    main()