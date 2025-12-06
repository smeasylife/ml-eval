"""
Streamlit 기반 모델 평가 대시보드 - 깔끔한 결과 중심
data1+증강 데이터로 학습하고 data2로 테스트하는 방식
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



# 페이지 설정
st.set_page_config(
    page_title="ML 모델 평가 대시보드(하이퍼 파라미터)",
    page_icon="🤖",
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
    """메인 애플리케이션 - 간소화된 평가 대시보드"""
    st.title("🤖 ML 모델 평가 대시보드")
    st.markdown("### 왼쪽에서 파라미터 설정 후 평가 실행 버튼을 눌러주세요")
    st.markdown("---")

    # 세션 상태 초기화
    if 'loocv_results' not in st.session_state:
        st.session_state.loocv_results = None
    if 'hyperparams' not in st.session_state:
        st.session_state.hyperparams = {}

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
        st.session_state.loocv_results = None
        st.session_state.hyperparams = {}

        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()

        try:
            progress_bar.progress(0.3)
            status_text.text("평가 실행 중...")

            # 하이퍼파라미터 설정
            hyperparams = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf
            }

            # LOOCV 평가를 위해 데이터 로드
            loocv_folders = ["data1", "data2"]
            X_loocv, y_loocv = load_folders_and_build_features(loocv_folders)

            progress_bar.progress(0.6)
            status_text.text("LOOCV 평가 중...")

            # LOOCV 평가 수행
            evaluator = ModelEvaluator()
            loocv_results = evaluator.loocv_evaluation(X_loocv, y_loocv, hyperparams=hyperparams)

            progress_bar.progress(0.9)
            status_text.text("결과 생성 중...")

            # 세션 상태에 결과 저장
            st.session_state.loocv_results = loocv_results
            st.session_state.hyperparams = hyperparams

        except Exception as e:
            st.error(f"평가 중 오류 발생: {str(e)}")

        progress_bar.progress(1.0)
        status_text.text("✅ 평가 완료!")

    # 메인 콘텐츠 - 평가 결과만 표시
    if st.session_state.loocv_results is not None:
        loocv_results = st.session_state.loocv_results
        hyperparams = st.session_state.hyperparams

        # 1. LOOCV 결과 (크게)
        st.markdown("---")
        st.header("🔍 LOOCV (Leave-One-Out Cross Validation)")

        # 평균 성능 메트릭
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("F1-Score", f"{loocv_results['mean_f1_score']:.4f}",
                     delta=f"±{loocv_results['std_f1_score']:.4f}")
        with col2:
            st.metric("Accuracy", f"{loocv_results['mean_accuracy']:.4f}",
                     delta=f"±{loocv_results['std_accuracy']:.4f}")
        with col3:
            st.metric("Precision", f"{loocv_results['mean_precision']:.4f}",
                     delta=f"±{loocv_results['std_precision']:.4f}")
        with col4:
            st.metric("샘플 수", f"{loocv_results['total_samples']}",
                     delta="LOOCV")

        # F1-Score 분포 (LOOCV에서는 정답 여부에 따른 분포)
        fold_data = loocv_results['fold_results']
        
        # 정답 여부를 F1-Score처럼 표시 (1.0: 정답, 0.0: 오답)
        accuracy_scores = [1.0 if fold['is_correct'] else 0.0 for fold in fold_data]
        
        fig_loocv_dist = px.histogram(
            x=accuracy_scores,
            nbins=2,
            title="Sample-wise Prediction Accuracy (LOOCV)",
            labels={'x': 'Accuracy per Sample', 'count': 'Number of Samples'},
            color_discrete_sequence=['lightgreen'] if accuracy_scores.count(1.0) > accuracy_scores.count(0.0) else ['lightcoral']
        )
        fig_loocv_dist.update_xaxes(tickvals=[0, 1], ticktext=['Incorrect (0.0)', 'Correct (1.0)'])
        fig_loocv_dist.update_layout(height=350)
        st.plotly_chart(fig_loocv_dist, use_container_width=True)

        # 2. Confusion Matrix
        st.markdown("---")
        st.header("🧩 Confusion Matrix")

        cm = loocv_results['confusion_matrix']
        fig_cm = plot_confusion_matrix_plotly(cm, CATEGORY_LIST, "")
        st.plotly_chart(fig_cm, use_container_width=True)

        # 3. Prediction Probability
        st.markdown("---")
        st.header("📊 Prediction Probability")

        # 정답/오답 경우에 따른 예측 확률 시각화
        try:
            if 'correct_probabilities' in loocv_results and 'incorrect_probabilities' in loocv_results:
                correct_probs = loocv_results['correct_probabilities']  # 정답 클래스에 대한 확률들
                incorrect_probs = loocv_results['incorrect_probabilities']  # 오답 클래스에 대한 확률들

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
                
            else:
                st.warning("LOOCV probability data not available")

        except Exception as e:
            st.warning(f"Prediction probability visualization failed: {str(e)}")

        # 4. 최종 평가 지표
        st.markdown("---")
        st.header("🎯 Final Scores")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("### F1-Score", f"{loocv_results['mean_f1_score']:.4f}")
        with col2:
            st.metric("### Accuracy", f"{loocv_results['mean_accuracy']:.4f}")

        # 요약 정보
        correct_count = sum(1 for fold in fold_data if fold['is_correct'])
        total_count = len(fold_data)
        
        st.info(f"""
        **Evaluation Summary**
        - **Total Samples**: {loocv_results['total_samples']}
        - **Overall F1-Score**: {loocv_results['mean_f1_score']:.4f}
        - **Overall Accuracy**: {loocv_results['mean_accuracy']:.4f}
        - **Correct Predictions**: {correct_count}/{total_count} ({100*correct_count/total_count:.1f}%)
        - **Incorrect Predictions**: {total_count-correct_count}/{total_count} ({100*(total_count-correct_count)/total_count:.1f}%)
        """)

    else:
        # 초기 화면 - 간단한 안내만
        st.info("👈 왼쪽 사이드바에서 파라미터를 설정하고 '모델 평가 실행' 버튼을 클릭하세요.")


if __name__ == "__main__":
    main()