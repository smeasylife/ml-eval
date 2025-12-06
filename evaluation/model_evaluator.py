"""
모델 평가를 위한 클래스
Random Forest 모델을 학습하고 평가합니다.
fit 한번만 하고 여러 평가를 수행하는 방식으로 설계되었습니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import time
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Tuple, Any

from dataset import CATEGORY_LIST
from src.features import load_folders_and_build_features


class ModelEvaluator:
    """머신러닝 모델 평가 클래스"""

    def __init__(self, category_list: list[str] = None):
        self.category_list = category_list or CATEGORY_LIST

    def basic_evaluation(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        기본 모델 평가를 수행합니다.

        Args:
            model: 평가할 모델
            X_test: 테스트 데이터
            y_test: 테스트 레이블
            model_name: 모델 이름

        Returns:
            평가 결과 딕셔너리
        """
        # 예측 수행
        y_pred = model.predict(X_test)

        # 4가지 평가 지표 계산
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # 클래스별 상세 지표
        precision_per_class = precision_score(y_test, y_pred, average=None,
                                            labels=self.category_list, zero_division=0)
        recall_per_class = recall_score(y_test, y_pred, average=None,
                                       labels=self.category_list, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, average=None,
                                labels=self.category_list, zero_division=0)

        # 혼동 행렬
        cm = confusion_matrix(y_test, y_pred, labels=self.category_list)

        # 결과 저장
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': dict(zip(self.category_list, precision_per_class)),
            'recall_per_class': dict(zip(self.category_list, recall_per_class)),
            'f1_per_class': dict(zip(self.category_list, f1_per_class)),
            'confusion_matrix': cm
        }

        return results

    def loocv_evaluation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        random_state: int = 42,
        hyperparams=None
    ) -> Dict[str, Any]:
        """
        Leave-One-Out Cross Validation을 수행합니다.
        data1과 data2만 사용하여 LOOCV를 수행합니다.
        
        Args:
            X: 특성 데이터 (data1 + data2)
            y: 레이블 데이터 (data1 + data2)
            random_state: 랜덤 시드
        
        Returns:
            LOOCV 평가 결과 딕셔너리
        """
        loo = LeaveOneOut()
        
        fold_results = []
        all_predictions = []
        all_actuals = []
        all_probabilities = []
        correct_probabilities = []  # 정답인 경우의 예측 확률
        incorrect_probabilities = []  # 오답인 경우의 예측 확률
        
        # 전체 데이터 포인트 수
        n_samples = len(X)
        
        for i, (train_idx, val_idx) in enumerate(loo.split(X), 1):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # 각 폴드마다 독립적인 모델 생성 및 학습
            if hyperparams is None:
                hyperparams = {}
            
            fold_model = RandomForestClassifier(
                n_estimators=hyperparams.get('n_estimators', 300),
                max_depth=hyperparams.get('max_depth', None),
                min_samples_split=hyperparams.get('min_samples_split', 2),
                min_samples_leaf=hyperparams.get('min_samples_leaf', 1),
                random_state=random_state,
                n_jobs=-1
            )
            fold_model.fit(X_train_fold, y_train_fold)
            
            # 해당 폴드의 검증 데이터로 예측
            y_pred = fold_model.predict(X_val_fold)
            y_proba = fold_model.predict_proba(X_val_fold)
            
            # LOOCV에서는 단일 샘플의 개별 지표가 의미가 없으므로, 정답 여부만 저장
            is_correct = (y_pred[0] == y_val_fold.iloc[0])
            
            fold_results.append({
                'fold': i,
                'sample_idx': val_idx[0],
                'prediction': y_pred[0],
                'actual': y_val_fold.iloc[0],
                'is_correct': is_correct
            })
            
            all_predictions.extend(y_pred)
            all_actuals.extend(y_val_fold)
            all_probabilities.extend(y_proba[0])  # LOOCV는 하나의 샘플만 예측하므로 [0]으로 접근
            
            # 정답/오답 여부에 따라 예측 확률 분류
            is_correct = (y_pred[0] == y_val_fold.iloc[0])
            if is_correct:
                # 정답인 경우: 실제 정답 클래스에 대한 확률만 저장
                actual_class = y_val_fold.iloc[0]
                class_index = self.category_list.index(actual_class)
                correct_probabilities.append(y_proba[0][class_index])
            else:
                # 오답인 경우: 예측한 오답 클래스에 대한 확률만 저장
                predicted_class = y_pred[0]
                class_index = self.category_list.index(predicted_class)
                incorrect_probabilities.append(y_proba[0][class_index])
        
        # 전체 예측에 대한 혼동 행렬
        overall_cm = confusion_matrix(all_actuals, all_predictions, labels=self.category_list)
        
        # 전체 예측 결과로 지표 계산
        overall_accuracy = accuracy_score(all_actuals, all_predictions)
        overall_precision = precision_score(all_actuals, all_predictions, average='weighted', zero_division=0)
        overall_recall = recall_score(all_actuals, all_predictions, average='weighted', zero_division=0)
        overall_f1 = f1_score(all_actuals, all_predictions, average='weighted', zero_division=0)
        
        # 통계 계산
        loocv_results = {
            'fold_results': fold_results,
            'mean_accuracy': overall_accuracy,
            'std_accuracy': np.std([1.0 if r['is_correct'] else 0.0 for r in fold_results]),
            'mean_precision': overall_precision,
            'std_precision': 0.0,  # 개별 precision 계산이 어려움
            'mean_recall': overall_recall,
            'std_recall': 0.0,  # 개별 recall 계산이 어려움
            'mean_f1_score': overall_f1,
            'std_f1_score': np.std([1.0 if r['is_correct'] else 0.0 for r in fold_results]),
            'confusion_matrix': overall_cm,
            'total_samples': n_samples,
            'prediction_probabilities': all_probabilities,
            'correct_probabilities': correct_probabilities,
            'incorrect_probabilities': incorrect_probabilities,
            'overall_predictions': all_predictions,
            'overall_actuals': all_actuals
        }
        
        return loocv_results

    def stratified_cross_validation(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_folds: int = 5,
        random_state: int = 42,
        hyperparams=None
    ) -> Dict[str, Any]:
        """
        올바른 Stratified K-Fold Cross Validation을 수행합니다.
        각 폴드마다 독립적으로 모델을 학습하고 평가합니다.

        Args:
            X: 특성 데이터
            y: 레이블 데이터
            n_folds: 폴드 수 (고정: 5)
            random_state: 랜덤 시드

        Returns:
            교차검증 결과 딕셔너리
        """
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        
        fold_results = []
        all_predictions = []
        all_actuals = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # 각 폴드마다 독립적인 모델 생성 및 학습
            if hyperparams is None:
                hyperparams = {}
            
            fold_model = RandomForestClassifier(
                n_estimators=hyperparams.get('n_estimators', 300),
                max_depth=hyperparams.get('max_depth', None),
                min_samples_split=hyperparams.get('min_samples_split', 2),
                min_samples_leaf=hyperparams.get('min_samples_leaf', 1),
                random_state=random_state,
                n_jobs=-1
            )
            fold_model.fit(X_train_fold, y_train_fold)
            
            # 해당 폴드의 검증 데이터로 예측
            y_pred = fold_model.predict(X_val_fold)
            
            # 지표 계산
            fold_accuracy = accuracy_score(y_val_fold, y_pred)
            fold_precision = precision_score(y_val_fold, y_pred, average='weighted', zero_division=0)
            fold_recall = recall_score(y_val_fold, y_pred, average='weighted', zero_division=0)
            fold_f1 = f1_score(y_val_fold, y_pred, average='weighted', zero_division=0)
            
            fold_results.append({
                'fold': fold,
                'accuracy': fold_accuracy,
                'precision': fold_precision,
                'recall': fold_recall,
                'f1_score': fold_f1
            })
            
            all_predictions.extend(y_pred)
            all_actuals.extend(y_val_fold)
        
        # 전체 예측에 대한 혼동 행렬
        overall_cm = confusion_matrix(all_actuals, all_predictions, labels=self.category_list)
        
        # 통계 계산
        cv_results = {
            'fold_results': fold_results,
            'mean_accuracy': np.mean([r['accuracy'] for r in fold_results]),
            'std_accuracy': np.std([r['accuracy'] for r in fold_results]),
            'mean_precision': np.mean([r['precision'] for r in fold_results]),
            'std_precision': np.std([r['precision'] for r in fold_results]),
            'mean_recall': np.mean([r['recall'] for r in fold_results]),
            'std_recall': np.std([r['recall'] for r in fold_results]),
            'mean_f1_score': np.mean([r['f1_score'] for r in fold_results]),
            'std_f1_score': np.std([r['f1_score'] for r in fold_results]),
            'confusion_matrix': overall_cm
        }
        
        return cv_results


class ModelTrainer:
    """모델 학습 클래스 - fit 한번만 수행"""

    @staticmethod
    def create_model(hyperparams=None):
        """Random Forest 모델 생성"""
        if hyperparams is None:
            hyperparams = {}
            
        return RandomForestClassifier(
            n_estimators=hyperparams.get('n_estimators', 300),
            max_depth=hyperparams.get('max_depth', None),
            min_samples_split=hyperparams.get('min_samples_split', 2),
            min_samples_leaf=hyperparams.get('min_samples_leaf', 1),
            random_state=42,
            n_jobs=-1
        )

    @staticmethod
    def train_and_evaluate_with_loocv():
        """
        Random Forest 모델 학습 및 평가 (data1+data2로 LOOCV)
        LOOCV는 data1과 data2만 사용합니다.
        
        Returns:
            평가 결과, 모델, 데이터
        """
        # LOOCV를 위한 데이터: data1 + data2만 사용
        loocv_folders = ["data1", "data2"]
        X_loocv, y_loocv = load_folders_and_build_features(loocv_folders)
        
        # LOOCV 평가 수행
        evaluator = ModelEvaluator()
        loocv_results = evaluator.loocv_evaluation(X_loocv, y_loocv)
        
        # 결과 포맷팅
        results = {
            'model_name': 'RF_LOOCV',
            'loocv': loocv_results,
            'f1_score': loocv_results['mean_f1_score'],
            'accuracy': loocv_results['mean_accuracy'],
            'precision': loocv_results['mean_precision'],
            'recall': loocv_results['mean_recall'],
            'f1_per_class': {},  # LOOCV는 전체 평가이므로 클래스별 지표 별도 계산 필요
            'precision_per_class': {},
            'recall_per_class': {},
            'confusion_matrix': loocv_results['confusion_matrix']
        }
        
        return results, None, (X_loocv, None, y_loocv, None)

    @staticmethod
    def train_and_evaluate(hyperparams=None):
        """
        Random Forest 모델 학습 및 평가 (data1+증강 데이터로 학습, data2로 테스트)
        fit() 1번만 호출하고 여러 평가를 수행합니다.
        
        Args:
            hyperparams: Random Forest 하이퍼파라미터 딕셔너리
        
        Returns:
            평가 결과, 모델, 데이터
        """
        # 학습 데이터: data1 + 증강 데이터
        train_folders = ["data1", "augmented_data"]
        X_train, y_train = load_folders_and_build_features(train_folders)
        
        # 테스트 데이터: data2 (독립적인 데이터셋)
        test_folders = ["data2"]
        X_test, y_test = load_folders_and_build_features(test_folders)
        
        # 모델 생성 및 학습 (이것이 유일한 fit 호출)
        model = ModelTrainer.create_model(hyperparams)
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # 학습된 모델로 여러 평가 수행
        evaluator = ModelEvaluator()
        
        # 1. 기본 평가 (data2로 테스트)
        basic_results = evaluator.basic_evaluation(model, X_test, y_test, "RF_Data1_Aug_To_Data2")
        basic_results['training_time'] = training_time
        
        # 2. Stratified K-Fold Cross Validation (학습 데이터에서만 수행)
        cv_results = evaluator.stratified_cross_validation(X_train, y_train, hyperparams=hyperparams)
        
        # 결과 통합
        results = {
            **basic_results,
            'cross_validation': cv_results
        }
        
        return results, model, (X_train, X_test, y_train, y_test)


def get_class_performance_table(results: Dict) -> pd.DataFrame:
    """클래스별 성능 표 생성"""
    class_metrics = pd.DataFrame({
        'Class': CATEGORY_LIST,
        'Precision': [results['precision_per_class'].get(cls, 0)
                     for cls in CATEGORY_LIST],
        'Recall': [results['recall_per_class'].get(cls, 0)
                  for cls in CATEGORY_LIST],
        'F1-Score': [results['f1_per_class'].get(cls, 0)
                    for cls in CATEGORY_LIST]
    })

    return class_metrics


def filter_perfect_f1_scores(f1_scores_dict: Dict) -> Dict:
    """
    F1-Score가 1.0인 결과를 제외한 딕셔너리를 반환합니다.
    
    Args:
        f1_scores_dict: 클래스별 F1-Score 딕셔너리
    
    Returns:
        F1-Score가 1.0이 아닌 클래스만 포함된 딕셔너리
    """
    return {cls: score for cls, score in f1_scores_dict.items() if score < 1.0}


def filter_perfect_f1_results(results: Dict) -> Dict:
    """
    F1-Score가 1.0인 결과를 제외한 결과를 반환합니다.
    
    Args:
        results: 평가 결과 딕셔너리
    
    Returns:
        F1-Score가 1.0인 클래스가 제외된 결과 딕셔너리
    """
    filtered_results = results.copy()
    
    # 클래스별 F1-Score에서 1.0 제외
    if 'f1_per_class' in filtered_results:
        filtered_results['f1_per_class'] = filter_perfect_f1_scores(filtered_results['f1_per_class'])
    
    # Precision과 Recall도 동일하게 필터링
    if 'precision_per_class' in filtered_results:
        filtered_results['precision_per_class'] = {
            cls: score for cls, score in filtered_results['precision_per_class'].items() 
            if filtered_results['f1_per_class'].get(cls, 0) < 1.0
        }
    
    if 'recall_per_class' in filtered_results:
        filtered_results['recall_per_class'] = {
            cls: score for cls, score in filtered_results['recall_per_class'].items() 
            if filtered_results['f1_per_class'].get(cls, 0) < 1.0
        }
    
    return filtered_results