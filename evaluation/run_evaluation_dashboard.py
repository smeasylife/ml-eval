"""
Streamlit 평가 대시보드 실행 스크립트
간단하게 모델 평가 대시보드를 실행합니다.
"""
import subprocess
import sys
import os
from pathlib import Path

def main():
    """Streamlit 대시보드 실행"""
    print("🌲 Random Forest 하이퍼파라미터 튜닝 대시보드를 시작합니다...")
    print("📊 http://localhost:8501 에서 확인하세요!")
    print("⏹️  종료하려면 Ctrl+C를 누르세요.")
    print()
    
    # 스크립트 경로
    script_path = Path(__file__).parent / "streamlit_evaluation.py"
    
    # Streamlit 실행
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(script_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 대시보드를 종료합니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("💡 의존성 설치가 필요할 수 있습니다:")
        print("   pip install -r requirements.txt")

if __name__ == "__main__":
    main()