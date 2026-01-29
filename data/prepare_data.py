#!/usr/bin/env python3
"""
데이터 준비 스크립트
실제 데이터를 다운로드하거나 예제 데이터를 생성합니다.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_data_directory():
    """데이터 디렉토리 생성"""
    data_dir = 'data/real_data'
    os.makedirs(data_dir, exist_ok=True)
    print(f"✓ Created directory: {data_dir}")
    return data_dir

def download_ett_data(data_dir):
    """ETT 데이터 다운로드 안내"""
    print("\n" + "="*60)
    print("ETT 데이터 다운로드")
    print("="*60)
    print("ETT 데이터는 수동으로 다운로드해야 합니다.")
    print("\n다운로드 방법:")
    print("1. GitHub에서 직접 다운로드:")
    print("   https://github.com/zhouhaoyi/ETDataset")
    print("\n2. wget 사용:")
    print(f"   cd {data_dir}")
    print("   wget https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv")
    print("   wget https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv")
    print("   wget https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv")
    print("   wget https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv")
    print("\n필요한 파일:")
    print("  - ETTh1.csv")
    print("  - ETTh2.csv")
    print("  - ETTm1.csv")
    print("  - ETTm2.csv")

def generate_sample_sst(data_dir):
    """예제 SST 데이터 생성"""
    print("\n생성 중: SST 예제 데이터...")
    dates = pd.date_range(start='2000-01-01', end='2020-12-31', freq='D')
    # 계절성 패턴을 가진 SST 데이터 생성
    sst_data = (20 + 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25) 
                + 2 * np.sin(4 * np.pi * np.arange(len(dates)) / 365.25)
                + np.random.normal(0, 0.5, len(dates)))
    
    df = pd.DataFrame({
        'date': dates,
        'sst': sst_data
    })
    
    filepath = os.path.join(data_dir, 'sst.csv')
    df.to_csv(filepath, index=False)
    print(f"✓ Generated: {filepath} ({len(df)} samples)")

def generate_sample_weather(data_dir):
    """예제 기상 데이터 생성"""
    print("\n생성 중: Weather 예제 데이터...")
    dates = pd.date_range(start='2010-01-01', end='2020-12-31', freq='H')
    n_features = 21
    
    data = {'date': dates}
    for i in range(n_features):
        # 각 특성에 다른 주기와 진폭을 가진 패턴 생성
        period = 24 * 365.25 / (i + 1)  # 다양한 주기
        data[f'feature_{i}'] = (
            np.random.normal(0, 1, len(dates)) 
            + 2 * np.sin(2 * np.pi * np.arange(len(dates)) / period)
            + 0.5 * np.sin(4 * np.pi * np.arange(len(dates)) / period)
        )
    
    df = pd.DataFrame(data)
    filepath = os.path.join(data_dir, 'weather.csv')
    df.to_csv(filepath, index=False)
    print(f"✓ Generated: {filepath} ({len(df)} samples, {n_features} features)")

def generate_sample_bearing(data_dir, bearing_id=1):
    """예제 베어링 데이터 생성"""
    print(f"\n생성 중: NASA Bearing 예제 데이터 (bearing_{bearing_id})...")
    n_samples = 10000
    time = np.arange(n_samples)
    
    # 베어링 진동 패턴 시뮬레이션
    base_freq = 100
    data = {
        'time': time,
        'ch1': (np.random.normal(0, 0.5, n_samples) 
                + 0.5 * np.sin(2 * np.pi * time / base_freq)
                + 0.2 * np.sin(4 * np.pi * time / base_freq)),
        'ch2': (np.random.normal(0, 0.5, n_samples) 
                + 0.5 * np.sin(2 * np.pi * time / (base_freq * 1.1))
                + 0.2 * np.sin(4 * np.pi * time / (base_freq * 1.1))),
        'ch3': (np.random.normal(0, 0.5, n_samples) 
                + 0.5 * np.sin(2 * np.pi * time / (base_freq * 0.9))
                + 0.2 * np.sin(4 * np.pi * time / (base_freq * 0.9))),
        'ch4': (np.random.normal(0, 0.5, n_samples) 
                + 0.5 * np.sin(2 * np.pi * time / (base_freq * 1.05))
                + 0.2 * np.sin(4 * np.pi * time / (base_freq * 1.05))),
    }
    
    df = pd.DataFrame(data)
    filepath = os.path.join(data_dir, f'bearing_{bearing_id}.csv')
    df.to_csv(filepath, index=False)
    print(f"✓ Generated: {filepath} ({len(df)} samples, 4 channels)")

def verify_data(data_dir):
    """다운로드된 데이터 검증"""
    print("\n" + "="*60)
    print("데이터 검증")
    print("="*60)
    
    ett_files = ['ETTh1.csv', 'ETTh2.csv', 'ETTm1.csv', 'ETTm2.csv']
    other_files = ['sst.csv', 'weather.csv', 'bearing_1.csv']
    
    print("\nETT 데이터:")
    for filename in ett_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"  ✓ {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            print(f"  ✗ {filename}: Not found")
    
    print("\n기타 데이터:")
    for filename in other_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"  ✓ {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            print(f"  ✗ {filename}: Not found")

def main():
    """메인 함수"""
    print("="*60)
    print("EnKoMa 실험 데이터 준비 스크립트")
    print("="*60)
    
    # 디렉토리 생성
    data_dir = create_data_directory()
    
    # ETT 데이터 다운로드 안내
    download_ett_data(data_dir)
    
    # 예제 데이터 생성
    print("\n" + "="*60)
    print("예제 데이터 생성 (테스트용)")
    print("="*60)
    print("\n실제 데이터가 없는 경우 테스트를 위해 예제 데이터를 생성합니다.")
    print("실제 실험을 위해서는 ETT 데이터를 다운로드하세요.\n")
    
    generate_sample_sst(data_dir)
    generate_sample_weather(data_dir)
    generate_sample_bearing(data_dir, bearing_id=1)
    
    # 데이터 검증
    verify_data(data_dir)
    
    print("\n" + "="*60)
    print("완료!")
    print("="*60)
    print("\n다음 단계:")
    print("1. ETT 데이터를 다운로드하세요 (위의 안내 참조)")
    print("2. config 파일에서 data_path를 확인하세요")
    print("3. compare_experiment.py를 실행하세요")
    print("\n예제:")
    print("  python compare_experiment.py configs/config_ETTh1.json")

if __name__ == "__main__":
    main()

