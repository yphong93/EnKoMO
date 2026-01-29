#!/usr/bin/env python3
"""
NASA Bearing Dataset Preprocessing Script

NASA bearing dataset의 개별 파일들을 하나의 CSV 파일로 합치는 스크립트입니다.
각 파일은 타임스탬프 형태의 이름을 가지고 있으며, 8개 센서 채널의 진동 데이터를 포함합니다.

Data format:
- 각 파일: 20,481 rows × 8 columns (4 bearings × 2 sensors each)
- Column names: Bearing1_X, Bearing1_Y, Bearing2_X, Bearing2_Y, Bearing3_X, Bearing3_Y, Bearing4_X, Bearing4_Y
- File naming: YYYY.MM.DD.HH.MM.SS
"""

import os
import csv
import math
from datetime import datetime
import re

def parse_filename_timestamp(filename):
    """파일명에서 타임스탬프 추출"""
    # 파일명 형태: 2003.10.22.12.06.24
    pattern = r'(\d{4})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})'
    match = re.match(pattern, filename)
    
    if match:
        year, month, day, hour, minute, second = map(int, match.groups())
        return datetime(year, month, day, hour, minute, second)
    else:
        return None

def load_single_file(filepath):
    """단일 데이터 파일 로드"""
    try:
        data = []
        with open(filepath, 'r') as f:
            for line in f:
                # 탭으로 구분된 8개 값 읽기
                values = line.strip().split('\t')
                if len(values) == 8:
                    # float으로 변환
                    row = [float(v) for v in values]
                    data.append(row)
        
        return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def calculate_mean(data):
    """평균 계산"""
    return sum(data) / len(data)

def calculate_rms(data):
    """RMS (Root Mean Square) 계산"""
    return math.sqrt(sum(x * x for x in data) / len(data))

def calculate_std(data):
    """표준편차 계산"""
    mean = calculate_mean(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return math.sqrt(variance)

def calculate_max_abs(data):
    """절댓값의 최대값 계산"""
    return max(abs(x) for x in data)

def aggregate_data(data_files, method='rms'):
    """
    각 파일의 데이터를 집계하여 시계열 데이터 생성
    
    Args:
        data_files: (timestamp, filepath) 튜플 리스트
        method: 집계 방법 ('mean', 'rms', 'std', 'max')
    
    Returns:
        tuple: (aggregated_data, timestamps, column_names)
    """
    aggregated_data = []
    timestamps = []
    
    print(f"Processing {len(data_files)} files with {method} aggregation...")
    
    for i, (timestamp, filepath) in enumerate(data_files):
        if i % 50 == 0:
            print(f"Processing file {i+1}/{len(data_files)}: {os.path.basename(filepath)}")
        
        # 파일 로드
        file_data = load_single_file(filepath)
        if file_data is None:
            continue
        
        # 각 컬럼별로 집계 (8개 컬럼)
        aggregated_row = []
        for col_idx in range(8):
            # 해당 컬럼의 모든 값들 추출
            column_data = [row[col_idx] for row in file_data]
            
            # 집계 방법에 따른 처리
            if method == 'mean':
                value = calculate_mean(column_data)
            elif method == 'rms':
                value = calculate_rms(column_data)
            elif method == 'std':
                value = calculate_std(column_data)
            elif method == 'max':
                value = calculate_max_abs(column_data)
            else:
                raise ValueError(f"Unknown aggregation method: {method}")
            
            aggregated_row.append(value)
        
        aggregated_data.append(aggregated_row)
        timestamps.append(timestamp)
    
    # 컬럼명 생성
    columns = [
        'Bearing1_X', 'Bearing1_Y', 
        'Bearing2_X', 'Bearing2_Y',
        'Bearing3_X', 'Bearing3_Y', 
        'Bearing4_X', 'Bearing4_Y'
    ]
    
    return aggregated_data, timestamps, columns

def create_bearing_csv(input_dir, output_path, bearing_id=1, method='rms', max_files=None):
    """
    NASA bearing dataset을 CSV 파일로 변환
    
    Args:
        input_dir: 입력 디렉토리 경로 (예: data/real_data/nasa_bearing/1st_test)
        output_path: 출력 CSV 파일 경로 (예: data/real_data/bearing_1.csv)
        bearing_id: 베어링 ID (1, 2, 3, 4)
        method: 집계 방법 ('mean', 'rms', 'std', 'max')
        max_files: 처리할 최대 파일 수 (None이면 모든 파일)
    """
    
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    print(f"🔍 Scanning directory: {input_dir}")
    
    # 모든 데이터 파일 찾기
    data_files = []
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        if os.path.isfile(filepath) and not filename.startswith('.'):
            timestamp = parse_filename_timestamp(filename)
            if timestamp is not None:
                data_files.append((timestamp, filepath))
    
    print(f"📁 Found {len(data_files)} data files")
    
    if len(data_files) == 0:
        raise ValueError("No valid data files found in the directory")
    
    # 시간순으로 정렬
    data_files.sort(key=lambda x: x[0])
    
    # 파일 수 제한
    if max_files is not None:
        data_files = data_files[:max_files]
        print(f"📊 Processing first {len(data_files)} files")
    
    print(f"⏰ Time range: {data_files[0][0]} to {data_files[-1][0]}")
    
    # 데이터 집계
    aggregated_data, timestamps, columns = aggregate_data(data_files, method=method)
    
    # 출력 디렉토리 생성
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # CSV 저장
    print(f"💾 Saving to: {output_path}")
    print(f"📊 Final dataset shape: ({len(aggregated_data)}, {len(columns)})")
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # 헤더 쓰기
        header = ['timestamp'] + columns
        writer.writerow(header)
        
        # 데이터 쓰기
        for i, (timestamp, row) in enumerate(zip(timestamps, aggregated_data)):
            csv_row = [timestamp.strftime('%Y-%m-%d %H:%M:%S')] + list(row)
            writer.writerow(csv_row)
    
    print("✅ Conversion completed successfully!")
    print(f"📈 Dataset info:")
    print(f"   - Shape: ({len(aggregated_data)}, {len(columns)})")
    print(f"   - Columns: {columns}")
    print(f"   - Time span: {timestamps[0]} to {timestamps[-1]}")
    print(f"   - Data points: {len(timestamps)} time steps")
    
    # 간단한 통계 출력
    print("\n📊 Basic Statistics:")
    for i, col in enumerate(columns):
        values = [row[i] for row in aggregated_data]
        mean_val = calculate_mean(values)
        std_val = calculate_std(values)
        min_val = min(values)
        max_val = max(values)
        print(f"   {col}: mean={mean_val:.4f}, std={std_val:.4f}, min={min_val:.4f}, max={max_val:.4f}")
    
    return aggregated_data, timestamps

def main():
    """메인 함수 - 기본 설정으로 변환 실행"""
    
    # 기본 설정
    input_dir = "data/real_data/nasa_bearing/1st_test"
    output_dir = "data/real_data"
    
    print("🏭 NASA Bearing Dataset Conversion Tool")
    print("=" * 50)
    
    # 기본 bearing_1.csv 파일 생성 (RMS 방법)
    try:
        print(f"\n🎯 Creating bearing_1.csv file (RMS method - best for vibration analysis)")
        
        data, timestamps = create_bearing_csv(
            input_dir=input_dir,
            output_path=f"{output_dir}/bearing_1.csv",
            bearing_id=1,
            method='rms',
            max_files=500  # 처음 500개 파일 처리
        )
        
        print(f"🏆 bearing_1.csv created successfully!")
        print(f"📄 File location: {output_dir}/bearing_1.csv")
        
        # 추가로 다른 방법들도 생성 (선택적)
        print(f"\n🔄 Creating additional datasets with different aggregation methods...")
        
        methods = ['mean', 'std', 'max']
        for method in methods:
            try:
                output_path = f"{output_dir}/bearing_1_{method}.csv"
                create_bearing_csv(
                    input_dir=input_dir,
                    output_path=output_path,
                    bearing_id=1,
                    method=method,
                    max_files=100  # 더 작은 샘플로 다른 방법들 테스트
                )
                print(f"✅ {output_path} created")
            except Exception as e:
                print(f"⚠️  Warning: Could not create {method} dataset: {e}")
        
    except Exception as e:
        print(f"❌ Error creating main dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
