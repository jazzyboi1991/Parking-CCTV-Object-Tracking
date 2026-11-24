# Parking CCTV Object Tracking

주차장 CCTV 영상에서 `car`, `person`, `shopping_cart` 세 클래스를 탐지하기 위한 과제 자료입니다. 현재 저장소에는 학습 데이터나 영상 파일을 포함하지 않습니다. 데이터는 각 출처의 이용 조건에 따라 별도로 내려받고, 테스트 영상은 권리 확인 후 로컬 `video/` 폴더에 준비해야 합니다.

## 빠른 시작

프로젝트 루트(`과제/`)에서 실행합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_training_images.py --carts 600 --seed 42
python scripts/verify_training_images.py
```

데이터 수집과 검토 절차는 아래를 참고하세요.

---

## 학습 이미지 수집 및 검증

목표 클래스는 `car`, `person`, `shopping_cart`입니다. 장면과 라벨을 같이 확인할 수 있도록 자동차·사람은 주차장/캠퍼스 고정 카메라 데이터에서, 쇼핑 카트는 트롤리 전용 데이터셋에서 수집합니다.

## 수집 데이터

- **자동차·사람:** [EPFL Multi-view Multi-class Detection](https://www.epfl.ch/labs/cvlab/data/data-multiclass/). 캠퍼스 도로와 주차 구역의 고정 카메라 이미지, 사람·자동차 상자 라벨을 사용합니다. EPFL은 연구 목적 사용을 허용하며, 결과 공개 시 출처 논문 인용을 요청합니다.
- **쇼핑 카트:** [Mendeley Shopping Trolley dataset](https://data.mendeley.com/datasets/vj9g4mn4vv/1), DOI `10.17632/vj9g4mn4vv.1`. YOLO 상자 라벨이 포함된 트롤리 데이터입니다(CC BY 4.0). 쇼핑 장면 외 이미지나 오라벨이 있을 수 있어 검토 후 승인합니다.
- 과제 테스트 영상(로컬 전용): `video/ClearPix-Camera-Grocery-Parking-clip-82-92s.mp4`. 저장소에는 영상이 포함되지 않습니다. 테스트 영상 프레임은 학습 이미지에 자동으로 섞지 않습니다.

## 직접 수집하기

Python 3.10 이상이 필요합니다. `requirements.txt`에는 이미지 검토용 Pillow, 영상 프레임 처리용 OpenCV, 객체 탐지용 Ultralytics(PyTorch 포함), 영상 다운로드용 yt-dlp를 적었습니다. FFmpeg를 사용하는 별도 영상 변환 작업에는 FFmpeg도 시스템에 설치해야 합니다.

```bash
python scripts/download_training_images.py --carts 600 --seed 42
python scripts/verify_training_images.py
```

수집 결과는 `images/training_candidates/`에 저장됩니다. 다운로드 코드는 파일의 SHA-256을 확인하고 EPFL 상자 좌표를 YOLO 형식으로 변환합니다. `--carts`는 목표 총량입니다. 같은 seed로 다시 실행하면서 수를 늘리면 기존 후보는 재사용하고 새 후보를 추가합니다. 예: `--carts 800 --seed 42`.

## 검토 및 학습용 폴더 만들기

1. `training_candidates/review/contact_sheets/`의 시트에서 초록 상자가 객체를 제대로 감싸는지, 실제 자동차·사람·쇼핑 카트인지, 프로젝트 장면에 쓸 만한지 확인합니다.
2. `training_candidates/review/review.csv`에서 사용할 행의 `approved` 칸만 `yes`로 바꿉니다. 부적합은 `no`, 판단이 어려운 항목은 빈칸으로 둡니다.
3. 승인한 이미지만 학습 폴더로 복사하고, EPFL의 동기화된 카메라 프레임은 프레임 ID 단위로 train/validation을 나눕니다.

```bash
python scripts/verify_training_images.py --promote
```

결과는 `images/training_candidates/verified_yolo_dataset/`에 생성되며 `data.yaml`은 클래스 순서를 `[car, person, shopping_cart]`로 지정합니다. `--promote` 실행 시 train/valid 파일 짝, 라벨 형식, 상자 좌표와 클래스별 객체 수까지 이어서 점검합니다. 이미지 의미 검토는 연락처 시트를 기준으로 승인한 행만 반영합니다.

## 스크립트

- `scripts/download_training_images.py`: EPFL 자동차·사람 데이터와 Mendeley 카트 후보 수집
- `scripts/verify_training_images.py`: 이미지·라벨 구조 검사, 검토표와 연락처 시트 생성, 승인 데이터셋 구성, 승격 후 감사
- `scripts/train_detector.py`: Ultralytics YOLO 학습 실행. `yolo` 명령 대신 가상환경의 Python으로 실행합니다.
- `scripts/predict_detector.py`: 학습한 가중치로 이미지·영상 추론. 작은 객체가 있는 CCTV에 맞춰 기본 입력 크기 1280, 신뢰도 임계값 0.10을 사용합니다.

승격 후 데이터셋을 점검하고 첫 모델을 학습하려면 과제 루트에서 실행합니다.

```bash
python scripts/train_detector.py
```

학습 옵션은 예를 들어 `python scripts/train_detector.py --epochs 50 --imgsz 640 --batch 8 --device mps`처럼 지정할 수 있습니다.

학습한 모델로 테스트 영상을 추론하려면 다음처럼 실행합니다.

```bash
python scripts/predict_detector.py --source video/KHOU-11-Grocery-Parking-CCTV-clip-39-44s.mp4 --imgsz 1280 --conf 0.10 --device mps --name predict-khou-39-44-improved
```

탐지 상자 영상은 `runs/detect/predict-khou-39-44-improved/`에 저장됩니다. 작은 CCTV 객체는 낮은 신뢰도 상자도 포함할 수 있으므로 결과를 눈으로 확인한 뒤 임계값을 조정하세요.

## 폴더 주의

- 이전에 잘못 추출한 CCTV 프레임과 Open Images 후보 폴더는 정리했습니다. 위 스크립트로 새 후보 데이터를 만들 수 있습니다.
- 출처별 사용 조건을 지킵니다. 이 저장소에서는 외부 이미지·라벨 데이터와 동영상 파일을 재배포하지 않습니다.
- EPFL: [Multi-view Multi-class Detection](https://www.epfl.ch/labs/cvlab/data/data-multiclass/). 연구 목적 사용 안내와 인용 조건을 확인하세요.
- Mendeley: [Shopping Trolley dataset](https://data.mendeley.com/datasets/vj9g4mn4vv/1), DOI `10.17632/vj9g4mn4vv.1`, CC BY 4.0. 재사용 시 출처와 라이선스를 표시하세요.
