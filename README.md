# Parking CCTV Object Detection

주차장 영상에서 **자동차(`car`)·사람(`person`)·쇼핑 카트(`shopping_cart`)**를 찾는 머신러닝 과제 저장소입니다. 학습 이미지 수집과 검토, YOLO 모델 학습, 영상 프레임 라벨 보완, 결과 영상 생성을 위한 스크립트와 현재 산출물을 정리했습니다.

현재 구현은 프레임마다 객체를 찾는 **객체 탐지**입니다. 객체에 프레임 사이의 ID를 부여하는 다중 객체 추적 기능은 포함하지 않습니다.

## 빠른 시작

저장소 루트(`과제/`)에서 실행합니다. Python 3.10 이상을 권장합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Apple Silicon Mac에서는 `--device mps`를 사용할 수 있습니다. MPS를 사용할 수 없는 환경은 `--device cpu`로 실행하세요. NVIDIA GPU는 설치된 PyTorch 환경에 맞춰 CUDA 장치를 지정하면 됩니다.

일부 영상 편집·인코딩 작업에만 FFmpeg가 필요합니다. macOS에서는 `brew install ffmpeg`로 설치할 수 있습니다. FFmpeg는 Python 패키지 목록에 포함되지 않습니다.

## 저장소 구성

```text
.
├── images/
│   ├── training_candidates/       # 출처별 원본 이미지와 검토 자료
│   │   ├── parking_lot_epfl/
│   │   ├── shopping_trolley_mendeley/
│   │   ├── review/                # 연락처 시트와 review.csv
│   │   └── verified_yolo_dataset/ # 승인된 train/valid 데이터
│   └── video_finetune/
│       ├── unlabeled/             # 초기 5초 클립 프레임
│       ├── labels_draft/          # 검토 전 라벨 초안
│       ├── review_overlays/       # 초기 클립 검토 이미지
│       ├── dataset/               # 초기 클립 파인튜닝 데이터
│       ├── original_person_review/ # 원본 영상에서 뽑은 프레임·라벨·오버레이
│       └── original_person_dataset/ # 최근 파인튜닝 데이터
├── original videos/               # 원본 영상과 5초 클립
├── results/                       # 공유·제출용 최종 모델과 감지 영상
├── runs/detect/                   # 학습 로그와 단계별 실행 결과
├── scripts/                       # 수집·검토·학습·추론 스크립트
├── 보고서/                         # 과제 보고서 파일
├── requirements.txt
└── yolo26n.pt                      # YOLO26n 시작 가중치
```

`runs/detect/`의 번호는 실험 순서입니다.

| 폴더 | 내용 |
|---|---|
| `01-train` | 기본 객체 탐지 학습 |
| `02-predict-viralhog-20-25` | 5초 클립 기본 모델 감지 |
| `03-video-label-draft` | 영상의 자동차·사람 라벨 초안 |
| `04-video-cart-draft` | 영상의 카트 라벨 초안 |
| `05-parking-video-finetune` | 주차장 데이터로 첫 파인튜닝 |
| `06-predict-viralhog-finetuned` | 첫 파인튜닝 모델의 5초 클립 감지 |
| `07-predict-viralhog-original-finetuned` | 첫 파인튜닝 모델의 원본 영상 감지 |
| `08-original-video-finetune` | 원본 영상에서 검토한 프레임을 추가한 재학습 |
| `09-predict-original-after-finetune` | 최신 모델의 원본 영상 감지 |

학습 중간 산출물과 시각화는 `runs/detect/<실험 폴더>/`에, 제출하거나 바로 확인할 결과물은 `results/`에 둡니다. `results/`에는 현재 5초 감지 영상, 원본 영상 감지 영상, 최신 `best.pt`가 있습니다.

## 이미지 데이터 만들기

### 1. 후보 이미지 수집

```bash
.venv/bin/python scripts/download_training_images.py --carts 600 --seed 42
```

이 스크립트는 EPFL의 자동차·사람 데이터와 Mendeley의 쇼핑 카트 데이터를 `images/training_candidates/`에 받습니다. 수집 이미지는 **후보**이므로 승인 없이 학습 데이터로 간주하면 안 됩니다. 같은 seed로 다시 실행하면 기존 파일을 재사용하면서 카트 목표 수를 늘릴 수 있습니다.

### 2. 이미지와 라벨을 직접 검토

```bash
.venv/bin/python scripts/verify_training_images.py
```

스크립트는 구조 검사 결과를 `images/training_candidates/review/review.csv`에 기록하고, 출처별 연락처 시트를 `review/contact_sheets/`에 만듭니다. 시트와 원본을 확인해 객체가 올바른지, 상자가 맞는지, 과제 장면에 적절한지 검토한 뒤 CSV의 `approved` 칸을 직접 표시합니다.

### 3. 승인한 이미지로 데이터셋 구성

```bash
.venv/bin/python scripts/verify_training_images.py --promote
```

승인된 행만 `images/training_candidates/verified_yolo_dataset/`로 복사합니다. 클래스 ID는 `0=car`, `1=person`, `2=shopping_cart`이며 `data.yaml`도 이 순서를 사용합니다. 이 과정은 구조와 상자 좌표도 확인하지만, 이미지 내용이 맞는지는 사람이 판단해야 합니다. 기존 승격 폴더가 있으면 다시 만들 수 있으므로, 재실행 전에 보존할 내용이 있는지 확인하세요.

현재 작업 데이터는 승인된 이미지 1,850장으로 구성되어 있습니다. 실제 파일 수는 CSV에서 승인한 항목과 수집 상태에 따라 달라질 수 있습니다.

## 기본 모델 학습

승인 데이터셋으로 YOLO26n을 학습하는 예시입니다.

```bash
.venv/bin/python scripts/train_detector.py \
  --data images/training_candidates/verified_yolo_dataset/data.yaml \
  --model yolo26n.pt \
  --epochs 50 \
  --imgsz 640 \
  --batch 8 \
  --device mps \
  --name baseline-yolo26n
```

가중치는 `runs/detect/baseline-yolo26n/weights/`에 저장됩니다. 사용 가능한 주요 옵션은 `--model`, `--data`, `--epochs`, `--imgsz`, `--batch`, `--device`, `--name`입니다. MPS 메모리가 부족하면 `--batch` 값을 낮추거나 `--imgsz`를 줄이세요.

## 영상 라벨 검토와 파인튜닝

영상 자동 라벨은 검토를 돕는 초안입니다. 누락된 객체를 추가하고 잘못된 박스를 고친 뒤 학습에 사용해야 합니다.

### 원본 영상에서 프레임 추출

```bash
.venv/bin/python scripts/prepare_original_video_review.py \
  --samples 20 \
  --device mps
```

기본 입력은 `original videos/ViralHog-Shopping-Carts-Collide-with-Cars-original.mp4`입니다. 프레임·라벨·오버레이는 `images/video_finetune/original_person_review/{frames,labels,overlays}/`에 저장됩니다. 오버레이를 보고 `labels/`의 YOLO 텍스트 파일에서 세 클래스의 박스를 모두 확인·수정하세요. 오버레이는 자동 초안 생성 시점의 이미지라 라벨 파일을 수정해도 함께 갱신되지 않습니다.

기본 출력 폴더가 이미 있으면 스크립트가 덮어쓰지 않고 종료합니다. 다시 샘플링하려면 `--output`으로 새 폴더를 지정하세요.

### 검토한 라벨로 데이터셋 생성

```bash
.venv/bin/python scripts/prepare_video_finetune.py \
  --frames images/video_finetune/original_person_review/frames \
  --labels images/video_finetune/original_person_review/labels \
  --output images/video_finetune/original_person_dataset \
  --repeat 10
```

기존 학습 이미지와 영상 프레임을 합친 YOLO 데이터셋을 만듭니다. `--repeat 10`은 각 영상 프레임을 학습 목록에 10회 넣어 비중을 높입니다. 새 시점을 만들어내지는 않습니다. 출력 폴더가 이미 있으면 덮어쓰지 않으므로, 다시 만들 때는 새 `--output` 경로를 사용하세요. 상자 가장자리에서 최대 `0.00001`의 반올림 오차는 경계 안으로 보정하며 더 큰 오류는 중단 사유로 처리합니다. 생성되는 `data.yaml`에는 로컬 경로가 기록되므로, 데이터셋을 다른 컴퓨터로 옮겼다면 해당 스크립트로 다시 생성하세요.

### 기존 모델에서 이어 학습

현재 원본 영상 파인튜닝 설정의 예시는 다음과 같습니다.

```bash
.venv/bin/python scripts/train_detector.py \
  --data images/video_finetune/original_person_dataset/data.yaml \
  --model runs/detect/05-parking-video-finetune/weights/best.pt \
  --epochs 30 \
  --imgsz 960 \
  --batch 4 \
  --device mps \
  --name next-original-video-finetune
```

`--model`에 이전 학습의 `best.pt`를 지정하면 그 가중치에서 파인튜닝합니다. 이름이 같은 실험 폴더가 이미 있으면 다른 `--name`을 사용하세요.

## 영상 감지

`predict_detector.py`는 영상 또는 이미지에 탐지 상자를 그려 저장하고, 프레임별 탐지 클래스 수와 객체 수를 출력합니다. 아래는 최신 모델로 5초 클립을 처리하는 예시입니다.

```bash
.venv/bin/python scripts/predict_detector.py \
  --model results/parking-video-finetuned-best.pt \
  --source "original videos/ViralHog-Shopping-Carts-Collide-with-Cars-clip-20-25s.mp4" \
  --imgsz 1280 \
  --conf 0.10 \
  --device mps \
  --name next-predict-clip
```

원본 영상은 `--source`만 다음처럼 바꿔 실행합니다.

```bash
.venv/bin/python scripts/predict_detector.py \
  --model results/parking-video-finetuned-best.pt \
  --source "original videos/ViralHog-Shopping-Carts-Collide-with-Cars-original.mp4" \
  --imgsz 1280 \
  --conf 0.10 \
  --device mps \
  --name next-predict-original
```

결과는 `runs/detect/<--name>/`에 저장됩니다. 재실행 시에는 기존 결과를 덮어쓰지 않도록 매번 다른 `--name`을 지정하세요. `--conf`를 낮추면 작은 객체를 더 많이 표시할 수 있지만 오탐도 늘 수 있으므로 출력 영상을 확인하며 조정합니다.

## 현재 모델 결과와 평가 범위

최근 원본 영상 파인튜닝의 기존 검증 세트(394장) 결과입니다.

| 클래스 | 정밀도 | 재현율 | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|
| 전체 | 0.921 | 0.908 | 0.937 | 0.736 |
| 자동차 | 0.991 | 0.987 | 0.995 | 0.982 |
| 사람 | 0.910 | 0.948 | 0.978 | 0.660 |
| 쇼핑 카트 | 0.863 | 0.790 | 0.837 | 0.565 |

검증 수치는 기존 데이터의 검증 이미지에서 측정했습니다. 파인튜닝에 원본 영상에서 뽑은 프레임이 들어갔으므로, 같은 원본 영상에 대한 감지 결과는 독립적인 성능 평가가 아닙니다. 일반화 성능을 보려면 학습에 포함하지 않은 다른 영상으로 평가해야 합니다.

## 스크립트 안내

| 파일 | 역할 |
|---|---|
| `download_training_images.py` | EPFL·Mendeley 이미지와 라벨 수집 |
| `verify_training_images.py` | 구조 검사, 검토 CSV·연락처 시트 생성, 승인 데이터셋 구성 |
| `train_detector.py` | YOLO 모델 학습 또는 파인튜닝 |
| `prepare_original_video_review.py` | 원본 영상에서 프레임 추출 및 라벨 초안 생성 |
| `prepare_video_finetune.py` | 검토된 영상 라벨을 기존 학습 데이터와 결합 |
| `create_video_label_review.py` | 5초 클립의 자동차·사람·카트 초안을 병합하고 검토용 오버레이 생성 |
| `predict_detector.py` | 이미지·영상 탐지와 결과 요약 출력 |
| `document_format_report.py` | 보고서 초안에 Word 서식 적용. `python-docx`와 `보고서/충북대학교_Machine_Learning_HW1 최종본.docx`, `보고서/머신러닝 객체 탐지 과제 보고서.docx`가 필요합니다. 이 입력 문서가 준비되지 않았다면 실행할 수 없습니다. |

## 데이터 출처와 이용

- 자동차·사람 후보 이미지는 [EPFL Multi-view Multi-class Detection](https://www.epfl.ch/labs/cvlab/data/data-multiclass/)에서 가져옵니다. 사용 전 출처의 연구 목적 이용 조건과 인용 안내를 확인하세요.
- 쇼핑 카트 후보 이미지는 [Mendeley Shopping Trolley dataset](https://data.mendeley.com/datasets/vj9g4mn4vv/1), DOI `10.17632/vj9g4mn4vv.1`에서 가져오며 CC BY 4.0 조건이 적용됩니다. 재사용 시 출처와 라이선스를 표시하세요.
- 테스트 영상 출처는 [ViralHog YouTube 영상](https://www.youtube.com/watch?v=SSh_GWLWv0o)입니다. 원본과 편집 클립을 공개하거나 재배포하기 전에 권리와 서비스 이용 조건을 확인하세요.

