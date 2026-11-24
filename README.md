# Parking CCTV Object Tracking

주차장 CCTV 영상에서 `car`, `person`, `shopping_cart` 세 클래스를 탐지하기 위한 과제 자료입니다. 현재 저장소에는 학습 데이터나 영상 파일을 포함하지 않습니다. 데이터는 각 출처의 이용 조건에 따라 별도로 내려받고, 테스트 영상은 권리 확인 후 로컬 `video/` 폴더에 준비해야 합니다.

## 빠른 시작

프로젝트 루트(`과제/`)에서 실행합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r scripts/requirements.txt
python scripts/download_training_images.py --carts 200 --seed 42
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

Python 3.10 이상과 Pillow가 필요합니다.

```bash
python scripts/download_training_images.py --carts 200 --seed 42
python scripts/verify_training_images.py
```

수집 결과는 `training_candidates/`에 저장됩니다. 다운로드 코드는 파일의 SHA-256을 확인하고 EPFL 상자 좌표를 YOLO 형식으로 변환합니다.

## 검토 및 학습용 폴더 만들기

1. `training_candidates/review/contact_sheets/`의 시트에서 초록 상자가 객체를 제대로 감싸는지, 실제 자동차·사람·쇼핑 카트인지, 프로젝트 장면에 쓸 만한지 확인합니다.
2. `training_candidates/review/review.csv`에서 사용할 행의 `approved` 칸만 `yes`로 바꿉니다. 부적합은 `no`, 판단이 어려운 항목은 빈칸으로 둡니다.
3. 승인한 이미지만 학습 폴더로 복사하고, EPFL의 동기화된 카메라 프레임은 프레임 ID 단위로 train/validation을 나눕니다.

```bash
python scripts/verify_training_images.py --promote
```

결과는 `training_candidates/verified_yolo_dataset/`에 생성되며 `data.yaml`은 클래스 순서를 `[car, person, shopping_cart]`로 지정합니다. 자동 검사는 파일 손상, 라벨 형식, 상자 좌표를 확인하고, 이미지 의미 검토는 연락처 시트를 기준으로 승인한 행만 반영합니다.

## 폴더 주의

- 이전에 잘못 추출한 CCTV 프레임과 Open Images 후보 폴더는 정리했습니다. 위 스크립트로 새 후보 데이터를 만들 수 있습니다.
- 출처별 사용 조건을 지킵니다. 이 저장소에서는 외부 이미지·라벨 데이터와 동영상 파일을 재배포하지 않습니다.
- EPFL: [Multi-view Multi-class Detection](https://www.epfl.ch/labs/cvlab/data/data-multiclass/). 연구 목적 사용 안내와 인용 조건을 확인하세요.
- Mendeley: [Shopping Trolley dataset](https://data.mendeley.com/datasets/vj9g4mn4vv/1), DOI `10.17632/vj9g4mn4vv.1`, CC BY 4.0. 재사용 시 출처와 라이선스를 표시하세요.
