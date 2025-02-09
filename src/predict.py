from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('runs/detect/train2/weights/best.pt')
    results = model.predict(source='data/dataset/valid/images/img_526.jpg', save_conf=True)
    print(results[0])