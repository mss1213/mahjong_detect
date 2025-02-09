from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('yolov10s.pt')
    model.train(data='data/dataset/dataset.yaml', epochs=100, device='cuda')