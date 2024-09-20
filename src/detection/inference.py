from ultralytics import YOLO

model = YOLO("yolov8m.pt")

results = model(["D:/Work/My Projects/LitterDetection/Litter-Voxel51-Hackathon/UAVVasteDataset/images/batch_01_frame_15.jpg"])

for result in results:
    boxes = result.boxes 
    probs = result.probs
    result.show()
