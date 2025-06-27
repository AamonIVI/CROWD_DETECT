import cv2
from ultralytics import YOLO


def human_detection(input_path, output_path):
    model = YOLO('yolov8n.pt')

    capture = cv2.VideoCapture(input_path)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        annotated_frame = draw_frame(frame, results[0])

        out.write(annotated_frame)

    capture.release()
    out.release()

def draw_frame(frame, results):
    for box in results.boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        label = results.names[cls_id]

        if label != 'person':
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(frame, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

if __name__ == "__main__":
    input_path = "input/crowd.mp4"
    output_path = "output/crowd_detected.mp4"
    human_detection(input_path, output_path)
