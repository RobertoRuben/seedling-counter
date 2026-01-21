import os
import cv2
import torch
from enum import Enum
from ultralytics import solutions


class Orientation(Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


def main():
    VIDEO_FILENAME = "MK-M5T6LTE113-H4.mp4"
    VIDEO_ORIENTATION = Orientation.VERTICAL
    largo_factor = 0.8

    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    model_path = os.path.join(base_dir, "src", "model", "best.pt")
    video_path = os.path.join(base_dir, VIDEO_FILENAME)
    output_dir = os.path.join(base_dir, "src", "app")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(
        output_dir, f"{os.path.splitext(VIDEO_FILENAME)[0]}_vision.mp4"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cap = cv2.VideoCapture(video_path)
    w, h, fps = (
        int(cap.get(x))
        for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
    )

    if VIDEO_ORIENTATION == Orientation.HORIZONTAL:
        posicion_x = int(w * 0.15)
        margen = int(h * (1 - largo_factor) / 2)
        region_points = [(posicion_x, margen), (posicion_x, h - margen)]
    else:
        posicion_y = int(h * 0.85)
        margen = int(w * (1 - largo_factor) / 2)
        region_points = [(margen, posicion_y), (w - margen, posicion_y)]

    counter = solutions.ObjectCounter(
        model=model_path,
        region=region_points,
        show=False,  
        show_in=False,  
        show_out=False, 
        classes=[0],
        conf=0.35,
        device=device,
    )

    video_writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    window_name = "Seedling Counter"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL) 
    cv2.resizeWindow(window_name, 1280, 720) 

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        annotated_frame = counter(frame).plot_im
        total_count = getattr(counter, "in_count", 0) + getattr(counter, "out_count", 0)

        cv2.putText(
            annotated_frame,
            f"Plantines: {total_count}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3,
        )

        video_writer.write(annotated_frame)
        cv2.imshow(window_name, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
