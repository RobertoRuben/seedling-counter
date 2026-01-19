import os

import cv2
import torch
from ultralytics import solutions


def main():
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    model_path = os.path.join(base_dir, "src", "model", "best.pt")
    video_path = os.path.join(base_dir, "20251229_064233.mp4")

    output_dir = os.path.join(base_dir, "src", "app")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "resultado_conteo_limpio.mp4")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Iniciando Seedling Counter en: {device}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video en {video_path}")
        return

    w, h, fps = (
        int(cap.get(x))
        for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
    )

    margin = 30
    if w > h:
        region_points = [
            (w // 2 - margin, 0),
            (w // 2 + margin, 0),
            (w // 2 + margin, h),
            (w // 2 - margin, h),
        ]
    else:
        region_points = [
            (0, h // 2 - margin),
            (w, h // 2 - margin),
            (w, h // 2 + margin),
            (0, h // 2 + margin),
        ]

    counter = solutions.ObjectCounter(
        model=model_path,
        region=region_points,
        show=False, 
        show_labels=False,
        show_conf=False,
        device=device,
        conf=0.35,
        iou=0.45,
        classes=[0],  
        line_width=2,
    )

    video_writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    window_name = "Seedling Counter - Procesando"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_count = 0
    skip_frames = 2 

    print("Procesando frames... Presiona 'q' para finalizar.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Procesamiento de video completado.")
            break

        frame_count += 1

        if frame_count % skip_frames != 0:
            continue

        results = counter(frame)
        annotated_frame = results.plot_im

        video_writer.write(annotated_frame)

        cv2.imshow(window_name, annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    print("-" * 40)
    print(f"Video guardado en {output_path}")
    print("-" * 40)


if __name__ == "__main__":
    main()
