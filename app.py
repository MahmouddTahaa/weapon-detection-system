import gradio as gr
import cv2
import numpy as np
import os
import tempfile
from weapon_detection_model import detect_weapons

webcam_active = False
cap = None

def webcam_stream():
    global webcam_active, cap
    
    if not webcam_active:
        yield None, "Webcam not active"
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        webcam_active = False
        yield None, "Failed to open webcam"
        return
    
    while webcam_active:
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame, status = detect_weapons(frame)
        yield processed_frame, status
    
    cap.release()
    cap = None
    yield None, "Webcam stopped"

def toggle_webcam():
    """Toggle webcam on/off"""
    global webcam_active
    webcam_active = not webcam_active
    return gr.update(value="Stop Webcam" if webcam_active else "Start Webcam", variant="stop" if webcam_active else "primary")

def process_video(video_path):
    if not video_path:
        return None, "No video selected"
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, "Could not open video file"
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    weapon_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        processed_frame, status = detect_weapons(frame)
        
        if "detected" in status:
            weapon_frames += 1
            
        processed_frame_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR)
        if processed_frame_bgr is not None:
            out.write(processed_frame_bgr)
        else:
            print(f"Warning: Frame {frame_count} is None, skipping write")
    
    cap.release()
    out.release()
    
    summary = f"Processed {frame_count} frames. Weapons detected in {weapon_frames} frames."
    if os.path.exists(output_path):
        return output_path, summary
    else:
        return None, "Failed to generate output video"

with gr.Blocks(title="Weapon Detection System") as demo:
    gr.Markdown("# Weapon Detection System")
    gr.Markdown("Using YOLO v3 for real-time weapon detection")
    
    with gr.Tabs():
        with gr.Tab("Webcam Detection"):
            detection_output = gr.Image(label="Detection Output")
            status_text = gr.Textbox(label="Detection Status", value="Webcam not active")
            
            webcam_button = gr.Button("Start Webcam")
            
            webcam_button.click(
                fn=toggle_webcam,
                outputs=[webcam_button]
            ).then(
                fn=webcam_stream,
                inputs=[],
                outputs=[detection_output, status_text]
            )
        
        with gr.Tab("Video Upload"):
            video_input = gr.Video(label="Upload Video")
            video_output = gr.Video(label="Processed Video")
            video_status = gr.Textbox(label="Processing Summary")
            
            process_btn = gr.Button("Process Video")
            process_btn.click(
                process_video,
                inputs=[video_input],
                outputs=[video_output, video_status]
            )
    
    gr.Markdown("## Instructions")
    gr.Markdown("""
    - **Webcam Detection**: Click 'Start Webcam' to begin real-time weapon detection from your webcam. Click 'Stop Webcam' to turn it off.
    - **Video Upload**: Upload a video file and click 'Process Video' to detect weapons in the video.
    """)

if __name__ == "__main__":
    demo.launch(share=False)
