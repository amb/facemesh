import mediapipe as mp
# import gradio as gr

# absl-py
# attrs>=19.1.0
# numpy
# protobuf>=3.11.4
# six
# wheel
# gradio
# mediapipe
# torch


# import torch
# Images
# torch.hub.download_url_to_file(
#     "https://artbreeder.b-cdn.net/imgs/c789e54661bfb432c5522a36553f.jpeg", "face1.jpg"
# )
# torch.hub.download_url_to_file(
#     "https://artbreeder.b-cdn.net/imgs/c86622e8cb58d490e35b01cb9996.jpeg", "face2.jpg"
# )

mp_face_mesh = mp.solutions.face_mesh

# Prepare DrawingSpec for drawing the face landmarks later.
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Run MediaPipe Face Mesh.


def inference(image):
    with mp_face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=2, min_detection_confidence=0.5
    ) as face_mesh:
        # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
        # BGR 2 RGB
        results = face_mesh.process(image[..., [2, 1, 0]])
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec,
            )
            return annotated_image


# article = "<p style='text-align: center'><a href='https://arxiv.org/abs/1907.06724'>"
# "Real-time Facial Surface Geometry from Monocular Video on Mobile GPUs</a>"
# " | <a href='https://github.com/google/mediapipe'>Github Repo</a></p>"

# gr.Interface(
#     inference,
#     [gr.inputs.Image(label="Input")],
#     gr.outputs.Image(type="pil", label="Output"),
#     title="Face Mesh",
#     description="Gradio demo for Face Mesh",
#     article=article,
#     examples=[["face1.jpg"], ["face2.jpg"]],
# ).launch()
