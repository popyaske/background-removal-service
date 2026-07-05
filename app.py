# app.py
import gradio as gr
from loadimg import load_img
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
import requests
from io import BytesIO
import os
from typing import Union, Tuple
import gc
import contextlib

# --- Конфигурация ---
# Модель уже лежит здесь
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "model_birefnet_fp16.onnx")

# --- Проверка существования модели ---
if not os.path.exists(LOCAL_MODEL_PATH):
    raise FileNotFoundError(
        f"Модель не найдена по пути: {LOCAL_MODEL_PATH}\n"
        "Убедитесь, что файл model_birefnet_fp16.onnx лежит в папке models/"
    )
print(f"Модель загружена: {LOCAL_MODEL_PATH}")

# --- ONNX Runtime с ограничением памяти ---
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'gpu_mem_limit': 5 * 1024 * 1024 * 1024,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
    }),
    'CPUExecutionProvider'
]

sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
sess_options.enable_cpu_mem_arena = False
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
sess_options.intra_op_num_threads = 1

session = ort.InferenceSession(LOCAL_MODEL_PATH, sess_options, providers=providers)

# Получаем информацию о входе/выходе
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"ONNX сессия создана. Input: {input_name}, Output: {output_name}")

# --- Предобработка ---
transform_image = transforms.Compose([
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# --- Основная функция инференса ---
def process(image: Image.Image) -> Image.Image:
    """
    Применяет BiRefNet (ONNX) для удаления фона.
    Возвращает изображение с альфа-каналом.
    """
    original_size = image.size
    with contextlib.ExitStack() as stack:
        input_tensor = transform_image(image).unsqueeze(0).cpu().numpy()
        outputs = session.run([output_name], {input_name: input_tensor.astype(np.float32)})
        output = outputs[0].squeeze()
        mask = 1 / (1 + np.exp(-output))
        mask_uint8 = (mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8)
        mask_pil = mask_pil.resize(original_size, Image.Resampling.BILINEAR)

        image.putalpha(mask_pil)

        del input_tensor, outputs, output, mask, mask_uint8

        gc.collect()

    return image

# --- Gradio-интерфейсы ---
def fn(image_input: Union[Image.Image, str]) -> Tuple[Image.Image, Image.Image]:
    im = load_img(image_input, output_type="pil")
    im = im.convert("RGB")
    origin = im.copy()
    processed = process(im)
    return origin, processed

def process_file(file_path: str) -> str:
    im = load_img(file_path)
    transparent = process(im)
    output_path = file_path.rsplit(".", 1)[0] + ".png"
    transparent.save(output_path)
    return output_path

# Пример url
url_example = "https://hips.hearstapps.com/hmg-prod/images/gettyimages-1229892983-square.jpg"

# Пример изображения
example_image_path = os.path.join("examples", "butterfly.jpg")
chameleon = None
if os.path.exists(example_image_path):
    chameleon = load_img(example_image_path)

slider1 = gr.ImageSlider(label="Processed Image", type="pil", format="png")
slider2 = gr.ImageSlider(label="Processed Image from URL", type="pil", format="png")
image_upload = gr.Image(label="Upload an image")
image_file_upload = gr.Image(label="Upload an image", type="filepath")
url_input = gr.Textbox(label="Paste an image URL")
output_file = gr.File(label="Output PNG File")

tab1 = gr.Interface(
    fn=fn,
    inputs=image_upload,
    outputs=slider1,
    examples=[[chameleon]] if chameleon else None,
    api_name="image",
    description="Загрузите изображение для удаления фона.",
)
tab2 = gr.Interface(
    fn=fn,
    inputs=url_input,
    outputs=slider2,
    examples=[[url_example]],
    api_name="text",
    description="Вставьте URL изображения.",
)
tab3 = gr.Interface(
    fn=process_file,
    inputs=image_file_upload,
    outputs=output_file,
    examples=[[example_image_path]] if os.path.exists(example_image_path) else None,
    api_name="png",
    description="Загрузите файл и получите PNG с прозрачным фоном.",
)

demo = gr.TabbedInterface(
    [tab1, tab2, tab3],
    ["Image Upload", "URL Input", "File Output"],
    title="Background Removal Tool (BiRefNet ONNX)",
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=False,
    )