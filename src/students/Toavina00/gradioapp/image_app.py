import gradio as gr
from gradioapp.image_editing.basic import transform_image
from gradioapp.image_editing.extra import (
    canny_detector,
    color_dithering,
    gray_dithering,
    histogram_equalizer,
    laplace_filter,
    sobel_h_filter,
    sobel_magnitude,
    sobel_v_filter,
)

with gr.Blocks() as app:
    gr.Markdown("# Image Editor")

    with gr.Row():
        with gr.Column():
            src = gr.Image()

        with gr.Column():
            dst = gr.Image()

    with gr.Tab("Basic"):
        with gr.Row():
            with gr.Column():
                brightness = gr.Slider(0.5, 1.5, value=1.0, label="Brightness")
                contrast = gr.Slider(0.5, 1.5, value=1.0, label="Contrast")
                rotate = gr.Slider(-180.0, 180.0, value=0.0, label="Rotate")
                blur = gr.Slider(0, 5, value=0, step=1, label="Blur")

            with gr.Column():
                with gr.Group():
                    grayscale = gr.Checkbox(False, label="Grayscale")
                    invert = gr.Checkbox(False, label="Invert Color")
                    flip_h = gr.Checkbox(False, label="Flip Horizontal")
                    flip_v = gr.Checkbox(False, label="Flip Vertical")

    with gr.Tab("Extra"):
        with gr.Row():
            with gr.Column():
                c_dither_btn = gr.Button("Dithering (Color)")
                g_dither_btn = gr.Button("Dithering (Grayscale)")
                laplace_btn = gr.Button("Laplace")

            with gr.Column():
                sobel_h_btn = gr.Button("Sobel Horizontal")
                sobel_v_btn = gr.Button("Sobel Vertical")
                sobel_m_btn = gr.Button("Sobel Magnitude")

            with gr.Column():
                canny_btn = gr.Button("Canny Detector")
                hist_eq_btn = gr.Button("Histogram Equalizer")

    reset_btn = gr.Button("Reset")

    transforms = [
        grayscale,
        invert,
        flip_h,
        flip_v,
        brightness,
        contrast,
        rotate,
        blur,
    ]  # Keep a list of transformations to facilitate event handling
    initial_value = [trans.value for trans in transforms]  # Keep track of the initial value for reset event

    gr.on(fn=transform_image, inputs=[src, *transforms], outputs=dst)
    gr.on(c_dither_btn.click, fn=color_dithering, inputs=src, outputs=dst)
    gr.on(g_dither_btn.click, fn=gray_dithering, inputs=src, outputs=dst)
    gr.on(sobel_h_btn.click, fn=sobel_h_filter, inputs=src, outputs=dst)
    gr.on(sobel_v_btn.click, fn=sobel_v_filter, inputs=src, outputs=dst)
    gr.on(sobel_m_btn.click, fn=sobel_magnitude, inputs=src, outputs=dst)
    gr.on(laplace_btn.click, fn=laplace_filter, inputs=src, outputs=dst)
    gr.on(canny_btn.click, fn=canny_detector, inputs=src, outputs=dst)
    gr.on(hist_eq_btn.click, fn=histogram_equalizer, inputs=src, outputs=dst)
    gr.on(reset_btn.click, inputs=src, fn=lambda src: [src, *initial_value], outputs=[dst, *transforms])
