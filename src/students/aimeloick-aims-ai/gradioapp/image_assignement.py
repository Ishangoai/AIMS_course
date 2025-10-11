import gradio as gr
from gradioapp.utils.image_assignment_utils import transform_image
css = """
.shadow-box {
    border: 2px solid #ddd;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    padding: 10px;
    border-radius: 8px;
    background-color: white;
}
"""

#gr.Interface(fn=greet, inputs="text", outputs="text", theme=theme)
def transform_images(image, grayscale_or_not,brightness,contrast_factor,degree,threshold):
    return transform_image(image,grayscale_or_not, brightness,contrast_factor,degree,threshold)


#Function to reset image
def reset_image():
   """
   Reset all image transformation controls to default.                                                                       
   """
   return "No Grayscale", 1.0, 1.0, 0
def process_folder(folder):
    if not folder:
        return "No folder uploaded"

    # Get the name of the first file in the folder for demonstration
    file_list = [d.name for d in folder]
    return f"Files uploaded: {', '.join(file_list)}"
with gr.Blocks(css=css) as app:
    gr.HTML("<hr>")
    gr.Markdown(
        '<center> <h2>🖼️ Image Editor — Grayscale, Brightness, Contrast & Rotation</h2></center>'
    )
    gr.HTML("<hr>")
    with gr.Row():
        btn1 = gr.Button("Image")
        btn2 = gr.Button("Folder")
    # autres composants...
    gr.HTML("<hr>")
    with gr.Row():
        gr.Markdown("# Please upload an image.")
    with gr.Row():
        # Input Column 1
        with gr.Column(elem_classes="shadow-box"):
            with gr.Row():
                with gr.Column():
                    image = gr.Image(label ="image", type="pil")
                with gr.Column():
                    output_img = gr.Image(label="Résultat", visible=True)
            grayscale_or_not = gr.Radio(choices=["Grayscale", "No Grayscale"], value="No Grayscale", label="Grayscale or Not")
            brightness = gr.Slider(0.5, 1.5, value= 1, label= "Brightness")
            contrast_factor = gr.Slider(0.5, 1.5, value= 1, label= "Contrast")
            degree = gr.Slider(-180, 180, value= 0, label= "Rotation")
            threshold = gr.Slider(100, 300, value= 80, label= "Rotation")
            #zoom_factor = gr.Slider(0, 10, value= 0, label= "Rotation")
            #transform_btn = gr.Button("Apply transformation")
            
            #result = gr.Textbox(label="Result")
            controls = [image,grayscale_or_not,brightness,contrast_factor,degree,threshold]
            for ctrl in controls:
                ctrl.change(transform_images, inputs=controls, outputs=output_img)
            with gr.Row():
                with gr.Column():
                     reset_btn = gr.Button("Reset")
                     reset_btn.click(fn=reset_image, outputs=[grayscale_or_not,brightness,contrast_factor,degree])
                with gr.Column():
            #download_btn.click(fn=save_image, inputs=output_img, outputs=gr.File(label="Download Edited Image"))
                     download_btn = gr.Button(" Download")

            #transform_btn.click(fn=transform_images,
            #               inputs=[image,grayscale_or_not,brightness,contrast_factor,degree,threshold],
            #              outputs = output_img)
     
        # Input Column 2
        with gr.Column(elem_classes="shadow-box"):
            gr.Markdown("Upload a folder to see its contents")
            # Set file_count to 'directory' to allow folder uploads
            input_folder = gr.File(file_count="directory", label="Upload a folder")
            output_text = gr.Textbox(label="Processed Files")
            submit_button = gr.Button("Process")
            # Link the button's click event to the function
            submit_button.click(fn=process_folder, inputs=input_folder, outputs=output_text)
            
            
    


#https://www.gradio.app/docs/gradio/image
#https://medium.com/@revelyuution/image-manipulation-in-python-using-pillow-62eb68aa8f93