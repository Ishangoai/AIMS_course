import gradio as gr
import os
from PIL import Image
from gradioapp.utils.image_assignment_utils import transform_image
import tempfile
import zipfile

def transform_folder(folder, grayscale_or_not, brightness, contrast_factor, degree, threshold):
    if not folder:
        return "No folder uploaded", None

    output_dir = tempfile.mkdtemp()
    processed_files = []

    for file_data in folder:
        file_path = getattr(file_data, "name", None)
        if not file_path or not os.path.exists(file_path):
            print(f"⚠️ Fichier introuvable ou non accessible : {file_data}")
            continue

        if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                with Image.open(file_path) as img:
                    img_transformed = transform_image(img, grayscale_or_not, brightness, contrast_factor, degree, threshold)

                    output_path = os.path.join(output_dir, os.path.basename(file_path))
                    img_transformed.save(output_path)
                    processed_files.append(output_path)
            except Exception as e:
                print(f"Erreur lors du traitement de {file_path}: {e}")

    if not processed_files:
        return "Aucune image valide trouvée dans le dossier.", None

    # Crée un fichier ZIP contenant toutes les images transformées
    zip_path = os.path.join(output_dir, "transformed_images.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for f in processed_files:
            zipf.write(f, os.path.basename(f))

    return f"{len(processed_files)} images transformées.", zip_path


with gr.Blocks() as app:
    gr.Markdown("## 📁 Batch Image Transformer")
    gr.Markdown("Upload un dossier d’images et applique des transformations à toutes les images.")

    input_folder = gr.File(file_count="directory", label="📂 Dossier d’images")
    grayscale_or_not = gr.Radio(["Grayscale", "No Grayscale"], value="No Grayscale", label="Grayscale or Not")
    brightness = gr.Slider(0.5, 1.5, value=1, label="Brightness")
    contrast_factor = gr.Slider(0.5, 1.5, value=1, label="Contrast")
    degree = gr.Slider(-180, 180, value=0, label="Rotation")
    threshold = gr.Slider(100, 300, value=80, label="Threshold")

    submit = gr.Button("🔄 Appliquer la transformation")
    output_text = gr.Textbox(label="Résultat")
    download_zip = gr.File(label="Télécharger le dossier compressé")

    submit.click(
        fn=transform_folder,
        inputs=[input_folder, grayscale_or_not, brightness, contrast_factor, degree, threshold],
        outputs=[output_text, download_zip]
    )

app.launch()
