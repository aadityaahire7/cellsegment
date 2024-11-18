import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import plotly.graph_objects as go
from flask import Flask, request, render_template, send_file

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        file = request.files["image"]
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        # Process the uploaded image
        image = cv2.imread(image_path, 0)
        _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Generate the three static plots
        dark_gray_background = np.ones_like(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)) * 50
        black_background = np.zeros_like(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        shiny_colored_masks_image = dark_gray_background.copy()
        output_image = dark_gray_background.copy()
        rect_with_cells_image = black_background.copy()

        for contour in contours:
            base_color = [random.randint(100, 200) for _ in range(3)]
            mask = np.zeros_like(image, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)

            for j in range(3):
                shiny_mask = (base_color[j] + dist_transform * 55).clip(0, 255).astype(np.uint8)
                shiny_colored_masks_image[:, :, j] = np.where(mask == 255, shiny_mask, shiny_colored_masks_image[:, :, j])

            cv2.drawContours(rect_with_cells_image, [contour], -1, (255, 255, 255), -1)
            cv2.rectangle(rect_with_cells_image, cv2.boundingRect(contour)[:2],
                          (cv2.boundingRect(contour)[0] + cv2.boundingRect(contour)[2],
                           cv2.boundingRect(contour)[1] + cv2.boundingRect(contour)[3]), (0, 255, 255), 2)
            cv2.drawContours(output_image, [contour], -1, (255, 255, 255), -1)
            cv2.drawContours(output_image, [contour], -1, (0, 255, 0), 2)

        plt.figure(figsize=(24, 8))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(rect_with_cells_image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(shiny_colored_masks_image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        static_plots_path = os.path.join(UPLOAD_FOLDER, "static_plots.png")
        plt.savefig(static_plots_path)
        plt.close()

        # Generate the interactive Plotly plot
        output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        fig = go.Figure()
        fig.add_trace(go.Image(z=output_image_rgb))

        centers = []
        annotations = []

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 0:
                diameter = round(np.sqrt(4 * area / np.pi), 2)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centers.append((cx, cy))
                    annotations.append(f"Cell {i + 1}: {diameter}")
                    fig.add_trace(go.Scatter(
                        x=[cx],
                        y=[cy],
                        text=[f"Cell {i + 1}: {diameter}"],
                        mode='markers',
                        marker=dict(size=8, color='white'),
                        hoverinfo='text'
                    ))

        fig.update_layout(
            title="Interactive Cell Contours with Diameters",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            width=800,
            height=800,
            plot_bgcolor='darkgrey',
            paper_bgcolor='darkgrey'
        )
        interactive_plot_path = os.path.join(UPLOAD_FOLDER, "interactive_plot.html")
        fig.write_html(interactive_plot_path)

        return render_template(
            "index.html",
            image_name=file.filename,
            static_plots="static_plots.png",
            interactive_plot="interactive_plot.html"
        )

    return render_template("index.html", image_name=None)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

if __name__ == "__main__":
    app.run(debug=True)
