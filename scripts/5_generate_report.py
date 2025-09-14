import os
import os.path as op
import yaml
import base64
from pathlib import Path

def load_config(config_path="config/config.yaml"):
    """Loads the configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def image_to_base64(img_path):
    """Converts an image file to a base64 string for embedding in HTML."""
    try:
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except FileNotFoundError:
        return None

def generate_report():
    """Generates a single HTML report with all analysis figures."""
    print("--- Generating HTML Report ---")
    
    # 1. Load configuration
    config = load_config()
    subjects = config['subjects']
    paths = config['paths']
    
    # Define figure directories
    grand_avg_dir = op.join(paths['results_dir'], 'readiness_potential')
    subject_erp_dir = op.join('figures', 'subject_erp_plots')
    report_fname = "analysis_report.html"

    # 2. Start building HTML content
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Readiness Potential Analysis Report</title>
        <style>
            body { font-family: sans-serif; margin: 2em; }
            h1, h2, h3 { text-align: center; color: #333; }
            .subject-section { border-top: 2px solid #ccc; padding-top: 20px; margin-top: 40px; }
            .figure-container { text-align: center; margin-bottom: 30px; }
            img { max-width: 90%; height: auto; border: 1px solid #ddd; }
            p.caption { font-style: italic; color: #666; }
        </style>
    </head>
    <body>
        <h1>Readiness Potential Analysis Report</h1>
    """

    # 3. Add Grand Average Section
    html += "<h2>Grand Average Results</h2>"
    ga_plots = [
        "grand_average_bp_waveforms_Frontal.png",
        "grand_average_bp_waveforms_Central.png",
        "grand_average_bp_waveforms_Parietal.png",
        "grand_average_bp_topography_diff.png",
        "grand_average_bp_joint_plot_diff.png",
        "grand_average_bp_stats_Cz.png"
    ]
    for plot_name in ga_plots:
        b64_img = image_to_base64(op.join(grand_avg_dir, plot_name))
        if b64_img:
            html += f'<div class="figure-container"><p class="caption">{plot_name}</p><img src="data:image/png;base64,{b64_img}"></div>'

    # 4. Add Individual Subject Section
    html += "<h2>Individual Subject Results</h2>"
    for subject in subjects:
        html += f'<div class="subject-section"><h3>Subject: {subject}</h3>'
        
        subject_plots = [
            f"{subject}_combined_butterfly_plot.png",
            f"{subject}_joint_plot_diff.png",
            f"{subject}_Fz_Cz_Pz_waveforms.png"
        ]
        
        for plot_name in subject_plots:
            b64_img = image_to_base64(op.join(subject_erp_dir, plot_name))
            if b64_img:
                html += f'<div class="figure-container"><p class="caption">{plot_name}</p><img src="data:image/png;base64,{b64_img}"></div>'
        
        html += '</div>'

    # 5. Finalize and save HTML
    html += "</body></html>"
    with open(report_fname, 'w') as f:
        f.write(html)
        
    print(f"\n--- Report successfully generated: {report_fname} ---")

if __name__ == "__main__":
    generate_report()