%%writefile app.py
import streamlit as st
import numpy as np
from skimage.filters import threshold_li
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
import os

# -----------------------------
# Optional PDF support
# -----------------------------
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    PDF_ENABLED = True
except ModuleNotFoundError:
    PDF_ENABLED = False

st.set_page_config(page_title="Concrete Crack Detection – IS 456", layout="wide")

st.title("🧱 Concrete Crack Width Detection")
st.caption("IS 456:2000 compliant | Image-based crack analysis")

# -----------------------------
# Exposure conditions (IS 456)
# -----------------------------
exposure_condition_list = [
    "Concrete surfaces protected against weather or aggressive conditions (except coastal area)",
    "Concrete surfaces sheltered from severe rain or freezing whilst wet",
    "Concrete exposed to condensation and rain",
    "Concrete continuously under water",
    "Concrete in contact with or buried under non-aggressive soil / ground water",
    "Concrete surfaces exposed to severe rain, alternate wetting and drying or severe condensation",
    "Concrete completely immersed in sea water",
    "Concrete exposed to coastal environment",
    "Concrete surfaces exposed to sea water spray, corrosive fumes or severe freezing whilst wet",
    "Concrete in contact with or buried under aggressive sub-soil / ground water",
    "Surface of members in tidal zone",
    "Members in direct contact with liquid or solid aggressive chemicals",
]

exposure_condition = st.selectbox(
    "Select Exposure Condition (IS 456:2000 – Table 3)",
    exposure_condition_list,
)

# -----------------------------
# IS 456 limits
# -----------------------------
def get_is_limit(exposure: str):
    if exposure in exposure_condition_list[:5]:
        return 0.30, "Mild / Moderate"
    if exposure in exposure_condition_list[5:8]:
        return 0.20, "Severe"
    return 0.10, "Very Severe / Extreme"

# -----------------------------
# Repair recommendations
# -----------------------------
def recommend_repair(width_mm: float) -> dict:
    if width_mm <= 0.10:
        return {
            "method": "Monitoring only",
            "details": ["Hairline crack", "Periodic inspection recommended"],
        }
    if width_mm <= 0.30:
        return {
            "method": "Surface crack sealing",
            "details": ["Polymer-modified or elastomeric sealant"],
        }
    if width_mm <= 0.50:
        return {
            "method": "Epoxy injection",
            "details": ["Structural crack", "Crack must be dormant and dry"],
        }
    if width_mm <= 1.00:
        return {
            "method": "Cementitious / microfine grouting",
            "details": ["Wide or wet crack", "Improves durability"],
        }
    return {
        "method": "Structural strengthening",
        "details": ["Jacketing / FRP wrapping / steel plate bonding"],
    }

# -----------------------------
# PDF generation
# -----------------------------
def generate_pdf(data: dict, images: dict) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(tmp.name, pagesize=A4, leftMargin=20, rightMargin=20)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>Concrete Crack Assessment Report</b>", styles["Title"]))
    elements.append(Spacer(1, 12))

    table_data = [
        ["Measured crack width (mm)", f"{data['width']:.3f}"],
        ["Exposure class", data["exposure"]],
        ["IS 456 permissible width (mm)", f"{data['limit']:.2f}"],
        ["Assessment", data["status"]],
        ["Recommended repair", data["repair"]],
    ]

    elements.append(Table(table_data, colWidths=[70 * mm, 80 * mm]))
    elements.append(Spacer(1, 15))

    for title, path in images.items():
        elements.append(Paragraph(f"<b>{title}</b>", styles["Heading3"]))
        elements.append(RLImage(path, width=150 * mm, height=90 * mm))
        elements.append(Spacer(1, 10))

    doc.build(elements)
    return tmp.name

# -----------------------------
# Image upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload concrete crack image",
    type=["jpg", "jpeg", "png"],
)

# -----------------------------
# Processing
# -----------------------------
if uploaded_file:
    img = np.array(Image.open(uploaded_file).convert("L"))

    thresh = threshold_li(img)
    binary_bool = binary_fill_holes(img < thresh)
    binary_img = (binary_bool * 255).astype(np.uint8)

    distance = distance_transform_edt(binary_bool)

    if np.max(distance) == 0:
        st.error("No crack detected. Upload a clearer image.")
        st.stop()

    max_width_pixels = 2 * np.max(distance)

    pixel_to_mm = st.number_input(
        "Pixel to mm scale (e.g. 0.02)",
        value=0.02,
        min_value=0.0001,
    )

    crack_width_mm = max_width_pixels * pixel_to_mm
    limit, exposure_class = get_is_limit(exposure_condition)
    status = "ACCEPTABLE" if crack_width_mm <= limit else "NOT ACCEPTABLE"
    repair = recommend_repair(crack_width_mm)

    # Save temp images
    tmpdir = tempfile.mkdtemp()
    orig_path = os.path.join(tmpdir, "original.png")
    bin_path = os.path.join(tmpdir, "binary.png")
    dist_path = os.path.join(tmpdir, "distance.png")

    Image.fromarray(img).save(orig_path)
    Image.fromarray(binary_img).save(bin_path)

    fig, ax = plt.subplots()
    ax.imshow(distance, cmap="jet")
    ax.axis("off")
    fig.savefig(dist_path, bbox_inches="tight")
    plt.close(fig)

    # Display
    col1, col2, col3 = st.columns(3)
    col1.image(img, "Original Image", use_column_width=True)
    col2.image(binary_img, "Binary Crack Mask", use_column_width=True)
    col3.image(dist_path, "Distance Map", use_column_width=True)

    st.markdown("## 📊 Crack Assessment Result")
    st.write(f"**Crack width:** {crack_width_mm:.3f} mm")
    st.write(f"**Exposure class:** {exposure_class}")
    st.write(f"**IS 456 limit:** {limit:.2f} mm")
    st.write(f"**Assessment:** {status}")

    st.markdown("## 🛠️ Repair Recommendation")
    st.write(f"**Method:** {repair['method']}")
    for d in repair["details"]:
        st.write(f"- {d}")

    if PDF_ENABLED:
        if st.button("📄 Generate PDF Report"):
            pdf_path = generate_pdf(
                {
                    "width": crack_width_mm,
                    "exposure": exposure_class,
                    "limit": limit,
                    "status": status,
                    "repair": repair["method"],
                },
                {
                    "Original Image": orig_path,
                    "Binary Crack Mask": bin_path,
                    "Distance Map": dist_path,
                },
            )

            with open(pdf_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Report",
                    f,
                    file_name="crack_assessment_report.pdf",
                    mime="application/pdf",
                )
    else:
        st.warning("PDF export disabled. Install `reportlab` via requirements.txt.")
