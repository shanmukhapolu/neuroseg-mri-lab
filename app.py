import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image
import io

st.set_page_config(page_title="NeuroSeg - MRI Segmentation Lab", layout="wide")

# -------------------------------------------------------------------------
# HELPER FUNCTION: Create a synthetic MRI scan if no upload is provided
# -------------------------------------------------------------------------
@st.cache_data
def generate_synthetic_mri():
    """Generates a realistic synthetic 2D brain MRI slice with a tumor."""
    img = np.zeros((256, 256), dtype=np.float32)
    
    # 1. Draw skull boundary
    cv2.ellipse(img, (128, 128), (90, 110), 0, 0, 360, 0.15, -1)
    # Inner brain matter boundary
    cv2.ellipse(img, (128, 128), (85, 105), 0, 0, 360, 0.4, -1)
    
    # 2. Add some simulated internal brain structures (ventricles, gray/white matter)
    cv2.ellipse(img, (110, 120), (15, 35), -10, 0, 360, 0.1, -1) # Left ventricle
    cv2.ellipse(img, (146, 120), (15, 35), 10, 0, 360, 0.1, -1)  # Right ventricle
    
    # 3. Add a synthetic "tumor" (bright, irregular mass in the upper right)
    tumor_center = (165, 95)
    # Base shape
    cv2.circle(img, tumor_center, 18, 0.85, -1)
    # Add random irregularities/lobes to the tumor
    cv2.circle(img, (175, 105), 12, 0.80, -1)
    cv2.circle(img, (155, 90), 10, 0.75, -1)
    
    # Add a dark surrounding edema ring (swelling)
    edema_mask = np.zeros_like(img)
    cv2.circle(edema_mask, tumor_center, 30, 1, -1)
    img[(edema_mask > 0) & (img < 0.6)] = 0.25
    
    # Smooth the synthetic image to look like a real scan and add light noise
    img = gaussian(img, sigma=1.5)
    noise = np.random.normal(0, 0.02, img.shape)
    img = np.clip(img + noise, 0, 1)
    
    return img

# -------------------------------------------------------------------------
# INTERACTION ALGORITHM 1: Region Growing / Flood Fill
# -------------------------------------------------------------------------
def region_growing(img, seed, tolerance):
    """
    Standard region growing algorithm. Starting at a seed point,
    adds neighboring pixels if their intensity is within tolerance of the seed.
    """
    h, w = img.shape
    segmented = np.zeros_like(img, dtype=np.uint8)
    
    seed_y, seed_x = int(seed[0]), int(seed[1])
    if not (0 <= seed_y < h and 0 <= seed_x < w):
        return segmented
        
    queue = [(seed_y, seed_x)]
    segmented[seed_y, seed_x] = 255
    seed_intensity = img[seed_y, seed_x]
    
    # Directions: 4-connectivity (Up, Down, Left, Right)
    dx = [0, 0, 1, -1]
    dy = [1, -1, 0, 0]
    
    while queue:
        cy, cx = queue.pop(0)
        
        for i in range(4):
            ny, nx = cy + dy[i], cx + dx[i]
            if 0 <= ny < h and 0 <= nx < w:
                if segmented[ny, nx] == 0:
                    intensity = img[ny, nx]
                    if abs(intensity - seed_intensity) <= tolerance:
                        segmented[ny, nx] = 255
                        queue.append((ny, nx))
                        
    return segmented

# -------------------------------------------------------------------------
# APP FRONTEND
# -------------------------------------------------------------------------
st.title("🧠 NeuroSeg: Interactive MRI Brain Tumor Segmentation Lab")
st.write("An advanced playground showcasing **classic medical computer vision** without expensive APIs.")

col_sidebar, col_main = st.columns([1, 3])

# Load or generate image FIRST to know dimensions for coordinate initialization
uploaded_file = st.file_uploader("Upload an MRI Slice (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    color_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    img = gray_img.astype(np.float32) / 255.0
else:
    img = generate_synthetic_mri()

img_height, img_width = img.shape

# Keep track of file changes to reset clicks
current_file_id = uploaded_file.name if uploaded_file else "synthetic"
if "prev_file" not in st.session_state or st.session_state.prev_file != current_file_id:
    st.session_state.coord_x = int(img_width * 0.64)  # Default on tumor center
    st.session_state.coord_y = int(img_height * 0.37)
    st.session_state.prev_file = current_file_id

with col_sidebar:
    st.header("1. Controls")
    algo = st.radio("Segmentation Method", ["Active Contour Model (Snakes)", "Region Growing"])
    
    st.markdown("---")
    st.subheader("Fine-Tuning Parameters")
    
    if algo == "Active Contour Model (Snakes)":
        st.info("💡 **Click on the scan image** to reposition the circle center, or use the sliders below.")
        
        # Pull coordinates directly from coordinate state, pushing changes back to session_state
        init_x = st.slider("Circle Center X", 0, img_width, int(st.session_state.coord_x), key="slider_x")
        init_y = st.slider("Circle Center Y", 0, img_height, int(st.session_state.coord_y), key="slider_y")
        st.session_state.coord_x = init_x
        st.session_state.coord_y = init_y
        
        max_radius = int(min(img_width, img_height) / 2)
        radius = st.slider("Initial Circle Radius", 5, max_radius, int(min(img_width, img_height) * 0.1))
        alpha = st.slider("Elasticity (Alpha)", 0.01, 0.50, 0.05, help="Controls how easily the snake stretches.")
        beta = st.slider("Stiffness (Beta)", 0.1, 5.0, 1.0, help="Controls how smooth/rigid the snake stays.")
        
    elif algo == "Region Growing":
        st.info("💡 **Click on the scan image** to place your seed point inside the tumor structure.")
        
        seed_x = st.slider("Seed Coordinate X", 0, img_width, int(st.session_state.coord_x), key="slider_x")
        seed_y = st.slider("Seed Coordinate Y", 0, img_height, int(st.session_state.coord_y), key="slider_y")
        st.session_state.coord_x = seed_x
        st.session_state.coord_y = seed_y
        
        tolerance = st.slider("Intensity Tolerance", 0.01, 0.30, 0.08, help="Max difference in brightness allowed to group pixels.")

# Main Screen layout
with col_main:
    st.subheader("Interactive Workspace")
    st.write("Click anywhere on the left scan below to dynamically shift target coordinates in real time.")
    
    # -------------------------------------------------------------------------
    # CANVAS GENERATION FOR INTERACTIVE CLICKS
    # -------------------------------------------------------------------------
    fig_click, ax_click = plt.subplots(figsize=(5, 5))
    ax_click.imshow(img, cmap='gray')
    ax_click.axis('off')
    
    if algo == "Active Contour Model (Snakes)":
        s_test = np.linspace(0, 2 * np.pi, 200)
        r_test = st.session_state.coord_y + radius * np.sin(s_test)
        c_test = st.session_state.coord_x + radius * np.cos(s_test)
        ax_click.plot(c_test, r_test, '--r', lw=2, label="Placement Ring")
    else:
        ax_click.plot(st.session_state.coord_x, st.session_state.coord_y, 'ro', markersize=8, label="Seed Point")
    
    buf = io.BytesIO()
    fig_click.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig_click)
    
    # We use a unique key combined with the coordinates so that canvas state registers clicks cleanly
    canvas_key = f"canvas_click_{st.session_state.coord_x}_{st.session_state.coord_y}_{algo}"
    clicked_coords = streamlit_image_coordinates(Image.open(buf), width=450, key=canvas_key)
    
    # Capture click feedback and convert back to native matrix resolution space
    if clicked_coords is not None:
        canvas_w = clicked_coords["width"]
        canvas_h = clicked_coords["height"]
        
        # Precise coordinate interpolation mapping
        raw_x = int((clicked_coords["x"] / canvas_w) * img_width)
        raw_y = int((clicked_coords["y"] / canvas_h) * img_height)
        
        clamped_x = max(0, min(raw_x, img_width - 1))
        clamped_y = max(0, min(raw_y, img_height - 1))
        
        if clamped_x != st.session_state.coord_x or clamped_y != st.session_state.coord_y:
            st.session_state.coord_x = clamped_x
            st.session_state.coord_y = clamped_y
            st.rerun()

    # -------------------------------------------------------------------------
    # SEGMENTATION PROCESSING PIPELINE
    # -------------------------------------------------------------------------
    st.markdown("### Processed Results")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Input MRI Scan")
    axes[0].axis('off')
    
    if algo == "Active Contour Model (Snakes)":
        s = np.linspace(0, 2*np.pi, 400)
        r = st.session_state.coord_y + radius * np.sin(s)
        c = st.session_state.coord_x + radius * np.cos(s)
        init_boundary = np.array([r, c]).T
        
        axes[0].plot(init_boundary[:, 1], init_boundary[:, 0], '--r', lw=2, label="Initial Boundary")
        axes[0].legend(loc="upper left")
        
        # Smooth and calculate Active Contour (Snakes)
        smoothed = gaussian(img, 2)
        snake = active_contour(smoothed, init_boundary, alpha=alpha, beta=beta, gamma=0.001)
        
        axes[1].imshow(img, cmap='gray')
        axes[1].plot(snake[:, 1], snake[:, 0], '-b', lw=3, label="Segmented Outline")
        axes[1].set_title("Segmented Output (Active Contour)")
        axes[1].axis('off')
        axes[1].legend(loc="upper left")
        
        st.success(f"📈 **Biomedical Metric:** Segmented tumor boundary length is approximately **{len(snake):.1f} pixels**.")

    elif algo == "Region Growing":
        axes[0].plot(st.session_state.coord_x, st.session_state.coord_y, 'ro', markersize=6, label="User Seed Point")
        axes[0].legend(loc="upper left")
        
        with st.spinner("Processing pixel matrices..."):
            mask = region_growing(img, (st.session_state.coord_y, st.session_state.coord_x), tolerance)
        
        overlay = np.stack([img, img, img], axis=-1)
        overlay[mask == 255] = [0.8, 0.1, 0.1]  # Highlight mask overlay in light red
        
        axes[1].imshow(overlay)
        axes[1].set_title("Segmented Output (Region Growing)")
        axes[1].axis('off')
        
        tumor_area_pixels = np.sum(mask == 255)
        st.success(f"📈 **Biomedical Metric:** Segmented tumor mass occupies **{tumor_area_pixels} total pixels**.")

    st.pyplot(fig)

    # Architectural Overview
    st.markdown("---")
    st.markdown("### Under the Hood: The Mathematical Algorithms")
    
    col_math1, col_math2 = st.columns(2)
    with col_math1:
        st.markdown("""
        #### How Active Contours (Snakes) Work
        This algorithm minimizes an energy function composed of internal and external forces:
        $$E_{snake} = \int_{0}^{1} (E_{internal}(v(s)) + E_{external}(v(s))) ds$$
        *   **Internal Energy** ($E_{internal}$): Prevents the contour from stretching out or bending too jaggedly (controlled by your **Alpha** and **Beta** sliders).
        *   **External Energy** ($E_{external}$): Attracts the contour to the brightest contrast points (edges) in the medical scan.
        """)
        
    with col_math2:
        st.markdown("""
        #### How Region Growing Works
        A classical seed-fill algorithm based on spatial **4-connectivity**. 
        1. It starts at your coordinate $(X, Y)$ and samples the target gray value.
        2. It checks adjacent pixels in a cross shape.
        3. If $|I_{neighbor} - I_{seed}| \\le \\text{Tolerance}$, it engulfs that pixel and repeats.
        """)
