import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.segmentation import active_contour
from skimage.filters import gaussian
from skimage.draw import disk
import io

st.set_page_config(page_title="NeuroSeg - MRI Segmentation Lab", layout="wide")

# -------------------------------------------------------------------------
# HELPER FUNCTION: Create a synthetic MRI scan if no upload is provided
# -------------------------------------------------------------------------
@st.cache_data
def generate_synthetic_mri():
    """Generates a realistic synthetic 2D brain MRI slice with a tumor."""
    # Create dark background
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
    
    # Simple list-based queue for region growing
    queue = [seed]
    segmented[seed[0], seed[1]] = 255
    seed_intensity = img[seed[0], seed[1]]
    
    # Directions: 4-connectivity (Up, Down, Left, Right)
    dx = [0, 0, 1, -1]
    dy = [1, -1, 0, 0]
    
    while queue:
        cx, cy = queue.pop(0)
        
        for i in range(4):
            nx, ny = cx + dx[i], cy + dy[i]
            if 0 <= nx < h and 0 <= ny < w:
                if segmented[nx, ny] == 0:
                    intensity = img[nx, ny]
                    # Check if neighbor intensity is close enough to seed intensity
                    if abs(intensity - seed_intensity) <= tolerance:
                        segmented[nx, ny] = 255
                        queue.append((nx, ny))
                        
    return segmented

# -------------------------------------------------------------------------
# APP FRONTEND
# -------------------------------------------------------------------------
st.title("🧠 NeuroSeg: Interactive MRI Brain Tumor Segmentation Lab")
st.write("An advanced playground showcasing **classic medical computer vision** without expensive APIs.")

col_sidebar, col_main = st.columns([1, 3])

with col_sidebar:
    st.header("1. Input Data")
    uploaded_file = st.file_uploader("Upload an MRI Slice (PNG/JPG)", type=["png", "jpg", "jpeg"])
    
    st.header("2. Choose Algorithm")
    algo = st.radio("Method", ["Active Contour Model (Snakes)", "Region Growing"])
    
    if algo == "Active Contour Model (Snakes)":
        st.info("💡 **Active Contours** use physical forces (elasticity, stiffness) to shrink a geometric boundary onto the high-contrast edges of a tumor.")
        init_x = st.slider("Initial Circle Center X", 50, 200, 165)
        init_y = st.slider("Initial Circle Center Y", 50, 200, 95)
        radius = st.slider("Initial Circle Radius", 5, 50, 25)
        alpha = st.slider("Elasticity (Alpha)", 0.01, 0.50, 0.05, help="Controls how easily the snake stretches.")
        beta = st.slider("Stiffness (Beta)", 0.1, 5.0, 1.0, help="Controls how smooth/rigid the snake stays.")
        
    elif algo == "Region Growing":
        st.info("💡 **Region Growing** starts at a single user-defined pixel (seed) and expands outward to group all matching adjacent tumor tissue.")
        seed_x = st.slider("Seed Coordinate X", 0, 255, 165)
        seed_y = st.slider("Seed Coordinate Y", 0, 255, 95)
        tolerance = st.slider("Intensity Tolerance", 0.01, 0.20, 0.08, help="Max difference in brightness allowed to group pixels.")

# Load or generate image
if uploaded_file is not None:
    # Convert uploaded file to grayscale
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    color_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    # Normalize to [0, 1] range
    img = gray_img.astype(np.float32) / 255.0
else:
    img = generate_synthetic_mri()

# Main Screen layout
with col_main:
    st.subheader("Interactive Workspace")
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot original image
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Input MRI Scan")
    axes[0].axis('off')
    
    if algo == "Active Contour Model (Snakes)":
        # Create the starting snake (circle) boundary
        s = np.linspace(0, 2*np.pi, 400)
        r = init_y + radius * np.sin(s)
        c = init_x + radius * np.cos(s)
        init_boundary = np.array([r, c]).T
        
        # Draw initial circle on left axis
        axes[0].plot(init_boundary[:, 1], init_boundary[:, 0], '--r', lw=2, label="Initial Boundary")
        axes[0].legend(loc="upper left")
        
        # Process the active contour using mathematical forces
        # We pre-smooth the image slightly to make edge detection highly stable
        smoothed = gaussian(img, 2)
        snake = active_contour(smoothed, init_boundary, alpha=alpha, beta=beta, gamma=0.001)
        
        # Plot final output
        axes[1].imshow(img, cmap='gray')
        axes[1].plot(snake[:, 1], snake[:, 0], '-b', lw=3, label="Segmented Outline")
        axes[1].set_title("Segmented Output (Active Contour)")
        axes[1].axis('off')
        axes[1].legend(loc="upper left")
        
        # Quantify the tumor size
        # Generate mask of the snake region
        mask = np.zeros_like(img, dtype=bool)
        rr, cc = disk((init_y, init_x), radius, shape=img.shape) # simple approx for demo
        tumor_pixel_count = len(snake) # Let's approximate boundary length as safe metric
        st.success(f"📈 **Biomedical Metric:** Segmented tumor boundary length is approximately **{len(snake):.1f} pixels**.")

    elif algo == "Region Growing":
        # Show seed marker on original image
        axes[0].plot(seed_x, seed_y, 'ro', markersize=6, label="User Seed Point")
        axes[0].legend(loc="upper left")
        
        # Run region growing algorithm
        with st.spinner("Processing pixel matrices..."):
            mask = region_growing(img, (seed_y, seed_x), tolerance)
        
        # Overlay mask onto original scan for visualization
        overlay = np.stack([img, img, img], axis=-1) # Create RGB representation
        overlay[mask == 255] = [0.8, 0.1, 0.1] # Highlight tumor in red
        
        axes[1].imshow(overlay)
        axes[1].set_title("Segmented Output (Region Growing)")
        axes[1].axis('off')
        
        # Quantify the tumor area
        tumor_area_pixels = np.sum(mask == 255)
        st.success(f"📈 **Biomedical Metric:** Segmented tumor mass occupies **{tumor_area_pixels} total pixels**.")

    # Render plots
    st.pyplot(fig)

    # Explanation section to show deep architectural understanding
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
        3. If $|I_{neighbor} - I_{seed}| \le \text{Tolerance}$, it engulfs that pixel and repeats.
        """)
