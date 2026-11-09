---
title: Crown Porosity
emoji: 🌳
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
license: mit
short_description: SAM 3 Box Prompt & CIELAB a* Foliage & Gap Analyzer
---

# Crown Canopy Foliage Density and Gap Analysis

This project combines **SAM 3 (Segment Anything Model 3)** with geometric prompts and an adaptive **CIELAB a\*** color-analysis algorithm to estimate visible foliage cover and canopy gaps in outdoor tree photographs, while distinguishing woody structures.

## How Can This Tool Support Tree Risk Assessment (TRA)?

Canopy foliage is often assessed visually as part of Tree Risk Assessment (TRA). These assessments can vary with the observer and viewing conditions, making results difficult to quantify consistently or compare. This tool aims to turn visible foliage cover and gaps in photographs into repeatable, semi-quantitative indicators that can supplement TRA records when photographs are taken consistently.

- **Organize canopy observations**: Mark the canopy area and estimate foliage cover and canopy gaps.
- **Record and compare observations**: Save annotated images and measurements to compare visible canopy changes when photographs are taken under similar conditions.
- **Support field inspections**: Provide visual information that tree inspectors can refer to when documenting canopy conditions.

This tool estimates only the foliage cover and gaps visible in a photograph. It **does not measure three-dimensional foliage volume**, diagnose internal decay, root conditions, cracks, or structural strength, or calculate likelihood of failure or risk levels. TRA decisions must also consider on-site inspections and other information, and must be made by qualified professionals. Camera angle, lighting, and occlusion can affect the results.

### Why Use Photograph Analysis Instead of LiDAR?

The two methods are suited to different questions. This project focuses on estimating visible foliage cover and gaps from tree photographs, which are generally easier to obtain and can provide visual, semi-quantitative records for TRA. LiDAR is more suitable for studies that require three-dimensional canopy structure or the spatial distribution of foliage, but it requires scanning equipment and a point-cloud processing workflow. Wind-driven leaf movement may increase fine-scale differences between LiDAR scans taken at different times, especially when comparing individual leaves or small gaps, but this does not mean that LiDAR data are generally invalid.

Photograph analysis was chosen for its accessibility and the project's analysis goals; **this does not mean that photographs are more accurate than LiDAR or can replace it**. LiDAR is more suitable when three-dimensional foliage volume or spatial structure is needed. Wind can affect photographs too: leaves moving during exposure may appear blurred or change the apparent gaps. A photograph captures a brief moment, which may reduce the time over which leaf movement is recorded compared with a longer scan, but it does not eliminate wind-related effects. For more consistent results, take photographs in calm conditions and, where possible, use a sufficiently fast shutter speed. When comparing photographs over time, keep the camera position, angle, and conditions as consistent as possible.

## What Does the Tool Do?

In simple terms, the tool uses a tree photograph to estimate how much of the canopy is covered by leaves and how much consists of gaps between branches and leaves. It also distinguishes the trunk and larger areas of background sky. The tool first identifies the tree to analyze, then classifies regions based on color and shape, and finally provides an annotated image and numerical results.

The process is:

1. **Upload a tree photograph**. If needed, draw a box around the canopy or trunk to analyze.
2. **Identify the tree area** to help avoid including nearby trees or background.
3. **Analyze leaves, branches, and gaps** within the selected area and estimate their proportions.
4. **Review the result image and measurements** to check whether the analysis matches what is visible in the photograph.

Common terms:

| Term | Explanation |
| :--- | :--- |
| **Canopy** | The crown of the tree, formed by its upper branches and leaves. |
| **Canopy envelope** | An imagined outline around the outer edge of the canopy; the area inside includes leaves, branches, and gaps. |
| **Canopy gap** | An opening between branches and leaves through which the background can be seen. |
| **Macro sky isolation** | The tool attempts to handle large areas of sky between major branches separately from smaller canopy gaps, so they do not distort the measurements. |
| **Model Threshold** | The confidence threshold used when the model identifies a canopy region. A higher value is stricter and may miss parts of the canopy; a lower value is more permissive and may include background. It is not a percentage of foliage or an image brightness value. |
| **Trunk Threshold** | The confidence threshold used when the model identifies the trunk and branches. A higher value is stricter and may miss some woody structures; a lower value may include areas that are not part of the trunk or branches. |
| **Max Gap Ratio inside Crown** | The minimum size, relative to the canopy envelope, for a connected opening to be treated as a large sky void and excluded from the canopy area. For example, `0.05` means 5%. A higher value means only larger openings are isolated; `0` disables this isolation. |
| **Green Leaf ExG Threshold** | The cutoff for the Excess Green (ExG) index used by the green-leaf protection mask. A higher value requires a stronger green signal and may exclude some leaves; a lower value includes weaker green signals and may also protect some non-leaf areas. |

Results can be affected by camera angle, lighting, occlusion, and tree species. Review the annotated image to make sure the canopy area is reasonable before interpreting the measurements.

### Photographing Recommendations

- **Photograph upward from near the base of the tree**, aiming to include the target tree's entire canopy.
- **Use a consistent method when comparing photographs**: try to keep the camera position, angle, and distance similar, and note differences in lighting.
- **Minimize other canopies in the frame**. If nearby canopies overlap or could be confused with the target, use the canopy box in the interface to mark the tree you want to analyze.

---

## Key Features

1. **Geometric constraints with box prompts**:
   * Uses a box prompt to constrain the foreground canopy spatially and reduce interference from surrounding trees and background.
2. **Primary-tree priority scoring**:
   * Combines four weights—Gaussian centering (45%), canopy-envelope size (25%), ground-level trunk anchor (20%), and model confidence (10%)—to select a primary tree in the image and classify the others as nearby trees.
3. **CIELAB a\* adaptive color analysis**:
   * Uses the CIE color-space threshold ($a^* < 128$) as a green reference, combined with Otsu adaptive thresholding and a normalized Excess Green (ExG) + HSV safeguard to help retain leaves in highlights and shadows.
4. **Woody-structure repair and extraction of backlit fine branches**:
   * Uses OpenCV contour filling to address dark sampling artifacts and separates backlit fine branches in color space ($L^* < 95, a^* \ge 126$), maintaining the canopy-area relationship:
     $$\text{Canopy Envelope} = \text{Green Foliage} + \text{Woody Structures} + \text{Gaps}$$
5. **Macro sky void isolation**:
   * Uses morphological bridging and connected-component analysis to filter large background sky openings between branches, reducing inflation of the canopy envelope and separating them from smaller gaps.
6. **Interactive web interface**:
   * Includes a FastAPI-based interface for uploading and analyzing a single image, drawing box prompts, and previewing a four-panel diagnostic image.

---

## Canopy Gaps and Macro Sky Isolation

When quantifying canopy structure, large areas of background sky visible between major branches can be mistaken for gaps within the canopy. The `isolate_macro_sky_voids` algorithm is designed to distinguish these areas.

### `max_gap_ratio` and `macro_sky_mask`

| Aspect | Maximum gap ratio (`max_gap_ratio`) | Macro sky isolation mask (`macro_sky_mask`) |
| :--- | :--- | :--- |
| **Role** | Input control parameter | Output mask produced by the algorithm |
| **Data type** | Floating-point scalar (for example, `0.05` means 5.0%) | Two-dimensional Boolean array (`np.ndarray`, shape `(H, W)`, `dtype=bool`) |
| **Reference** | Percentage of the canopy-envelope area | Spatial distribution of identified sky pixels |
| **How it works** | Computes an area cutoff: `area_cutoff = envelope area × max_gap_ratio` | After morphological bridging and 8-connected-component analysis, voids whose individual area is at least `area_cutoff` are included in the mask |
| **Effect** | A larger value raises the area cutoff, so fewer voids are classified as macro sky and more gaps are retained; a smaller value classifies more voids as macro sky | Masked pixels are excluded from the canopy-envelope denominator and left uncolored in the four-panel diagnostic image |

---

## Environment and Installation

This project uses `uv` for Python project and dependency management and targets Python 3.12.

### 1. Set Up the Environment and Install Dependencies
```bash
# Create or synchronize the project environment and install dependencies
uv sync
```

### 2. Start the Web Application
```bash
# Launch the canopy analysis web application
uv run samplingAnalysis/app.py
```
