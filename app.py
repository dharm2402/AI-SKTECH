import os
import uuid
import base64
import io
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available. Using basic enhancement.")

from flask import Flask, request, jsonify, render_template, url_for
import math

app = Flask(__name__, template_folder='templates', static_folder='static')

# Folder where AI processed images are stored
RESULTS_DIR = os.path.join(app.static_folder, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Initialize AI clients (optional - will use fallback if not available)
openai_client = None
gemini_client = None

# Keep QuickDraw classes (trimmed for readability)
QUICKDRAW_CLASSES = ['circle', 'square', 'triangle', 'cube', 'sphere', 'cylinder', 'cone', 'pyramid']


@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------
# Shape detection utilities
# -----------------------------

def detect_shape_type(contour):
    """Detect what type of geometric shape the contour represents"""
    if len(contour) < 3:
        return None, None

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return None, None

    circularity = 4 * math.pi * area / (perimeter * perimeter)
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 1.0

    if circularity > 0.75:
        return "circle", (x + w//2, y + h//2, min(w, h) // 2)
    elif vertices == 3:
        return "triangle", approx.reshape(-1, 2)
    elif vertices == 4:
        if 0.9 <= aspect_ratio <= 1.1:
            return "square", (x, y, w, h)
        else:
            return "rectangle", (x, y, w, h)
    elif vertices == 5:
        return "pentagon", approx.reshape(-1, 2)
    elif vertices == 6:
        return "hexagon", approx.reshape(-1, 2)
    elif vertices >= 8 and circularity > 0.6:
        return "circle", (x + w//2, y + h//2, min(w, h) // 2)
    else:
        return "smooth", cv2.convexHull(contour)


# -----------------------------
# Drawing perfect 2D projections of 3D shapes
# -----------------------------

def draw_cube_iso(img, center, size, line_thickness=6):
    """Draw a simple isometric cube (2D projection) on the image (PIL ImageDraw or numpy image)."""
    cx, cy = center
    s = size // 2
    # Offset for isometric projection
    ox = int(s * 0.6)
    oy = int(s * 0.35)
    # 8 cube corners (projected)
    A = (cx - s, cy - s)
    B = (cx + s, cy - s)
    C = (cx + s, cy + s)
    D = (cx - s, cy + s)
    A2 = (A[0] + ox, A[1] - oy)
    B2 = (B[0] + ox, B[1] - oy)
    C2 = (C[0] + ox, C[1] - oy)
    D2 = (D[0] + ox, D[1] - oy)
    # Draw on numpy image (BGR)
    if CV2_AVAILABLE:
        # lines for front face
        cv2.line(img, A, B, (0, 0, 0), line_thickness)
        cv2.line(img, B, C, (0, 0, 0), line_thickness)
        cv2.line(img, C, D, (0, 0, 0), line_thickness)
        cv2.line(img, D, A, (0, 0, 0), line_thickness)
        # back face
        cv2.line(img, A2, B2, (0, 0, 0), line_thickness)
        cv2.line(img, B2, C2, (0, 0, 0), line_thickness)
        cv2.line(img, C2, D2, (0, 0, 0), line_thickness)
        cv2.line(img, D2, A2, (0, 0, 0), line_thickness)
        # connectors
        cv2.line(img, A, A2, (0, 0, 0), line_thickness)
        cv2.line(img, B, B2, (0, 0, 0), line_thickness)
        cv2.line(img, C, C2, (0, 0, 0), line_thickness)
        cv2.line(img, D, D2, (0, 0, 0), line_thickness)
    else:
        draw = ImageDraw.Draw(img)
        draw.line([A, B, C, D, A], fill=(0, 0, 0), width=line_thickness)
        draw.line([A2, B2, C2, D2, A2], fill=(0, 0, 0), width=line_thickness)
        draw.line([A, A2], fill=(0, 0, 0), width=line_thickness)
        draw.line([B, B2], fill=(0, 0, 0), width=line_thickness)
        draw.line([C, C2], fill=(0, 0, 0), width=line_thickness)
        draw.line([D, D2], fill=(0, 0, 0), width=line_thickness)


def draw_sphere(img, center, radius, line_thickness=6):
    cx, cy = center
    if CV2_AVAILABLE:
        cv2.circle(img, (cx, cy), radius, (0, 0, 0), line_thickness)
        # optional equator ellipse
        cv2.ellipse(img, (cx, cy), (int(radius*0.9), int(radius*0.35)), 0, 0, 360, (0,0,0), line_thickness)
    else:
        draw = ImageDraw.Draw(img)
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=(0,0,0), width=line_thickness)
        draw.ellipse([cx - int(radius*0.9), cy - int(radius*0.35), cx + int(radius*0.9), cy + int(radius*0.35)], outline=(0,0,0), width=line_thickness)


def draw_cylinder(img, center, radius, height, line_thickness=6):
    cx, cy = center
    top = cy - height//2
    bottom = cy + height//2
    if CV2_AVAILABLE:
        # top ellipse
        cv2.ellipse(img, (cx, top), (radius, int(radius*0.4)), 0, 0, 360, (0,0,0), line_thickness)
        # bottom ellipse (dashed/solid)
        cv2.ellipse(img, (cx, bottom), (radius, int(radius*0.4)), 0, 0, 360, (0,0,0), line_thickness)
        # sides
        cv2.line(img, (cx - radius, top), (cx - radius, bottom), (0,0,0), line_thickness)
        cv2.line(img, (cx + radius, top), (cx + radius, bottom), (0,0,0), line_thickness)
    else:
        draw = ImageDraw.Draw(img)
        draw.ellipse([cx - radius, top - int(radius*0.4), cx + radius, top + int(radius*0.4)], outline=(0,0,0), width=line_thickness)
        draw.ellipse([cx - radius, bottom - int(radius*0.4), cx + radius, bottom + int(radius*0.4)], outline=(0,0,0), width=line_thickness)
        draw.line([(cx - radius, top), (cx - radius, bottom)], fill=(0,0,0), width=line_thickness)
        draw.line([(cx + radius, top), (cx + radius, bottom)], fill=(0,0,0), width=line_thickness)


def draw_cone(img, center, base_radius, height, line_thickness=6):
    cx, cy = center
    top = cy - height//2
    bottom = cy + height//2
    if CV2_AVAILABLE:
        # base ellipse
        cv2.ellipse(img, (cx, bottom), (base_radius, int(base_radius*0.4)), 0, 0, 360, (0,0,0), line_thickness)
        # sides to apex
        cv2.line(img, (cx - base_radius, bottom), (cx, top), (0,0,0), line_thickness)
        cv2.line(img, (cx + base_radius, bottom), (cx, top), (0,0,0), line_thickness)
    else:
        draw = ImageDraw.Draw(img)
        draw.ellipse([cx - base_radius, bottom - int(base_radius*0.4), cx + base_radius, bottom + int(base_radius*0.4)], outline=(0,0,0), width=line_thickness)
        draw.line([(cx - base_radius, bottom), (cx, top)], fill=(0,0,0), width=line_thickness)
        draw.line([(cx + base_radius, bottom), (cx, top)], fill=(0,0,0), width=line_thickness)


def draw_pyramid(img, center, base_size, height, line_thickness=6):
    cx, cy = center
    s = base_size // 2
    top = (cx, cy - height//2)
    A = (cx - s, cy + height//2)
    B = (cx + s, cy + height//2)
    if CV2_AVAILABLE:
        cv2.line(img, A, B, (0,0,0), line_thickness)
        cv2.line(img, A, top, (0,0,0), line_thickness)
        cv2.line(img, B, top, (0,0,0), line_thickness)
    else:
        draw = ImageDraw.Draw(img)
        draw.line([A, B], fill=(0,0,0), width=line_thickness)
        draw.line([A, top], fill=(0,0,0), width=line_thickness)
        draw.line([B, top], fill=(0,0,0), width=line_thickness)


# -----------------------------
# 3D object recognition (heuristic)
# -----------------------------

# Reuse the detect_3d_object_from_sketch implementation but we'll keep it here as a function
# so predict() can call it and then render a perfect 3D projection.

def _detect_lines_and_ellipses(gray):
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    raw_lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=40, minLineLength=20, maxLineGap=5)
    lines = []
    if raw_lines is not None:
        for l in raw_lines:
            x1, y1, x2, y2 = l[0]
            lines.append((x1, y1, x2, y2))

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    ellipse_contours = []
    for cnt in contours:
        if len(cnt) < 20:
            continue
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        try:
            ellipse = cv2.fitEllipse(cnt)
        except:
            continue
        (cx, cy), (MA, ma), angle = ellipse
        if MA <= 0 or ma <= 0:
            continue
        ratio = max(MA, ma) / min(MA, ma)
        if ratio < 3.0:
            ellipse_contours.append(cnt)

    return lines, ellipse_contours


def _group_line_angles(lines, angle_eps=np.deg2rad(12)):
    if not lines:
        return 0
    angles = []
    for x1, y1, x2, y2 in lines:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        angle = math.atan2(dy, dx)
        if angle < 0:
            angle += math.pi
        angles.append(angle)
    groups = []
    for a in angles:
        placed = False
        for g in groups:
            if abs(a - g) < angle_eps:
                placed = True
                break
        if not placed:
            groups.append(a)
    return len(groups)


def _count_polygon_types(contours):
    counts = {"triangles": 0, "quads": 0, "pentagons": 0, "hexagons": 0}
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        if peri < 20:
            continue
        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
        v = len(approx)
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        if v == 3:
            counts["triangles"] += 1
        elif v == 4:
            counts["quads"] += 1
        elif v == 5:
            counts["pentagons"] += 1
        elif v == 6:
            counts["hexagons"] += 1
    return counts


def detect_3d_object_from_sketch(image_bytes):
    if not CV2_AVAILABLE:
        return [{"label": "opencv_not_available", "confidence": 0.0}]

    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return [{"label": "decode_error", "confidence": 0.0}]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines, ellipse_contours = _detect_lines_and_ellipses(gray)
    poly_counts = _count_polygon_types(contours)
    n_lines = len(lines)
    n_ellipses = len(ellipse_contours)
    angle_groups = _group_line_angles(lines)

    total_area = sum(cv2.contourArea(c) for c in contours)
    img_area = gray.shape[0] * gray.shape[1]
    coverage = total_area / img_area if img_area > 0 else 0.0

    scores = {"cube_box": 0.0, "sphere": 0.0, "cylinder": 0.0, "cone": 0.0, "pyramid": 0.0, "unknown_3d_object": 0.1}

    # Sphere heuristic: one or more ellipses, low line count
    if n_ellipses >= 1 and n_lines <= 10:
        scores["sphere"] += 0.6 + 0.05 * (n_ellipses - 1)
    if n_lines > 12:
        scores["sphere"] -= 0.2

    # Cube/box heuristic: many quads and multiple angle groups
    if poly_counts["quads"] >= 2:
        scores["cube_box"] += 0.5 + 0.1 * max(0, poly_counts["quads"] - 2)
    if angle_groups >= 3:
        scores["cube_box"] += 0.2
    if n_lines >= 6:
        scores["cube_box"] += 0.1

    # Cylinder heuristic: ellipses + parallel vertical lines
    if n_ellipses >= 1 and n_lines >= 2:
        scores["cylinder"] += 0.5
        if angle_groups in [1, 2, 3]:
            scores["cylinder"] += 0.1
    if n_ellipses >= 2:
        scores["cylinder"] += 0.1

    # Cone heuristic: triangle(s) with an ellipse/base
    if poly_counts["triangles"] >= 1 and n_ellipses >= 1:
        scores["cone"] += 0.5
    if poly_counts["triangles"] >= 2:
        scores["cone"] += 0.1

    # Pyramid heuristic: multiple triangles, few ellipses
    if poly_counts["triangles"] >= 2 and n_ellipses == 0:
        scores["pyramid"] += 0.5
    if poly_counts["triangles"] >= 3:
        scores["pyramid"] += 0.1

    if coverage < 0.005:
        scores["unknown_3d_object"] += 0.5

    max_score = max(scores.values()) if scores else 1.0
    predictions = []
    for label, score in scores.items():
        conf = 0.0 if max_score <= 0 else score / max_score
        predictions.append({"label": label, "confidence": float(max(0.0, min(1.0, conf)))})
    predictions.sort(key=lambda x: x["confidence"], reverse=True)
    return predictions[:3]


# -----------------------------
# Render a perfect 3D projection image based on detected label
# -----------------------------

def render_perfect_3d(label, canvas_size=(1024, 1024)):
    # Create white background
    if CV2_AVAILABLE:
        img = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255
        center = (canvas_size[0] // 2, canvas_size[1] // 2)
        size = min(canvas_size) // 3
        if label == 'sphere':
            draw_sphere(img, center, size)
        elif label == 'cylinder':
            draw_cylinder(img, center, int(size*0.8), int(size*1.2))
        elif label == 'cone':
            draw_cone(img, center, int(size*0.9), int(size*1.2))
        elif label == 'pyramid':
            draw_pyramid(img, center, int(size*1.2), int(size*1.2))
        else:  # cube_box or fallback
            draw_cube_iso(img, center, int(size*1.1))
        # Convert to PNG bytes
        is_success, buffer = cv2.imencode('.png', img)
        return buffer.tobytes()
    else:
        img = Image.new('RGB', canvas_size, (255, 255, 255))
        center = (canvas_size[0] // 2, canvas_size[1] // 2)
        size = min(canvas_size) // 3
        if label == 'sphere':
            draw_sphere(img, center, size)
        elif label == 'cylinder':
            draw_cylinder(img, center, int(size*0.8), int(size*1.2))
        elif label == 'cone':
            draw_cone(img, center, int(size*0.9), int(size*1.2))
        elif label == 'pyramid':
            draw_pyramid(img, center, int(size*1.2), int(size*1.2))
        else:
            draw_cube_iso(img, center, int(size*1.1))
        output = io.BytesIO()
        img.save(output, format='PNG', optimize=True)
        return output.getvalue()


# -----------------------------
# Enhanced endpoint: predict -> will detect 3D object then return perfect 3D projection
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Missing upload: field name 'file'"}), 400

    uploaded = request.files["file"]
    image_bytes = uploaded.read()

    if not image_bytes or len(image_bytes) == 0:
        return jsonify({"error": "Empty file uploaded"}), 400

    try:
        # First, run heuristic 3D recognizer
        guesses = detect_3d_object_from_sketch(image_bytes)
        top = guesses[0]
        label = top['label']
        confidence = top['confidence']

        # If confidence is low, still render the most likely 3D candidate
        print(f"3D detection top: {label} ({confidence:.2f})")

        # Map internal labels to simpler labels used in renderer
        label_map = {
            'cube_box': 'cube',
            'sphere': 'sphere',
            'cylinder': 'cylinder',
            'cone': 'cone',
            'pyramid': 'pyramid'
        }
        render_label = label_map.get(label, 'cube')

        # Render a perfect 3D projection image
        rendered_bytes = render_perfect_3d(render_label)

        # Save the processed image
        filename = f"result_{uuid.uuid4().hex}.png"
        out_path = os.path.join(RESULTS_DIR, filename)
        with open(out_path, "wb") as f:
            f.write(rendered_bytes)

        return jsonify({
            "output_url": url_for("static", filename=f"results/{filename}"),
            "method": "3d-heuristic-render",
            "predictions": guesses
        })

    except Exception as e:
        print(f"Image processing error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 500


# Keep the QuickDraw and recognize3d endpoints but they can be used separately
@app.route("/quickdraw", methods=["POST"])
def quickdraw():
    return jsonify({"error": "Use /predict for 3D automatic rendering in this server."}), 400


@app.route("/recognize3d", methods=["POST"])
def recognize3d():
    if "file" not in request.files:
        return jsonify({"error": "Missing upload: field name 'file'"}), 400
    uploaded = request.files["file"]
    image_bytes = uploaded.read()
    try:
        predictions = detect_3d_object_from_sketch(image_bytes)
        return jsonify({"predictions": predictions, "model": "3d-heuristic"})
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True)
