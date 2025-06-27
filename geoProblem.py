import cv2
import numpy as np
import pytesseract
import re
import math

# --- Load and preprocess image ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
img = cv2.imread(r"C:\geoProblem.jpg")
if img is None:
    raise FileNotFoundError("Image not found")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_for_contours = cv2.convertScaleAbs(gray, alpha=2.5, beta=0)
gray_for_contours = cv2.GaussianBlur(gray_for_contours, (5,5), 0)
_, thresh = cv2.threshold(gray_for_contours, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, np.array([90,50,50]), np.array([130,255,255]))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))

contours = [c for c in cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] if cv2.contourArea(c) > 50]
blue_contours = [c for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] if cv2.contourArea(c) > 50]

# --- OCR for blue numbers ---
ocr_image = cv2.bitwise_not(mask)
raw_text = pytesseract.image_to_string(ocr_image, config='--psm 6')
print("OCR raw text:\n", repr(raw_text))

numbers = re.findall(r'\d+\.?\d*', raw_text)
dimensions = {f'L{i}': float(num) for i, num in enumerate(numbers)}
print("\nExtracted dimensions:")
for label, val in dimensions.items():
    print(f"{label}: {val}")

# --- Blue shape analysis ---
print("\nBlue Shape Analysis:")
blue_quad_area = None
for i, cnt in enumerate(blue_contours):
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    area = cv2.contourArea(cnt)
    shape = "Unknown"
    if len(approx) == 3:
        shape = "Triangle"
    elif len(approx) == 4 and cv2.isContourConvex(approx):
        sides = [np.linalg.norm(approx[(j+1)%4][0] - approx[j][0]) for j in range(4)]
        if all(np.isclose(sides[0], s, atol=2) for s in sides[1:]):
            shape = "Square"
        elif np.isclose(sides[0], sides[2], atol=2) and np.isclose(sides[1], sides[3], atol=2):
            shape = "Rectangle"
        else:
            shape = "Quadrilateral"
        if shape in ["Square", "Rectangle", "Quadrilateral"]:
            if blue_quad_area is None or area > blue_quad_area:
                blue_quad_area = area
    elif len(approx) > 10:
        shape = "Circle (approx)"
    else:
        shape = f"Polygon ({len(approx)} sides)"
    print(f"Blue Shape {i}: {shape}, Pixel Area: {area:.1f}")

if blue_quad_area:
    print(f"\nCalculated Area of the blue quadrilateral (ABCD): {blue_quad_area:.1f} pixels^2")
else:
    print("\nNo blue quadrilateral detected.")

# --- General shape analysis ---
print("\nGeneral Shape Analysis:")
for i, cnt in enumerate(contours):
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    area = cv2.contourArea(cnt)
    shape = "Unknown"
    if len(approx) == 3:
        shape = "Triangle"
    elif len(approx) == 4 and cv2.isContourConvex(approx):
        sides = [np.linalg.norm(approx[(j+1)%4][0] - approx[j][0]) for j in range(4)]
        if all(np.isclose(sides[0], s, atol=2) for s in sides[1:]):
            shape = "Square"
        elif np.isclose(sides[0], sides[2], atol=2) and np.isclose(sides[1], sides[3], atol=2):
            shape = "Rectangle"
        else:
            shape = "Quadrilateral"
    elif len(approx) > 10:
        shape = "Circle (approx)"
    else:
        shape = f"Polygon ({len(approx)} sides)"
    print(f"Shape {i}: {shape}, Pixel Area: {area:.1f}")

# --- Geometric calculations ---
def hypotenuse(a, b):
    return math.sqrt(a**2 + b**2)

def trapezoid_area(base1, base2, height):
    return 0.5 * (base1 + base2) * height

# Replace these with actual extracted dimensions as needed
ab = 8.0
bc = 6.0
dc = 2.0
ad = bc  # assuming right trapezoid

print("\n--- Geometric Calculations ---")
print(f"Calculated AC (hypotenuse of ABC): {hypotenuse(ab, bc):.2f} cm")
print(f"Calculated AD (height of trapezoid): {ad:.2f} cm")
print(f"Calculated Area of Trapezoid ABCD: {trapezoid_area(ab, dc, ad):.2f} cm^2")
