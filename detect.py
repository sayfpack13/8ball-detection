import cv2
import numpy as np

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image from path: " + image_path)
    return image

def detect_inner_blue_area(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No blue area found in the image.")
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    x, y, w, h = cv2.boundingRect(approx)
    x += 12
    y += 12
    w -= 24
    h -= 24
    return x, y, x + w, y + h

def draw_rectangle(image, x1, y1, x2, y2, color=(0, 255, 0), thickness=2):
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image

def detect_balls(image, roi_x1, roi_y1, roi_x2, roi_y2):
    roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_ball = np.array([0, 100, 100])
    upper_ball = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, lower_ball, upper_ball)
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                                    param1=40, param2=10, minRadius=9, maxRadius=13)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0, :]
    else:
        return []

def detect_white_ball(image, roi_x1, roi_y1, roi_x2, roi_y2):
    roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                                    param1=40, param2=15, minRadius=8, maxRadius=12)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        return circles[0][0][0:2]
    else:
        return None

def draw_detected_balls(image, balls, roi_x1, roi_y1):
    for ball in balls:
        cx, cy, r = ball
        cv2.circle(image, (cx + roi_x1, cy + roi_y1), r, (0, 0, 255), 2)

def draw_crosshair(image, ball, roi_x1, roi_y1, color=(0, 255, 0), thickness=2):
    cx, cy, r = ball
    cx_global = cx + roi_x1
    cy_global = cy + roi_y1
    cv2.circle(image, (cx_global, cy_global), r, color, thickness)
    cv2.line(image, (cx_global - r, cy_global), (cx_global + r, cy_global), color, thickness)
    cv2.line(image, (cx_global, cy_global - r), (cx_global, cy_global + r), color, thickness)

def simulate_reflections_after_collision(start_point, initial_velocity,
                                         roi_x1, roi_y1, roi_x2, roi_y2,
                                         max_collisions=20, speed_threshold=1.0, damping=0.98):
    reflection_points = [start_point]
    current_x, current_y = map(float, start_point)
    dx, dy = initial_velocity
    collision_count = 0
    ball_radius = 11

    while collision_count < max_collisions:
        speed = np.sqrt(dx ** 2 + dy ** 2)
        if speed < speed_threshold:
            break

        next_x = current_x + dx
        next_y = current_y + dy
        reflected = False

        if next_x < roi_x1 + ball_radius:
            dx *= -1
            reflected = True
        elif next_x > roi_x2 - ball_radius:
            dx *= -1
            reflected = True
        if next_y < roi_y1 + ball_radius:
            dy *= -1
            reflected = True
        elif next_y > roi_y2 - ball_radius:
            dy *= -1
            reflected = True

        current_x += dx
        current_y += dy
        reflection_points.append((int(current_x), int(current_y)))

        if reflected:
            collision_count += 1
            dx *= damping
            dy *= damping
            if collision_count >= 1: # Stop after the first wall collision for simplicity
                break

    return reflection_points

def draw_reflection_path(image, reflection_points, thickness=2, color=(255, 255, 255)):
    for i in range(len(reflection_points) - 1):
        cv2.line(image, reflection_points[i], reflection_points[i + 1], color, thickness)


def detect_reflection_lines(image, crosshair_center, roi_x1, roi_y1, roi_x2, roi_y2, detected_balls_data=None, cue_ball_data=None):
    cx, cy = crosshair_center
    detected_lines = []
    center_line = None

    # 1. Detect the short white line near the cue ball's center
    roi_size_cue = 60
    roi_half_cue = roi_size_cue // 2
    roi_cue = image[cy - roi_half_cue:cy + roi_half_cue, cx - roi_half_cue:cx + roi_half_cue].copy()
    if roi_cue.size == 0:
        return []

    hsv_cue = cv2.cvtColor(roi_cue, cv2.COLOR_BGR2HSV)
    lower_white_cue = np.array([0, 0, 150])
    upper_white_cue = np.array([180, 70, 255])
    mask_white_cue = cv2.inRange(hsv_cue, lower_white_cue, upper_white_cue)
    edges_white_cue = cv2.Canny(mask_white_cue, 50, 150)

    lines_white_cue = cv2.HoughLinesP(edges_white_cue, 1, np.pi / 180, threshold=10, minLineLength=5, maxLineGap=3)
    candidate_center_lines = []
    origin_threshold_sq = 25

    if lines_white_cue is not None:
        center_roi = (roi_half_cue, roi_half_cue)
        for line in lines_white_cue:
            x1, y1, x2, y2 = line[0]
            start_roi = (x1, y1)
            end_roi = (x2, y2)

            dist_sq_start = (start_roi[0] - center_roi[0])**2 + (start_roi[1] - center_roi[1])**2
            dist_sq_end = (end_roi[0] - center_roi[0])**2 + (end_roi[1] - center_roi[1])**2

            if dist_sq_start < origin_threshold_sq:
                candidate_center_lines.append(((x1 + cx - roi_half_cue, y1 + cy - roi_half_cue), (x2 + cx - roi_half_cue, y2 + cy - roi_half_cue)))
            elif dist_sq_end < origin_threshold_sq:
                candidate_center_lines.append(((x2 + cx - roi_half_cue, y2 + cy - roi_half_cue), (x1 + cx - roi_half_cue, y1 + cy - roi_half_cue))) # Ensure start is closer

        if candidate_center_lines:
            center_line = candidate_center_lines[0]
            detected_lines.append(center_line)
            cv2.line(roi_cue, (center_line[0][0] - (cx - roi_half_cue), center_line[0][1] - (cy - roi_half_cue)),
                     (center_line[1][0] - (cx - roi_half_cue), center_line[1][1] - (cy - roi_half_cue)), (255, 255, 255), 2)

    # 2. Simulate the reflection path
    if center_line and cue_ball_data is not None:
        p1_center, p2_center = center_line
        start_point = np.array([cx, cy], dtype=float)
        end_point = np.array([p2_center[0], p2_center[1]], dtype=float)
        initial_velocity = end_point - start_point

        reflection_points = simulate_reflections_after_collision(
            start_point.astype(int), initial_velocity.astype(int),
            roi_x1, roi_y1, roi_x2, roi_y2, max_collisions=2
        )
        draw_reflection_path(image, reflection_points, color=(255, 255, 255))
        detected_lines.extend([(reflection_points[i], reflection_points[i+1]) for i in range(len(reflection_points) - 1)])

    cv2.imshow("ROI around cue ball for white line", cv2.resize(roi_cue, (300, 300)))
    return detected_lines


def main():
    image_path = "1.png" # Using the latest output image
    image = load_image(image_path)
    try:
        roi_x1, roi_y1, roi_x2, roi_y2 = detect_inner_blue_area(image)
        print(f"ROI: ({roi_x1}, {roi_y1}) to ({roi_x2}, {roi_y2})")
        balls_detected = detect_balls(image, roi_x1, roi_y1, roi_x2, roi_y2)
        print(f"Detected balls: {balls_detected}")
        white_ball = detect_white_ball(image, roi_x1, roi_y1, roi_x2, roi_y2)
        print(f"Detected white ball: {white_ball}")

        filtered_balls = []
        cue_ball_data = None
        if white_ball is not None:
            for ball in balls_detected:
                cx, cy, r = ball
                if np.linalg.norm(np.array([cx, cy]) - np.array(white_ball)) < 5:
                    cue_ball_data = ball
                    break
            if cue_ball_data is not None:
                tolerance = cue_ball_data[2] * 1.2 if len(cue_ball_data) > 2 else 15
                for ball in balls_detected:
                    if np.linalg.norm(np.array([ball[0], ball[1]]) - np.array([cue_ball_data[0], cue_ball_data[1]])) > tolerance:
                        filtered_balls.append(ball)
            else:
                filtered_balls = balls_detected.copy()
        else:
            filtered_balls = balls_detected.copy()

        image_with_balls = draw_rectangle(image.copy(), roi_x1, roi_y1, roi_x2, roi_y2)
        draw_detected_balls(image_with_balls, filtered_balls, roi_x1, roi_y1)

        if cue_ball_data is not None:
            draw_crosshair(image_with_balls, cue_ball_data, roi_x1, roi_y1, color=(0, 255, 0), thickness=2)
            cue_ball_center_global = (int(cue_ball_data[0] + roi_x1), int(cue_ball_data[1] + roi_y1))

            reflection_lines = detect_reflection_lines(image_with_balls, cue_ball_center_global, roi_x1, roi_y1, roi_x2, roi_y2, detected_balls_data=filtered_balls, cue_ball_data=cue_ball_data)
            center_line = None
            outer_lines = []
            if reflection_lines:
                for line in reflection_lines:
                    if center_line is None:
                        center_line = line
                    else:
                        outer_lines.append(line)

            # The reflection path is now drawn within detect_reflection_lines
            # We don't need to simulate and draw it here again

        else:
            print("White ball not detected.")

        cv2.imshow("Detected Pool Balls with Reflections", image_with_balls)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except ValueError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()