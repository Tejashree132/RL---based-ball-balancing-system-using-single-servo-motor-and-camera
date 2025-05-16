import cv2
import numpy as np
from gpiozero import Servo
from time import sleep
import random

# Initialize servo on GPIO pin 17
servo = Servo(17)

# Initialize Q-learning parameters
ALPHA = 0.1    # Learning rate
GAMMA = 0.9    # Discount factor
EPSILON = 0.1  # Exploration rate

# State space: position of the ball (X coordinate)
# Action space: servo movement (-1 = left, 0 = stay, 1 = right)
actions = [-1, 0, 1]

# Initialize Q-table with zeros for each state-action pair
Q_table = np.zeros((10, 3))  # 10 discrete states and 3 possible actions

# Function to map X-coordinate to state (discretized)
def get_state(x, frame_width):
    return int((x / frame_width) * 10)  # Map to 10 discrete states

def map_x_to_servo(x, frame_width):
    """Map x-position of the ball to servo value between -1 to 1."""
    return (x / frame_width) * 2 - 1  # Normalized to [-1, 1]

def get_red_ball_position(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define red color range
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create masks and detect circles
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    blurred = cv2.GaussianBlur(mask, (9, 9), 2)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1.2, 50,
                               param1=100, param2=30, minRadius=5, maxRadius=50)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :1]:
            x, y, r = i
            return x, y, r
    return None

# Function to choose action using epsilon-greedy strategy
def choose_action(state):
    if random.random() < EPSILON:
        return random.choice(actions)  # Explore
    return actions[np.argmax(Q_table[state])]  # Exploit

# Function to update the Q-table based on reward and next state
def update_q_table(state, action, reward, next_state):
    action_idx = actions.index(action)
    best_next_action = np.max(Q_table[next_state])
    Q_table[state, action_idx] += ALPHA * (reward + GAMMA * best_next_action - Q_table[state, action_idx])

# Initialize webcam
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Main loop for RL training
while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = get_red_ball_position(frame)

    if result:
        x, y, r = result
        state = get_state(x, frame_width)

        # Choose action based on Q-table
        action = choose_action(state)

        # Map action to servo movement
        servo_pos = action  # -1 = left, 0 = stay, 1 = right
        servo.value = max(-1, min(1, servo_pos))

        # Calculate reward (positive reward if ball is near the center)
        reward = 10 if abs(x - frame_width / 2) < 50 else -1

        # Get the next state
        next_state = get_state(x, frame_width)

        # Update Q-table
        update_q_table(state, action, reward, next_state)

        # Visual feedback
        cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
        cv2.putText(frame, f"Ball Position: ({x}, {y})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow("Red Ball Tracking with RL Servo Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
