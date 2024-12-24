import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
import sys
import numpy.typing as npt
import pygame
import matplotlib.pyplot as plt
import control as ctrl
from typing import Tuple
np.random.seed(42)

class System:
    """Struct for system matracies
    
    Attributes:
        A: A matrix
        B: B matrix
        C: C matrix
        D: D matrix
    """
    def __init__(
        self,
        A: npt.NDArray[np.float64],
        B: npt.NDArray[np.float64],
        C: npt.NDArray[np.float64],
        D: npt.NDArray[np.float64]) -> None:
        self.A = A
        self.B = B
        self.C = C
        self.D = D

def dynamics(
    state: npt.NDArray[np.float64],
    m: float,
    M: float,
    L: float,
    g: float,
    d: float,
    u: float
    ) -> npt.NDArray[np.float64]:
    """Dynamics for the inverted pendulam

    Args:
        state: the state of the system (x, xdot, theta, thetadot)
        m_p: mass of the pole
        M: mass of the cart
        L: the length of the pole
        g: gravity
        d: damping factor
        u: force input applied to cart

    Returns:
        dynamics xdot
    """
    Sy = np.sin(state.item(2))
    Cy = np.cos(state.item(2))

    # dx computation
    xdot = state.item(1)
    xddot = (1 / (m * L**2 * (M + m * (1 - Cy**2)))) * (
        -m * m * L**2 * g * Cy * Sy
        + m * L**2 * (m * L * state.item(3)**2 * Sy - d * state.item(1))
    ) + m * L**2 * (1 / (m * L**2 * (M + m * (1 - Cy**2)))) * u
    thetadot = state.item(3)
    thetaddot = (1 / (m * L**2 * (M + m * (1 - Cy**2)))) * (
        (m + M) * m * g * L * Sy
        - m * L * Cy * (m * L * state.item(3)**2 * Sy - d * state.item(1))
    ) - m * L * Cy * (1 / (m * L**2 * (M + m * (1 - Cy**2)))) * u
    
    xDot = np.array([xdot, xddot, thetadot, thetaddot])

    return xDot

def kalmanDynamics(A: npt.NDArray[np.float64], B: npt.NDArray[np.float64], state:npt.NDArray[np.float64], y: float):
    """dynamic function for kalman filter

    Args:
        A: kalman filter a matrix
        B: kalman filter b matrix
        state: kalman filter state
        y: measument of system 

    Returns:
        kalman filter dyamics
    """
    return A@state + B @ y


def get_state_matracies(
    m_p: float,
    M: float,
    L: float,
    g: float,
    d: float,
    s: float) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Get linearized dynamics of the system

    Args:
        m_p: mass of pole
        M: mass of the cart
        L: length of the pole
        g: gravity
        d: damping
        b: idk

    Returns:
        A, B, C
    """
    
    A = np.array([
        [0, 1, 0, 0],
        [0, -d/M, -m_p*g/M, 0],
        [0, 0, 0, 1],
        [0, -s*d/(M*L), -s*(m_p+M)*g/(M*L), 0]
        ])

    B = np.array([
        [0],
        [1/M],
        [0],
        [s*1/(M*L)]
    ])
    C = np.array([[1, 0, 0, 0]])   # shape (1,4)
    return A, B, C

def simulate_LQG(
    system: System,
    kalFilter: System,
    K: npt.NDArray[np.float64],
    m_p:float,
    M: float,
    L: float,
    g: float, 
    d: float, 
    alpha: float,
    beta: float) -> Tuple[list[npt.NDArray[np.float64]], list[npt.NDArray[np.float64]], list[float], list[float]]:
    # simulation parameters
    dt = .0001
    t = 0.
    tf = 50
    # zero state init
    state = np.zeros((4,))
    state[2]= np.pi - 30*(np.pi/360)
    kal_state = np.zeros((4,))
    true_measurement: list[npt.NDArray[np.float64]] = []
    true_measurement.append(state.copy())
    estimate_vector: list[npt.NDArray[np.float64]]= []
    estimate_vector.append(np.zeros(4))
    times_list = [0.]
    u_list = [0.]
    # init control at 0
    u = 0.
    while t < tf:
        # increment time
        t += dt
        times_list.append(t)
        # draw noise and disturbances
        uDist = np.sqrt(alpha)*np.random.randn(4)
        uNoise = np.sqrt(beta)*np.random.randn()
        # get the dynamics for the system
        xdot = dynamics(state, m_p, M, L, g, d, u)
        # propogate dynamics with disturbance
        state += dt*(xdot  + uDist)
        # add the true state measurement
        true_measurement.append(state.copy())
        # get measurment with noise
        y = (system.C@state).item() + uNoise  
        # construct kalman input
        kal_input: npt.NDArray[np.float64] = np.array([u, y])
        # propogate kalman filter dynamics
        kal_state += dt*kalmanDynamics(kalFilter.A, kalFilter.B, kal_state, kal_input)
        # get estimate
        y_hat = kalFilter.C@ kal_state + kalFilter.D @ kal_input
        estimate_vector.append(y_hat.copy())
        # get linearization with 1 unit walk after 10 seconds
        if t < 10:
            state_diff = y_hat.copy() - np.array([0, 0, 0, 0])
        else:
            state_diff = y_hat.copy() - np.array([5, 0, 0, 0])
        # define control law 
        u = (-K @ state_diff).item()
        u_list.append(u)
    return true_measurement, estimate_vector, times_list, u_list

def simulate_estimation(
    system: System,
    kalFilter: System,
    m_p:float,
    M: float,
    L: float,
    g: float, 
    d: float, 
    alpha: float,
    beta: float) -> Tuple[list[npt.NDArray[np.float64]], list[npt.NDArray[np.float64]], list[float], list[float]]:
    dt = .01
    t = 0.
    tf = 50
    # zero state init
    state: npt.NDArray[np.float64]= np.zeros((4,))
    kal_state = np.zeros((4,))
    true_measurement: list[np.float64] = []
    true_measurement.append(state.copy())
    estimate_vector: list[np.float64] = []
    estimate_vector.append(np.zeros(4))
    times_list = [0.]
    u_list = [0.]
    # init control at 0
    u = 0.
    while t < tf:
        # increment time
        t += dt
        times_list.append(t)
        # draw noise and disturbances
        uDist = np.sqrt(alpha)*np.sqrt(dt)*np.random.randn(4)
        uNoise = np.sqrt(beta)*np.random.randn()
        # rule based actuation 
        if t > 1 and t < 1.2:
            u = 100
        elif t > 15 and t < 15.2:
            u = -100
        else:
            u = 0
        u_list.append(u)
        # get the dynamics for the system
        xdot = dynamics(state, m_p, M, L, g, d, u)
        # propogate dynamics with disturbance
        state += dt*(xdot  + uDist)
        # add the true state measurement
        true_measurement.append(state.copy())
        # get measurment with noise
        y = state.item(0) + uNoise  
        kal_input = np.array([u, y])
        # propogate kalman filter dynamics
        kal_state += dt*kalmanDynamics(kalFilter.A, kalFilter.B, kal_state, kal_input)
        # get estimate
        y_hat = kalFilter.C@ kal_state + kalFilter.D@ kal_input
        estimate_vector.append(y_hat)
    return true_measurement, estimate_vector, times_list, u_list

def draw_background(screen, width, height, camera_center, scale):
    """
    Draw a ground line and repeated vertical lines ("fence posts")
    so the user sees motion as the cart moves.
    """
    BLACK = (0, 0, 0)
    GROUND_Y = int(height * 0.75)  # ground line near bottom of screen
    
    # Draw the ground line
    pygame.draw.line(screen, BLACK, (0, GROUND_Y), (width, GROUND_Y), 3)
    
    # Fence spacing (in "meters" of world space):
    fence_spacing = 2.0
    # We'll draw fences in a horizontal range ~2x screen width around camera
    x_min = camera_center - (width/scale)
    x_max = camera_center + (width/scale)
    
    # Step through each fence post location in world coordinates
    x_fence_positions = np.arange(x_min, x_max, fence_spacing)
    for x_fence in x_fence_positions:
        # Convert fence's world x to screen x
        fence_x_screen = int(width//2 + (x_fence - camera_center)*scale)
        # Draw a simple vertical line from ground up a bit
        pygame.draw.line(
            screen, BLACK,
            (fence_x_screen, GROUND_Y),
            (fence_x_screen, GROUND_Y - 30),
            2
        )

def animate_cart_pendulum(
    times, 
    states, 
    inputs, 
    L=2.0, 
    scale=50, 
    arrow_scale=0.1,
    frame_skip=50,
    save_video=False,
    video_filename="cart_pendulum_simulation.mp4"
):
    """
    Animate a cart-pendulum system using pygame, with camera auto-panning
    and an optional scrolling background. Save animation as a video if needed.
    """
    pygame.init()

    width, height = 800, 400
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Cart-Pendulum with Scrolling Background")
    clock = pygame.time.Clock()

    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED   = (255, 0, 0)
    BLUE  = (0, 0, 255)

    cart_width  = 80
    cart_height = 40

    # We'll place the cart's "track" horizontally near the middle
    origin_y = height // 2

    times  = np.array(times)
    states = np.array(states)
    inputs = np.array(inputs)

    max_index = len(times) - 1
    run = True
    i = 0

    # List to store frames for video
    frames = []

    while run and i <= max_index:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        
        screen.fill(WHITE)

        x, x_dot, theta, theta_dot = states[i]
        u = inputs[i]

        # "Camera center" in world coords. We keep cart near center of screen
        camera_center = x

        # 1) Draw a background that scrolls based on camera_center
        draw_background(screen, width, height, camera_center, scale)

        # 2) Now draw the cart-pendulum
        cart_x_screen = int(width//2 + (x - camera_center)*scale)
        cart_y_screen = origin_y

        # Draw the cart
        cart_rect = pygame.Rect(
            cart_x_screen - cart_width // 2,
            cart_y_screen - cart_height // 2,
            cart_width,
            cart_height
        )
        pygame.draw.rect(screen, BLACK, cart_rect)

        # Pivot at top center of the cart
        pivot_x_screen = cart_x_screen
        pivot_y_screen = cart_y_screen - cart_height // 2

        # Pendulum
        pend_length_pixels = L * scale
        pend_end_x = pivot_x_screen + pend_length_pixels * np.sin(theta)
        pend_end_y = pivot_y_screen + pend_length_pixels * np.cos(theta)

        pygame.draw.line(
            screen, RED,
            (pivot_x_screen, pivot_y_screen),
            (pend_end_x, pend_end_y),
            4
        )
        pygame.draw.circle(screen, RED, (int(pend_end_x), int(pend_end_y)), 8)

        # 3) Draw the input arrow
        arrow_len_pixels = int(abs(u) * arrow_scale * scale)
        if u > 0:
            arrow_start = (cart_x_screen + cart_width//2, cart_y_screen)
            arrow_end   = (cart_x_screen + cart_width//2 + arrow_len_pixels, cart_y_screen)
        else:
            arrow_start = (cart_x_screen - cart_width//2 - arrow_len_pixels, cart_y_screen)
            arrow_end   = (cart_x_screen - cart_width//2, cart_y_screen)

        if abs(u) > 1e-3:
            pygame.draw.line(screen, BLUE, arrow_start, arrow_end, 5)

        pygame.display.flip()

        # Save frame if saving video
        if save_video:
            frame = pygame.surfarray.array3d(screen)
            frame = np.transpose(frame, (1, 0, 2))  # Pygame uses (width, height, channels)
            frames.append(frame)

        i += frame_skip
        clock.tick(60)

    pygame.quit()

    # Save video if required
    if save_video:
        clip = ImageSequenceClip(frames, fps=60)
        clip.write_videofile(video_filename, codec="libx264")

    
def main():
    # parameters 
    m_p = 1
    M = 5
    L = 2
    g = -10
    d= 1
    b = 1 # pend up = 1
    # get state matracies
    A, B, C = get_state_matracies(m_p=m_p, M=M, L=L, g=g, d=d, s=b)
    # no input to measurement
    D = np.zeros((C.shape[0], B.shape[1]))
    linSystem = System(A,B,C, D)
    ## Check controllability and observability
    rank_ctrb = ctrl.ctrb(A, B)
    print(f"Rank controlabillity matrix {np.linalg.matrix_rank(rank_ctrb)}")
    rank_obs = ctrl.obsv(A, C)
    print(f"Rank observability matrix {np.linalg.matrix_rank(rank_obs)}")
    # define covariance for disturbances and noise
    beta = .04
    alpha = .0002
    Vd = np.eye(4)*alpha
    Vn = beta
    # define aggregate system input matracies
    # get observer gains
    Kf, _, _ = ctrl.lqe(A, np.eye(4), C, Vd, Vn)
    Q = np.eye(4)
    R = np.eye(1)
    K, _, _, = ctrl.lqr(A, B, Q, R)
    # Define Kalman filter system
    KalA = A - Kf@C
    KalB = np.hstack((B, Kf))
    KalC = np.eye(4)
    KalD = 0*np.hstack((B, Kf))
    kalFilter = System(KalA, KalB, KalC, KalD)
    # define LQR gains
    Q = np.diag([1, 1, 1, 1])
    R = .000001
    K, _, _ = ctrl.lqr(A, B, Q, R)
    true_measurement, estimate_vector, times_list, u_list = simulate_LQG(
        linSystem, kalFilter, K, m_p, M, L, g, d, alpha, beta)
    # plot estimate and truth
    true_measurement = np.array(true_measurement)
    estimate_vector = np.array(estimate_vector)
    fig1, ax = plt.subplots(5, 1)
    for i in range(4):
        ax_ = ax[i]
        ax_.plot(times_list, true_measurement[:, i], label='True x')
        ax_.plot(times_list, estimate_vector[:, i], label='Estimated x')
        ax_.set_ylabel(f'x{i}')
        ax_.set_xlabel('Time (s)')
    ax_ = ax[4]
    ax_.plot(times_list, u_list)
    ax_.set_ylabel(f'u')
    ax_.set_xlabel('Time (s)')
    fig1.legend()
    fig1.suptitle('LQR Controller Cart Position: Truth vs. Kalman Estimate')
    plt.show()
    # animate_cart_pendulum(times_list, true_measurement, u_list, L)
   
    
if __name__ == "__main__":
    main()
    