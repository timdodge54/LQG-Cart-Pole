import numpy as np
import numpy.typing as npt
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
    m_p: float,
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
    
    x = state.item(0)
    x_dot = state.item(1)
    theta = state.item(2)
    theta_dot = state.item(3)
    S = np.sin(theta)
    C = np.cos(theta)
    D = m_p * L* L * (M+m_p * (1-C**2))
    x_ddot = ((1/D)*(-m_p**2*L**2*g*C*S + m_p*L**2*(m_p*L*theta_dot**2*S - d*x_dot)) + m_p*L*L*(1/D)*u)
    theta_ddot = ((1/D)*((m_p+M)*m_p*g*L*S - m_p*L*C*(m_p*L*theta_dot**2*S - d*x_dot)) 
                  - m_p*L*C*(1/D)*u)

    xDot = np.array([x_dot, x_ddot, theta_dot, theta_ddot])

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
    b: float) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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
        [0, -d/M, b*m_p*g/M, 0],
        [0, 0, 0, 1],
        [0, -b*d/(M*L), -b*(m_p+M)*g/(M*L), 0]
        ])

    B = np.array([
        [0],
        [1/M],
        [0],
        [b*1/(M*L)]
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
    # state[2]= np.pi - .1
    kal_state = np.zeros((4,))
    true_measurement = []
    true_measurement.append(state.copy())
    estimate_vector = []
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
            state_diff = y_hat.copy() - np.array([0, 0, np.pi, 0])
        else:
            state_diff = y_hat.copy() - np.array([1, 0, np.pi, 0])
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
    dt = .0001
    t = 0.
    tf = 50
    # zero state init
    state = np.zeros((4,))
    kal_state = np.zeros((4,))
    true_measurement = []
    true_measurement.append(state.copy())
    estimate_vector = []
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


def main():
    # parameters 
    m_p = 1
    M = 5
    L = 2
    g = -10
    d= 1
    b = -1 # pend up = 1
    # get state matracies
    A, B, C = get_state_matracies(m_p, M, L, g, d, b)
    # no input to measurement
    D = np.zeros((C.shape[0], B.shape[1]))
    linSystem = System(A,B,C, D)
    ## Check controllability and observability
    rank_ctrb = ctrl.ctrb(A, B)
    print(f"Rank controlabillity matrix {np.linalg.matrix_rank(rank_ctrb)}")
    rank_obs = ctrl.obsv(A, C)
    print(f"Rank observability matrix {np.linalg.matrix_rank(rank_obs)}")
    # define covariance for disturbances and noise
    beta = 1
    alpha = .1
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
    Q = np.diag([1, 1, 10, 100])
    R = .0001
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
   
    
if __name__ == "__main__":
    main()
    