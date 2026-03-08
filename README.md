# Peg-in-Hole Teleoperado

**Course:** T3001B - Foundation of Robotics Gpo 101
**Team:** Dream Team
**Professor:** Dr Alberto Muñoz 

## Team Members

| Name | Matrícula | GitHub |
|------|-----------|--------|
| Hector Tovar | A00840308 | [@htovarm7](https://github.com/htovarm7) |
| José Luis Domínguez Morales | A01285873 | [@JLDominguezM](https://github.com/JLDominguezM) |
| Paola Llamas Hernandez | A01178479 | [@PaolaLlh18](https://github.com/PaolaLlh18) |
| Jocelyn Anahi Velarde Barrón | A01285780 | [@JocelynVelarde](https://github.com/JocelynVelarde) |

## Description

Bilateral teleoperation simulation of a **Peg-in-Hole** insertion task using two 3-DOF (3R) planar robots communicating over UDP. A human operator controls the **master robot** via keyboard, while the **slave robot** tracks the master's end-effector position and performs the physical insertion with contact force feedback.

### System Architecture

```
┌─────────────────┐     UDP 9001 (xd, gripper)     ┌─────────────────┐
│   Master Robot   │ ──────────────────────────────► │   Slave Robot    │
│  (Operator PC)   │ ◄────────────────────────────── │  (Task PC)       │
└─────────────────┘     UDP 9002 (Fe, contact)      └─────────────────┘
```

- **Master** (`master_robot.py`): Simulates a 3R planar robot controlled by the operator using arrow keys. Uses **Computed Torque Control** for precise joint-space tracking and sends Cartesian end-effector positions to the slave. Receives and displays contact forces as haptic feedback.

- **Slave** (`slave_robot.py`): Receives desired positions from the master and tracks them using **Impedance Control** in task space. Implements a spring-based contact model for the peg-hole interaction and sends contact forces back to the master.

- **Network Test** (`net_test.py`): UDP round-trip latency and packet loss test to verify the network is suitable for real-time teleoperation (< 10 ms RTT, < 1% loss).

### Key Concepts

| Concept | Implementation |
|---------|---------------|
| Forward Kinematics | 3R planar (3 revolute joints) |
| Inverse Kinematics | Damped Least Squares (DLS) |
| Master Control | Computed Torque (nonlinearity cancellation) |
| Slave Control | Impedance Control (spring-damper in task space) |
| Dynamic Model | Inertia matrix, Coriolis, gravity compensation |
| Contact Model | Elastic wall contact with force threshold detection |
| Communication | UDP sockets, JSON-encoded messages at ~100 Hz |
| Integration | Explicit Euler, dt = 0.01 s |

### Robot Parameters

- Link lengths: L1 = 0.35 m, L2 = 0.30 m, L3 = 0.20 m
- Link masses: M1 = 1.5 kg, M2 = 1.0 kg, M3 = 0.5 kg
- Peg radius: 8 mm | Hole radius: 9 mm (1 mm clearance)

## Usage

### 1. Test Network Connectivity

```bash
# On PC A (receiver):
python3 net_test.py --mode server

# On PC B (sender):
python3 net_test.py --mode client --ip <IP_OF_PC_A>
```

### 2. Run the Slave (on the task PC)

```bash
python3 slave_robot.py --master-ip <MASTER_IP>
```

### 3. Run the Master (on the operator PC)

```bash
python3 master_robot.py --slave-ip <SLAVE_IP>
```

For local testing on a single machine, omit the IP flags (defaults to `127.0.0.1`).

### Controls (Master)

| Key | Action |
|-----|--------|
| Arrow Up / W | Move end-effector +Y |
| Arrow Down / S | Move end-effector -Y |
| Arrow Left / A | Move end-effector -X |
| Arrow Right / D | Move end-effector +X |
| Q | Open gripper |
| E | Close gripper |
| ESC | Quit |

## Requirements

- Python 3
- NumPy
- Matplotlib (TkAgg backend)
