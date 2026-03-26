# GBM6710-labs
This repository contains all the code used to complete lab assignements for the GBM6710 _Robotique médicale_ course at Polytechnique Montreal.

---

## Requirements for the installation

### Git
Check that git is installed:
```bash
git --version
```
If it is not installed, please follow the official installation instructions for your operating system:
https://git-scm.com/downloads

### Miniconda
Check that Miniconda is installed:
```bash
conda --version
```
If it is not installed, please follow the official installation instructions for your operating system:
https://www.anaconda.com/docs/getting-started/miniconda/install

---

## Installation steps

### Step 1 – Clone the repository
Open a terminal (Command Prompt, PowerShell, or shell) and navigate to the directory where you want to clone the repository.
```bash
cd <path/to/the/repository>
```
Clone the repository:
```bash
git clone https://github.com/AntoineGuenette/GBM6710-labs
cd GBM6710-labs
```
Verify you're in the correct directory by checking for the required files:
```bash
ls
```
You should see the `lab1`, `lab2` and `lab3` folders.

### Step 2 - Setup a Conda environment
Create a dedicated conda environment named **vwflow**:
```bash
conda create -n gbm6710 python=3.12.12
```
Activate the environment:
```bash
conda activate gbm6710
```
Install the required Python packages:
```bash
pip install -r requirements.txt
```

---

## Available commands

All scripts should be executed **from the root of the repository (`GBM6710-labs`)** using the Python module syntax:

```bash
python -m <module_path>
```

Using `-m` ensures that Python correctly resolves the internal package imports used throughout the repository.

---

### Lab 1 : Kinematics

This lab implements forward and inverse kinematics as well as workspace analysis tools for the robotic system.

Run the **forward kinematics** module:
```bash
python -m lab1.src.forward_kinematics
```
This computes the end-effector pose given a set of joint angles.

Run the **inverse kinematics** module:
```bash
python -m lab1.src.inverse_kinematics
```
This numerically computes the joint configuration required to reach a target pose.

Run the **attainable workspace volume** computation:
```bash
python -m lab1.src.attainable_volume
```
This script samples the robot configuration space to estimate the reachable workspace volume.

---

### Lab 2 : Biopsy planning

Run the trajectory planning simulation:

```bash
python -m lab2.src.trajectory_planning
```

This module computes a feasible biopsy needle trajectory based on:

- The selected biopsy mode
- Tumor position
- Calibration coordinates

>[!Warning]
> Make sure the biopsy mode, tumor coordinates, and calibration parameters are updated in the script before launching the program.

---

### Lab 3 : (to be implemented)

Commands for Lab 3 will be added once the laboratory exercises are released.

---

### Troubleshooting

If you encounter errors such as:

```
ModuleNotFoundError
```

make sure that:

1. You are located in the **root repository directory**:

```bash
cd GBM6710-labs
```

2. The correct conda environment is activated:

```bash
conda activate gbm6710
```

3. The command is executed using `python -m` rather than running the script directly.
