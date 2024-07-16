# MininetIDS

MininetIDS is a consolidated environment for training, testing, and deploying Machine Learning-based Intrusion Detection Systems (ML-IDS) in Software-Defined Networks (SDN). It leverages Mininet for network emulation and the Ryu controller for SDN management, providing a comprehensive platform for researchers and network security professionals to develop and evaluate advanced IDS solutions in SDN environments.

## Features

- Dataset management (import, select, analyze, preprocess)
- Machine learning model training (Logistic Regression, KNN, Naive Bayes, Decision Tree, Random Forest)
- Network topology management
- Integration with Ryu controller for IDS functionality
- Feature selection and data preprocessing tools

## Included Datasets

MininetIDS comes with two pre-included datasets for testing and evaluation:

1. MininetFlows
   - Generated using Mininet
   - Contains DoS attacks:
     - ICMP flood
     - TCP SYN flood
     - UDP flood
     - LAND attack
   - Includes normal traffic:
     - HTTP
     - TCP
     - UDP
     - ICMP

2. NSL-KDD
   - Widely used benchmark dataset for network intrusion detection research
   - Improved version of the original KDD Cup 1999 dataset
   - Contains various types of network attacks and normal traffic

These datasets provide a starting point for testing and evaluating IDS models within the MininetIDS environment. The MininetFlows dataset offers simulated traffic that closely matches the Mininet environment, while NSL-KDD provides a standard benchmark for comparison with other IDS research.
## Installation

1. Update your system:
   ```bash
   sudo apt-get update

3. Install Git:
   ```bash
   sudo apt-get install git

5. Clone the repository:
   ```bash
   git clone https://github.com/ranauzairahmed/MininetIDS.git

7. Navigate to the project directory:
   ```bash
   cd MininetIDS

9. Make the installation script executable:
    ```bash
   chmod +x install.sh

11. Run the installation script:
    ```bash
    ./install.sh

## Usage

To start the MininetIDS interface, run:
```bash
python3 MininetIDS.py
```

This will launch the command-line interface where you can use various commands to manage datasets, train models, and control the IDS.

For a full list of commands, use the `help` command in the MininetIDS interface.

## Example Demonstration
[NSL-KDD with MininetIDS](https://drive.google.com/file/d/1ZEJXnq1i4ojhsLwCxiel0xxOfYbzm3Hq/view?usp=sharing)

## License

[MIT License](LICENSE)

### Associated with [Capital University of Science & Technology](https://cust.edu.pk)
  
Faculty of Computing  
BSCS - Final Year Project  

### Supervisor:
Dr. Muhammad Siraj Rathore  
muhammad.siraj@cust.edu.pk  
[Faculty Profile](https://cust.edu.pk/our_team/dr-m-siraj-rathore/)  
[Google Scholar](https://scholar.google.com/citations?user=SX-lTOAAAAAJ&hl=en)  
[ResearchGate](https://www.researchgate.net/profile/Muhammad-Rathore-2)  

### Group:
Rana Uzair Ahmed  
ranauzairahmed.official@gmail.com  
[LinkedIn](https://www.linkedin.com/in/ranauzairahmed/)  

Raja Tayyab Nawaz  
tayyabnawaz177@gmail.com  
[LinkedIn](https://www.linkedin.com/in/rajatayyabnawaz177/)  


#### Reference
[SDN Network DDoS Detection Using Machine Learning](https://github.com/dz43developer/sdn-network-ddos-detection-using-machine-learning)
