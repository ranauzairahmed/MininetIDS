from mininet.topo import Topo
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.cli import CLI
from mininet.node import OVSKernelSwitch, RemoteController

class LargerTreeTopo(Topo):
    """
    This topology has the following tree-like structure:
    - 10 switches (s1 as root, s2-s4 as level 1, s5-s10 as leaves) using OVSKernelSwitch with OpenFlow13 protocol
    - 12 hosts (h1 to h12)
    
    The topology is organized as follows:
    - Root switch (s1) is connected to three level 1 switches (s2, s3, s4)
    - Each level 1 switch is connected to two leaf switches
    - Each leaf switch is connected to two hosts
    
    Host configuration:
    - MAC addresses range from 00:00:00:00:00:01 to 00:00:00:00:00:12
    - IP addresses range from 10.0.0.1/24 to 10.0.0.12/24
    """
    
    def build(self):
        # Add switches
        s1 = self.addSwitch('s1', cls=OVSKernelSwitch, protocols='OpenFlow13')
        
        # Level 1 switches
        level1_switches = []
        for i in range(2, 5):
            switch = self.addSwitch(f's{i}', cls=OVSKernelSwitch, protocols='OpenFlow13')
            level1_switches.append(switch)
            self.addLink(s1, switch)
        
        # Leaf switches and hosts
        host_count = 1
        for i, parent_switch in enumerate(level1_switches):
            for j in range(2):
                leaf_switch = self.addSwitch(f's{5+i*2+j}', cls=OVSKernelSwitch, protocols='OpenFlow13')
                self.addLink(parent_switch, leaf_switch)
                
                # Add two hosts to each leaf switch
                for _ in range(2):
                    host = self.addHost(f'h{host_count}',
                                        mac=f"00:00:00:00:00:{host_count:02d}",
                                        ip=f"10.0.0.{host_count}/24")
                    self.addLink(leaf_switch, host)
                    host_count += 1

def startNetwork():
    topo = LargerTreeTopo()
    c0 = RemoteController('c0', ip='0.0.0.0', port=6653)
    net = Mininet(topo=topo, link=TCLink, controller=c0)

    net.start()
    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    startNetwork()
