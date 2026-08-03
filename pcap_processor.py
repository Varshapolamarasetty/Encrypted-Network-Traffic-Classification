"""
PCAP Feature Extraction Module
Extracts flow features from pcap files for VPN traffic classification
"""

import dpkt
import socket
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
import os


def get_flow_key(ip_src: str, ip_dst: str, port_src: int, port_dst: int, protocol: int) -> str:
    """Create a unique flow key (bidirectional)"""
    # Normalize flow key (smaller IP first)
    if ip_src < ip_dst:
        return f"{ip_src}:{port_src}-{ip_dst}:{port_dst}-{protocol}"
    elif ip_src > ip_dst:
        return f"{ip_dst}:{port_dst}-{ip_src}:{port_src}-{protocol}"
    else:
        # Same IP, use port order
        if port_src < port_dst:
            return f"{ip_src}:{port_src}-{ip_dst}:{port_dst}-{protocol}"
        else:
            return f"{ip_dst}:{port_dst}-{ip_src}:{port_src}-{protocol}"


def extract_flow_features(pcap_file_path: str) -> List[Dict]:
    """
    Extract flow features from a pcap file
    
    Args:
        pcap_file_path: Path to the pcap file
        
    Returns:
        List of dictionaries containing flow features
    """
    flows = defaultdict(lambda: {
        'packets': [],
        'bytes': [],
        'timestamps': [],
        'directions': []  # 1 for forward, -1 for backward
    })
    
    try:
        with open(pcap_file_path, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            
            # First pass: collect all packets
            for timestamp, buf in pcap:
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    
                    # Check if it's IP packet
                    if not isinstance(eth.data, dpkt.ip.IP):
                        continue
                    
                    ip = eth.data
                    src_ip = socket.inet_ntoa(ip.src)
                    dst_ip = socket.inet_ntoa(ip.dst)
                    
                    # Get protocol and ports
                    protocol = ip.p
                    src_port = 0
                    dst_port = 0
                    
                    if protocol == dpkt.ip.IP_PROTO_TCP:
                        if isinstance(ip.data, dpkt.tcp.TCP):
                            src_port = ip.data.sport
                            dst_port = ip.data.dport
                    elif protocol == dpkt.ip.IP_PROTO_UDP:
                        if isinstance(ip.data, dpkt.udp.UDP):
                            src_port = ip.data.sport
                            dst_port = ip.data.dport
                    else:
                        continue
                    
                    # Create flow key
                    flow_key = get_flow_key(src_ip, dst_ip, src_port, dst_port, protocol)
                    
                    # Determine direction (forward = src->dst, backward = dst->src)
                    # Use first packet direction as forward
                    if len(flows[flow_key]['packets']) == 0:
                        direction = 1  # Forward
                    else:
                        # Check if this packet matches forward direction
                        first_src = flows[flow_key]['packets'][0]['src_ip']
                        if src_ip == first_src:
                            direction = 1  # Forward
                        else:
                            direction = -1  # Backward
                    
                    packet_size = len(buf)
                    
                    flows[flow_key]['packets'].append({
                        'src_ip': src_ip,
                        'dst_ip': dst_ip,
                        'src_port': src_port,
                        'dst_port': dst_port,
                        'size': packet_size,
                        'timestamp': timestamp,
                        'direction': direction
                    })
                    flows[flow_key]['bytes'].append(packet_size)
                    flows[flow_key]['timestamps'].append(timestamp)
                    flows[flow_key]['directions'].append(direction)
                    
                except Exception as e:
                    continue
            
            # Second pass: extract features for each flow
            flow_features = []
            
            for flow_key, flow_data in flows.items():
                if len(flow_data['packets']) < 2:
                    continue  # Need at least 2 packets for meaningful features
                
                packets = flow_data['packets']
                timestamps = np.array(flow_data['timestamps'])
                bytes_list = np.array(flow_data['bytes'])
                directions = np.array(flow_data['directions'])
                
                # Sort by timestamp to ensure correct order
                sorted_indices = np.argsort(timestamps)
                timestamps = timestamps[sorted_indices]
                bytes_list = bytes_list[sorted_indices]
                directions = directions[sorted_indices]
                
                # Duration (time between first and last packet)
                duration = timestamps[-1] - timestamps[0]
                if duration == 0:
                    duration = 0.000001  # Avoid division by zero
                
                # Separate forward and backward packets
                forward_mask = directions == 1
                backward_mask = directions == -1
                
                forward_timestamps = timestamps[forward_mask]
                backward_timestamps = timestamps[backward_mask]
                forward_bytes = bytes_list[forward_mask]
                backward_bytes = bytes_list[backward_mask]
                
                # Forward Inter-Arrival Times (FIAT)
                if len(forward_timestamps) > 1:
                    fiat = np.diff(np.sort(forward_timestamps))
                    min_fiat = float(np.min(fiat)) if len(fiat) > 0 else 0.0
                    max_fiat = float(np.max(fiat)) if len(fiat) > 0 else 0.0
                    mean_fiat = float(np.mean(fiat)) if len(fiat) > 0 else 0.0
                    total_fiat = float(np.sum(fiat)) if len(fiat) > 0 else 0.0
                else:
                    min_fiat = max_fiat = mean_fiat = total_fiat = 0.0
                
                # Backward Inter-Arrival Times (BIAT)
                if len(backward_timestamps) > 1:
                    biat = np.diff(np.sort(backward_timestamps))
                    min_biat = float(np.min(biat)) if len(biat) > 0 else 0.0
                    max_biat = float(np.max(biat)) if len(biat) > 0 else 0.0
                    mean_biat = float(np.mean(biat)) if len(biat) > 0 else 0.0
                    total_biat = float(np.sum(biat)) if len(biat) > 0 else 0.0
                else:
                    min_biat = max_biat = mean_biat = total_biat = 0.0
                
                # Flow Inter-Arrival Times (FLOWIAT)
                if len(timestamps) > 1:
                    flowiat = np.diff(np.sort(timestamps))
                    min_flowiat = float(np.min(flowiat)) if len(flowiat) > 0 else 0.0
                    max_flowiat = float(np.max(flowiat)) if len(flowiat) > 0 else 0.0
                    mean_flowiat = float(np.mean(flowiat)) if len(flowiat) > 0 else 0.0
                    std_flowiat = float(np.std(flowiat)) if len(flowiat) > 0 else 0.0
                else:
                    min_flowiat = max_flowiat = mean_flowiat = std_flowiat = 0.0
                
                # Flow statistics
                total_packets = len(packets)
                total_bytes = np.sum(bytes_list)
                flowPktsPerSecond = total_packets / duration if duration > 0 else 0.0
                flowBytesPerSecond = total_bytes / duration if duration > 0 else 0.0
                
                # Active and Idle times
                # Active time: time between consecutive packets in same direction
                # Idle time: time between packets in opposite directions
                active_times = []
                idle_times = []
                
                sorted_indices = np.argsort(timestamps)
                sorted_directions = directions[sorted_indices]
                sorted_timestamps = timestamps[sorted_indices]
                
                for i in range(len(sorted_timestamps) - 1):
                    time_diff = sorted_timestamps[i + 1] - sorted_timestamps[i]
                    if sorted_directions[i] == sorted_directions[i + 1]:
                        active_times.append(time_diff)
                    else:
                        idle_times.append(time_diff)
                
                if len(active_times) > 0:
                    min_active = float(np.min(active_times))
                    max_active = float(np.max(active_times))
                    mean_active = float(np.mean(active_times))
                    std_active = float(np.std(active_times))
                else:
                    min_active = max_active = mean_active = std_active = 0.0
                
                if len(idle_times) > 0:
                    min_idle = float(np.min(idle_times))
                    max_idle = float(np.max(idle_times))
                    mean_idle = float(np.mean(idle_times))
                    std_idle = float(np.std(idle_times))
                else:
                    min_idle = max_idle = mean_idle = std_idle = 0.0
                
                # Create feature dictionary in EXACT order matching training data
                # Order: duration, total_fiat, total_biat, min_fiat, min_biat, max_fiat, max_biat, 
                #        mean_fiat, mean_biat, flowPktsPerSecond, flowBytesPerSecond, 
                #        min_flowiat, max_flowiat, mean_flowiat, std_flowiat,
                #        min_active, mean_active, max_active, std_active,
                #        min_idle, mean_idle, max_idle, std_idle
                features = {
                    'duration': float(duration),
                    'total_fiat': total_fiat,  # Must be 2nd
                    'total_biat': total_biat,  # Must be 3rd
                    'min_fiat': min_fiat,
                    'min_biat': min_biat,
                    'max_fiat': max_fiat,
                    'max_biat': max_biat,
                    'mean_fiat': mean_fiat,
                    'mean_biat': mean_biat,
                    'flowPktsPerSecond': flowPktsPerSecond,
                    'flowBytesPerSecond': flowBytesPerSecond,
                    'min_flowiat': min_flowiat,
                    'max_flowiat': max_flowiat,
                    'mean_flowiat': mean_flowiat,
                    'std_flowiat': std_flowiat,
                    'min_active': min_active,
                    'mean_active': mean_active,
                    'max_active': max_active,
                    'std_active': std_active,
                    'min_idle': min_idle,
                    'mean_idle': mean_idle,
                    'max_idle': max_idle,
                    'std_idle': std_idle
                }
                
                flow_features.append(features)
            
            return flow_features
            
    except Exception as e:
        print(f"Error processing pcap file: {e}")
        return []


def process_pcap_file(pcap_file_path: str) -> Dict:
    """
    Process a single pcap file and return features for prediction
    
    Args:
        pcap_file_path: Path to the pcap file
        
    Returns:
        Dictionary with flow features (first flow or aggregated)
    """
    flows = extract_flow_features(pcap_file_path)
    
    if not flows:
        return None
    
    # Return the first flow (or you could aggregate all flows)
    # For real-time processing, we'll use the most recent/largest flow
    if len(flows) > 1:
        # Sort by duration (longest first) or packet count
        flows.sort(key=lambda x: x.get('duration', 0), reverse=True)
    
    return flows[0]
