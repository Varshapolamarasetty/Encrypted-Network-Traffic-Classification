"""
Optimized Suspicious PCAP Feature Extraction Module
Extracts ONLY the 38 features used by the suspicious detection model
No need to extract all 77 features and then filter
"""

import dpkt
import socket
import numpy as np
from collections import defaultdict
from typing import Dict, List
import os

# The 39 features actually used by the model (in the correct order)
MODEL_FEATURES = [
    'Flow Duration',
    'Total Fwd Packets', 
    'Total Backward Packets',
    'Fwd Packets Length Total',
    'Bwd Packets Length Total',
    'Fwd Packet Length Max',
    'Fwd Packet Length Mean', 
    'Fwd Packet Length Std',
    'Bwd Packet Length Max',
    'Bwd Packet Length Mean',
    'Bwd Packet Length Std',
    'Flow Packets/s',
    'Flow IAT Mean',
    'Flow IAT Std', 
    'Flow IAT Max',
    'Fwd IAT Total',
    'Fwd IAT Mean',
    'Fwd IAT Std',
    'Fwd IAT Max',
    'Fwd IAT Min',
    'Fwd Header Length',
    'Bwd Header Length',
    'Fwd Packets/s',  # Missing feature!
    'Bwd Packets/s',
    'Packet Length Max',
    'Packet Length Mean',
    'Packet Length Std',
    'Packet Length Variance',
    'Avg Packet Size',
    'Avg Fwd Segment Size', 
    'Avg Bwd Segment Size',
    'Subflow Fwd Packets',
    'Subflow Fwd Bytes',
    'Subflow Bwd Packets',
    'Subflow Bwd Bytes',
    'Init Fwd Win Bytes',
    'Init Bwd Win Bytes',
    'Fwd Act Data Packets',
    'Idle Mean'
]

def get_flow_key(ip_src: str, ip_dst: str, port_src: int, port_dst: int, protocol: int) -> str:
    """Create a unique flow key (bidirectional)"""
    if ip_src < ip_dst:
        return f"{ip_src}:{port_src}-{ip_dst}:{port_dst}-{protocol}"
    elif ip_src > ip_dst:
        return f"{ip_dst}:{port_dst}-{ip_src}:{port_src}-{protocol}"
    else:
        if port_src < port_dst:
            return f"{ip_src}:{port_src}-{ip_dst}:{port_dst}-{protocol}"
        else:
            return f"{ip_dst}:{port_dst}-{ip_src}:{port_src}-{protocol}"

def extract_model_features_from_pcap(pcap_file_path: str) -> List[Dict]:
    """
    Extract ONLY the 39 features used by the suspicious detection model
    
    Args:
        pcap_file_path: Path to pcap file
        
    Returns:
        List of dictionaries containing exactly 39 model features
    """
    flows = defaultdict(lambda: {
        'packets': [],
        'bytes': [],
        'timestamps': [],
        'directions': [],  # 1 for forward, -1 for backward
        'packet_lengths': [],
        'header_lengths': [],
        'iat_times': [],
        'active_times': [],
        'idle_times': []
    })
    
    try:
        with open(pcap_file_path, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            
            # First pass: collect all packets
            for timestamp, buf in pcap:
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                    if not isinstance(eth.data, dpkt.ip.IP):
                        continue
                    
                    ip = eth.data
                    if not isinstance(ip.data, (dpkt.tcp.TCP, dpkt.udp.UDP)):
                        continue
                    
                    # Extract flow information
                    ip_src = socket.inet_ntoa(ip.src)
                    ip_dst = socket.inet_ntoa(ip.dst)
                    protocol = ip.p
                    
                    # Extract port information
                    if isinstance(ip.data, dpkt.tcp.TCP):
                        port_src = ip.data.sport
                        port_dst = ip.data.dport
                    else:  # UDP
                        port_src = ip.data.sport
                        port_dst = ip.data.dport
                    
                    flow_key = get_flow_key(ip_src, ip_dst, port_src, port_dst, protocol)
                    
                    # Determine direction (forward = 1, backward = -1)
                    direction = 1 if ip_src < ip_dst else -1
                    
                    # Packet data
                    packet_len = len(ip.data) if ip.data else 0
                    header_len = len(ip) - len(ip.data) if ip.data else len(ip)
                    
                    # Update flow statistics
                    flows[flow_key]['packets'].append(packet_len)
                    flows[flow_key]['bytes'].append(packet_len)
                    flows[flow_key]['timestamps'].append(timestamp)
                    flows[flow_key]['directions'].append(direction)
                    flows[flow_key]['packet_lengths'].append(packet_len)
                    flows[flow_key]['header_lengths'].append(header_len)
                    
                except Exception as e:
                    continue
            
            # Second pass: calculate ONLY the 38 required features for each flow
            flow_features = []
            
            for flow_key, flow_data in flows.items():
                if len(flow_data['packets']) < 2:
                    continue
                
                timestamps = flow_data['timestamps']
                packets = flow_data['packets']
                directions = flow_data['directions']
                packet_lengths = flow_data['packet_lengths']
                header_lengths = flow_data['header_lengths']
                
                # Basic flow metrics
                flow_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.000001
                
                # Packet counts
                total_packets = len(packets)
                fwd_packets = sum(1 for d in directions if d > 0)
                bwd_packets = sum(1 for d in directions if d < 0)
                
                # Byte counts
                total_bytes = sum(packets)
                fwd_bytes = sum(p for p, d in zip(packets, directions) if d > 0)
                bwd_bytes = sum(p for p, d in zip(packets, directions) if d < 0)
                
                # Packet length statistics
                if packet_lengths:
                    packet_len_min = min(packet_lengths)
                    packet_len_max = max(packet_lengths)
                    packet_len_mean = np.mean(packet_lengths)
                    packet_len_std = np.std(packet_lengths)
                    packet_len_variance = np.var(packet_lengths)
                else:
                    packet_len_min = packet_len_max = packet_len_mean = packet_len_std = packet_len_variance = 0.0
                
                # Header lengths
                fwd_header_len = sum(h for h, d in zip(header_lengths, directions) if d > 0)
                bwd_header_len = sum(h for h, d in zip(header_lengths, directions) if d < 0)
                
                # Inter-arrival times
                iat_times = []
                for i in range(1, len(timestamps)):
                    iat_times.append(timestamps[i] - timestamps[i-1])
                
                if iat_times:
                    flow_iat_mean = np.mean(iat_times)
                    flow_iat_std = np.std(iat_times)
                    flow_iat_max = max(iat_times)
                else:
                    flow_iat_mean = flow_iat_std = flow_iat_max = 0.0
                
                # Forward IAT
                fwd_iats = []
                for i in range(1, len(timestamps)):
                    if directions[i] > 0:
                        fwd_iats.append(timestamps[i] - timestamps[i-1])
                
                if fwd_iats:
                    fwd_iat_total = sum(fwd_iats)
                    fwd_iat_mean = np.mean(fwd_iats)
                    fwd_iat_std = np.std(fwd_iats)
                    fwd_iat_max = max(fwd_iats)
                    fwd_iat_min = min(fwd_iats)
                else:
                    fwd_iat_total = fwd_iat_mean = fwd_iat_std = fwd_iat_max = fwd_iat_min = 0.0
                
                # Packet rates
                if flow_duration > 0:
                    flow_packets_per_second = total_packets / flow_duration
                    flow_bytes_per_second = total_bytes / flow_duration
                    fwd_packets_per_second = fwd_packets / flow_duration if fwd_packets > 0 else 0.0
                    bwd_packets_per_second = bwd_packets / flow_duration if bwd_packets > 0 else 0.0
                else:
                    flow_packets_per_second = flow_bytes_per_second = fwd_packets_per_second = bwd_packets_per_second = 0.0
                
                # Active and idle times
                active_threshold = 0.1  # 100ms
                active_times = []
                idle_times = []
                
                for i in range(1, len(timestamps)):
                    if (timestamps[i] - timestamps[i-1]) <= active_threshold:
                        active_times.append(timestamps[i] - timestamps[i-1])
                    else:
                        idle_times.append(timestamps[i] - timestamps[i-1])
                
                idle_mean = np.mean(idle_times) if idle_times else 0.0
                
                # Average packet sizes
                avg_packet_size = packet_len_mean if packet_lengths else 0.0
                avg_fwd_segment_size = fwd_bytes / fwd_packets if fwd_packets > 0 else 0.0
                avg_bwd_segment_size = bwd_bytes / bwd_packets if bwd_packets > 0 else 0.0
                
                # Subflow statistics
                subflow_fwd_packets = fwd_packets
                subflow_fwd_bytes = fwd_bytes
                subflow_bwd_packets = bwd_packets
                subflow_bwd_bytes = bwd_bytes
                
                # Window sizes (simplified)
                init_fwd_win_bytes = 8192  # Default
                init_bwd_win_bytes = 8192  # Default
                
                # Create feature dictionary with ONLY 39 model features
                features = {
                    # Flow & Packet Features
                    'Flow Duration': flow_duration,
                    'Total Fwd Packets': fwd_packets,
                    'Total Backward Packets': bwd_packets,
                    'Fwd Packets Length Total': fwd_bytes,
                    'Bwd Packets Length Total': bwd_bytes,
                    
                    # Packet Length Statistics
                    'Fwd Packet Length Max': packet_len_max,
                    'Fwd Packet Length Mean': packet_len_mean,
                    'Fwd Packet Length Std': packet_len_std,
                    'Bwd Packet Length Max': packet_len_max,
                    'Bwd Packet Length Mean': packet_len_mean,
                    'Bwd Packet Length Std': packet_len_std,
                    'Packet Length Max': packet_len_max,
                    'Packet Length Mean': packet_len_mean,
                    'Packet Length Std': packet_len_std,
                    'Packet Length Variance': packet_len_variance,
                    'Avg Packet Size': avg_packet_size,
                    'Avg Fwd Segment Size': avg_fwd_segment_size,
                    'Avg Bwd Segment Size': avg_bwd_segment_size,
                    
                    # Flow Rates
                    'Flow Packets/s': flow_packets_per_second,
                    'Fwd Packets/s': fwd_packets_per_second,
                    'Bwd Packets/s': bwd_packets_per_second,
                    
                    # Inter-Arrival Times
                    'Flow IAT Mean': flow_iat_mean,
                    'Flow IAT Std': flow_iat_std,
                    'Flow IAT Max': flow_iat_max,
                    'Fwd IAT Total': fwd_iat_total,
                    'Fwd IAT Mean': fwd_iat_mean,
                    'Fwd IAT Std': fwd_iat_std,
                    'Fwd IAT Max': fwd_iat_max,
                    'Fwd IAT Min': fwd_iat_min,
                    
                    # Header & Window Features
                    'Fwd Header Length': fwd_header_len,
                    'Bwd Header Length': bwd_header_len,
                    'Init Fwd Win Bytes': init_fwd_win_bytes,
                    'Init Bwd Win Bytes': init_bwd_win_bytes,
                    
                    # Subflow Features
                    'Subflow Fwd Packets': subflow_fwd_packets,
                    'Subflow Fwd Bytes': subflow_fwd_bytes,
                    'Subflow Bwd Packets': subflow_bwd_packets,
                    'Subflow Bwd Bytes': subflow_bwd_bytes,
                    
                    # Active/Idle Features
                    'Fwd Act Data Packets': fwd_packets,
                    'Idle Mean': idle_mean
                }
                
                flow_features.append(features)
            
            return flow_features
            
    except Exception as e:
        print(f"Error processing pcap file: {e}")
        return []

def process_suspicious_pcap_optimized(pcap_file_path: str) -> Dict:
    """
    Process a single pcap file and return 39 features for suspicious detection
    
    Args:
        pcap_file_path: Path to pcap file
        
    Returns:
        Dictionary with exactly 39 flow features (first flow)
    """
    flows = extract_model_features_from_pcap(pcap_file_path)
    
    if not flows:
        return None
    
    # Return the first flow (or you could aggregate all flows)
    if len(flows) > 1:
        flows.sort(key=lambda x: x.get('Flow Duration', 0), reverse=True)
    
    return flows[0]
