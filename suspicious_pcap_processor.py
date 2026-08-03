"""
Suspicious PCAP Feature Extraction Module
Extracts 77 features from pcap files for suspicious traffic detection
Matches the feature names used in merged_classified_sampled_datasus.csv
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
    Extract 77 flow features from a pcap file for suspicious traffic detection
    
    Args:
        pcap_file_path: Path to pcap file
        
    Returns:
        List of dictionaries containing 77 flow features
    """
    flows = defaultdict(lambda: {
        'packets': [],
        'bytes': [],
        'timestamps': [],
        'directions': [],  # 1 for forward, -1 for backward
        'packet_lengths': [],
        'header_lengths': [],
        'flags': defaultdict(int),
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
                    # Simple heuristic: smaller IP is considered forward
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
                    
                    # Extract TCP flags
                    if isinstance(ip.data, dpkt.tcp.TCP):
                        tcp = ip.data
                        flows[flow_key]['flags']['FIN'] += tcp.flags & dpkt.tcp.TH_FIN != 0
                        flows[flow_key]['flags']['SYN'] += tcp.flags & dpkt.tcp.TH_SYN != 0
                        flows[flow_key]['flags']['RST'] += tcp.flags & dpkt.tcp.TH_RST != 0
                        flows[flow_key]['flags']['PSH'] += tcp.flags & dpkt.tcp.TH_PUSH != 0
                        flows[flow_key]['flags']['ACK'] += tcp.flags & dpkt.tcp.TH_ACK != 0
                        flows[flow_key]['flags']['URG'] += tcp.flags & dpkt.tcp.TH_URG != 0
                        flows[flow_key]['flags']['ECE'] += tcp.flags & dpkt.tcp.TH_ECE != 0
                        flows[flow_key]['flags']['CWR'] += tcp.flags & dpkt.tcp.TH_CWR != 0
                    
                except Exception as e:
                    continue
            
            # Second pass: calculate features for each flow
            flow_features = []
            
            for flow_key, flow_data in flows.items():
                if len(flow_data['packets']) < 2:
                    continue
                
                timestamps = flow_data['timestamps']
                packets = flow_data['packets']
                directions = flow_data['directions']
                packet_lengths = flow_data['packet_lengths']
                header_lengths = flow_data['header_lengths']
                flags = flow_data['flags']
                
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
                    flow_iat_min = min(iat_times)
                else:
                    flow_iat_mean = flow_iat_std = flow_iat_max = flow_iat_min = 0.0
                
                # Forward and backward IAT
                fwd_iats = []
                bwd_iats = []
                for i in range(1, len(timestamps)):
                    if directions[i] > 0:
                        fwd_iats.append(timestamps[i] - timestamps[i-1])
                    else:
                        bwd_iats.append(timestamps[i] - timestamps[i-1])
                
                if fwd_iats:
                    fwd_iat_total = sum(fwd_iats)
                    fwd_iat_mean = np.mean(fwd_iats)
                    fwd_iat_std = np.std(fwd_iats)
                    fwd_iat_max = max(fwd_iats)
                    fwd_iat_min = min(fwd_iats)
                else:
                    fwd_iat_total = fwd_iat_mean = fwd_iat_std = fwd_iat_max = fwd_iat_min = 0.0
                
                if bwd_iats:
                    bwd_iat_total = sum(bwd_iats)
                    bwd_iat_mean = np.mean(bwd_iats)
                    bwd_iat_std = np.std(bwd_iats)
                    bwd_iat_max = max(bwd_iats)
                    bwd_iat_min = min(bwd_iats)
                else:
                    bwd_iat_total = bwd_iat_mean = bwd_iat_std = bwd_iat_max = bwd_iat_min = 0.0
                
                # Packet rates
                if flow_duration > 0:
                    flow_packets_per_second = total_packets / flow_duration
                    flow_bytes_per_second = total_bytes / flow_duration
                    fwd_packets_per_second = fwd_packets / flow_duration if fwd_packets > 0 else 0.0
                    bwd_packets_per_second = bwd_packets / flow_duration if bwd_packets > 0 else 0.0
                else:
                    flow_packets_per_second = flow_bytes_per_second = 0.0
                    fwd_packets_per_second = bwd_packets_per_second = 0.0
                
                # Active and idle times
                active_threshold = 0.1  # 100ms
                active_times = []
                idle_times = []
                
                for i in range(1, len(timestamps)):
                    if (timestamps[i] - timestamps[i-1]) <= active_threshold:
                        active_times.append(timestamps[i] - timestamps[i-1])
                    else:
                        idle_times.append(timestamps[i] - timestamps[i-1])
                
                if active_times:
                    active_mean = np.mean(active_times)
                    active_std = np.std(active_times)
                    active_max = max(active_times)
                    active_min = min(active_times)
                else:
                    active_mean = active_std = active_max = active_min = 0.0
                
                if idle_times:
                    idle_mean = np.mean(idle_times)
                    idle_std = np.std(idle_times)
                    idle_max = max(idle_times)
                    idle_min = min(idle_times)
                else:
                    idle_mean = idle_std = idle_max = idle_min = 0.0
                
                # Subflow statistics
                subflow_fwd_packets = fwd_packets
                subflow_fwd_bytes = fwd_bytes
                subflow_bwd_packets = bwd_packets
                subflow_bwd_bytes = bwd_bytes
                
                # Window sizes
                # For TCP, we can extract window sizes from SYN packets
                init_fwd_win_bytes = 8192  # Default, would need deeper packet analysis
                init_bwd_win_bytes = 8192  # Default
                
                # Bulk rates (simplified)
                fwd_avg_bulk_rate = 0.0
                fwd_avg_bytes_bulk = 0.0
                fwd_avg_packets_bulk = 0.0
                bwd_avg_bulk_rate = 0.0
                bwd_avg_bytes_bulk = 0.0
                bwd_avg_packets_bulk = 0.0
                
                # Average packet sizes
                avg_packet_size = packet_len_mean if packet_lengths else 0.0
                avg_fwd_segment_size = fwd_bytes / fwd_packets if fwd_packets > 0 else 0.0
                avg_bwd_segment_size = bwd_bytes / bwd_packets if bwd_packets > 0 else 0.0
                
                # Down/Up ratio
                down_up_ratio = bwd_bytes / fwd_bytes if fwd_bytes > 0 else 0.0
                
                # Create feature dictionary with exact 77 feature names
                features = {
                    # Basic features
                    'Protocol': protocol,
                    'Flow Duration': flow_duration,
                    'Total Fwd Packets': fwd_packets,
                    'Total Backward Packets': bwd_packets,
                    'Fwd Packets Length Total': fwd_bytes,
                    'Bwd Packets Length Total': bwd_bytes,
                    
                    # Packet length statistics
                    'Fwd Packet Length Max': packet_len_max,
                    'Fwd Packet Length Min': packet_len_min,
                    'Fwd Packet Length Mean': packet_len_mean,
                    'Fwd Packet Length Std': packet_len_std,
                    'Bwd Packet Length Max': packet_len_max,  # Same as fwd for simplicity
                    'Bwd Packet Length Min': packet_len_min,  # Same as fwd for simplicity
                    'Bwd Packet Length Mean': packet_len_mean,  # Same as fwd for simplicity
                    'Bwd Packet Length Std': packet_len_std,   # Same as fwd for simplicity
                    
                    # Flow rates
                    'Flow Bytes/s': flow_bytes_per_second,
                    'Flow Packets/s': flow_packets_per_second,
                    'Fwd Packets/s': fwd_packets_per_second,
                    'Bwd Packets/s': bwd_packets_per_second,
                    
                    # IAT statistics
                    'Flow IAT Mean': flow_iat_mean,
                    'Flow IAT Std': flow_iat_std,
                    'Flow IAT Max': flow_iat_max,
                    'Flow IAT Min': flow_iat_min,
                    'Fwd IAT Total': fwd_iat_total,
                    'Fwd IAT Mean': fwd_iat_mean,
                    'Fwd IAT Std': fwd_iat_std,
                    'Fwd IAT Max': fwd_iat_max,
                    'Fwd IAT Min': fwd_iat_min,
                    'Bwd IAT Total': bwd_iat_total,
                    'Bwd IAT Mean': bwd_iat_mean,
                    'Bwd IAT Std': bwd_iat_std,
                    'Bwd IAT Max': bwd_iat_max,
                    'Bwd IAT Min': bwd_iat_min,
                    
                    # Flags
                    'Fwd PSH Flags': flags.get('PSH', 0),
                    'Bwd PSH Flags': flags.get('PSH', 0),  # Same as fwd
                    'Fwd URG Flags': flags.get('URG', 0),
                    'Bwd URG Flags': flags.get('URG', 0),  # Same as fwd
                    
                    # Header lengths
                    'Fwd Header Length': fwd_header_len,
                    'Bwd Header Length': bwd_header_len,
                    
                    # Packet statistics
                    'Packet Length Min': packet_len_min,
                    'Packet Length Max': packet_len_max,
                    'Packet Length Mean': packet_len_mean,
                    'Packet Length Std': packet_len_std,
                    'Packet Length Variance': packet_len_variance,
                    
                    # Flag counts
                    'FIN Flag Count': flags.get('FIN', 0),
                    'SYN Flag Count': flags.get('SYN', 0),
                    'RST Flag Count': flags.get('RST', 0),
                    'PSH Flag Count': flags.get('PSH', 0),
                    'ACK Flag Count': flags.get('ACK', 0),
                    'URG Flag Count': flags.get('URG', 0),
                    'CWE Flag Count': flags.get('ECE', 0),
                    'ECE Flag Count': flags.get('ECE', 0),
                    
                    # Ratios and averages
                    'Down/Up Ratio': down_up_ratio,
                    'Avg Packet Size': avg_packet_size,
                    'Avg Fwd Segment Size': avg_fwd_segment_size,
                    'Avg Bwd Segment Size': avg_bwd_segment_size,
                    
                    # Bulk rates
                    'Fwd Avg Bytes/Bulk': fwd_avg_bytes_bulk,
                    'Fwd Avg Packets/Bulk': fwd_avg_packets_bulk,
                    'Fwd Avg Bulk Rate': fwd_avg_bulk_rate,
                    'Bwd Avg Bytes/Bulk': bwd_avg_bytes_bulk,
                    'Bwd Avg Packets/Bulk': bwd_avg_packets_bulk,
                    'Bwd Avg Bulk Rate': bwd_avg_bulk_rate,
                    
                    # Subflow statistics
                    'Subflow Fwd Packets': subflow_fwd_packets,
                    'Subflow Fwd Bytes': subflow_fwd_bytes,
                    'Subflow Bwd Packets': subflow_bwd_packets,
                    'Subflow Bwd Bytes': subflow_bwd_bytes,
                    
                    # Window sizes
                    'Init Fwd Win Bytes': init_fwd_win_bytes,
                    'Init Bwd Win Bytes': init_bwd_win_bytes,
                    
                    # Active/idle statistics
                    'Fwd Act Data Packets': fwd_packets,  # Simplified
                    'Fwd Seg Size Min': packet_len_min,  # Simplified
                    'Active Mean': active_mean,
                    'Active Std': active_std,
                    'Active Max': active_max,
                    'Active Min': active_min,
                    'Idle Mean': idle_mean,
                    'Idle Std': idle_std,
                    'Idle Max': idle_max,
                    'Idle Min': idle_min
                }
                
                flow_features.append(features)
            
            return flow_features
            
    except Exception as e:
        print(f"Error processing suspicious pcap file: {e}")
        return []


def process_suspicious_pcap_file(pcap_file_path: str) -> Dict:
    """
    Process a single pcap file and return 77 features for suspicious detection
    
    Args:
        pcap_file_path: Path to pcap file
        
    Returns:
        Dictionary with 77 flow features (first flow or aggregated)
    """
    flows = extract_flow_features(pcap_file_path)
    
    if not flows:
        return None
    
    # Return the first flow (or you could aggregate all flows)
    # For real-time processing, we'll use the most recent/largest flow
    if len(flows) > 1:
        # Sort by duration (longest first) or packet count
        flows.sort(key=lambda x: x.get('Flow Duration', 0), reverse=True)
    
    return flows[0]
