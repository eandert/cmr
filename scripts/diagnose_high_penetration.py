#!/usr/bin/env python3
"""
Diagnostic script to capture real SUMO data and analyze detection/tracking behavior
at high CAV penetration rates.

This script:
1. Runs a SUMO simulation at specified CAV penetration
2. Records raw detection data, ground truth, and fusion results
3. Saves to JSON for replay in unit tests
4. Analyzes for anomalies: vehicle jumps, duplicate tracks, mismatches

Run from src directory: cd src && python ../scripts/diagnose_high_penetration.py
"""

import os
import sys
import json
import math
import time as time_module
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Add src to path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(script_dir, '..', 'src')
sys.path.insert(0, src_dir)

import traci
import ground_truth
import sensor_fusion
import utils
from config_templates import get_pointpillars_os1_128_config
from config import sensor_type, detector_type
from error import ErrorPackage, ErrorType


@dataclass 
class VehicleState:
    """State of a single vehicle at a point in time."""
    vehicle_id: str
    x: float
    y: float
    angle: float
    vx: float
    vy: float
    vehicle_type: int
    
    def to_dict(self):
        return asdict(self)


@dataclass
class DetectionRecord:
    """A single detection record."""
    source_cav: str
    target_vehicle_id: str
    x: float
    y: float
    angle: float
    cov_trace: float
    is_self_localization: bool
    
    def to_dict(self):
        return asdict(self)


@dataclass
class TrackRecord:
    """A single global track record."""
    track_id: int
    x: float
    y: float
    cov_trace: float
    dx: float
    dy: float
    
    def to_dict(self):
        return asdict(self)


@dataclass
class FrameData:
    """Data captured for a single frame."""
    step: int
    sim_time: float
    ground_truth: List[VehicleState] = field(default_factory=list)
    detections: List[DetectionRecord] = field(default_factory=list)
    global_tracks: List[TrackRecord] = field(default_factory=list)
    num_cavs: int = 0
    num_vehicles: int = 0
    
    def to_dict(self):
        return {
            'step': self.step,
            'sim_time': self.sim_time,
            'ground_truth': [gt.to_dict() for gt in self.ground_truth],
            'detections': [d.to_dict() for d in self.detections],
            'global_tracks': [t.to_dict() for t in self.global_tracks],
            'num_cavs': self.num_cavs,
            'num_vehicles': self.num_vehicles
        }


def run_diagnostic(
    fast_forward_steps: int = 1000,
    warmup_steps: int = 100,
    record_steps: int = 100,
    cav_probability: float = 1.0,
    output_file: str = None
):
    """
    Run diagnostic simulation and capture data.
    
    Args:
        fast_forward_steps: Steps to fast-forward (no processing, just let vehicles spawn)
        warmup_steps: Steps to run with processing but no recording (let tracks stabilize)
        record_steps: Number of steps to record
        cav_probability: Fraction of vehicles that are CAVs (1.0 = 100%)
        output_file: Path to save JSON output (None = auto-generate)
    """
    import sensor
    import localizer
    
    print(f"=" * 70)
    print(f"HIGH PENETRATION DIAGNOSTIC")
    print(f"=" * 70)
    print(f"CAV Penetration: {cav_probability * 100:.0f}%")
    print(f"Fast-forward steps: {fast_forward_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Recording steps: {record_steps}")
    print()
    
    # Get config
    config = get_pointpillars_os1_128_config()
    config["cav_probability"] = cav_probability
    config["use_gpem_model"] = True
    config["use_quadratic"] = False
    config["use_trust_scoring"] = False
    
    # Start SUMO
    sumo_config_path = os.path.join(src_dir, "..", config["sumo_config_path"])
    sumoBinary = "sumo"
    sumoCmd = [
        sumoBinary,
        "-c", sumo_config_path,
        "--step-length", str(config["step_length"]),
        "--no-warnings",
        "--no-step-log"
    ]
    
    simulation_label = f"diag_{os.getpid()}_{int(time_module.time() * 1000) % 100000}"
    traci.start(sumoCmd, label=simulation_label)
    traci_conn = traci.getConnection(simulation_label)
    
    print("SUMO started successfully")
    
    # Error package (no errors for diagnostic)
    error_package_instance = ErrorPackage(ErrorType.SINGLE_SENSOR_EXTRINSICS, 0)
    
    # Get GPEM flags
    use_gpem_model = config.get("use_gpem_model", False)
    use_quadratic = config.get("use_quadratic", False)
    
    # Build sensor packages (same as traci_interface)
    detector_allocation = config.get("detector_allocation")
    localizer_allocation = config.get("localizer_allocation")
    
    if detector_allocation is not None and localizer_allocation is not None:
        total_det = sum(w for w, _ in detector_allocation)
        total_loc = sum(w for w, _ in localizer_allocation)
        det_weights = [(w / total_det if total_det else 1.0 / len(detector_allocation), n) for w, n in detector_allocation]
        loc_weights = [(w / total_loc if total_loc else 1.0 / len(localizer_allocation), n) for w, n in localizer_allocation]
        base_sensor_type = config.get("distribution_base_sensor_type", "OS1_128")
        base_extrinsics = config.get("sensor_extrinsics", [(0.0, 0.0, 0.0)])
        sensor_packages_for_managers = []
        for det_prob, det_name in det_weights:
            for loc_prob, loc_name in loc_weights:
                prob = det_prob * loc_prob
                _detector_type = getattr(detector_type.DetectorType, det_name)
                one_sensor = sensor.Sensor(
                    sensor_type=getattr(sensor_type.SensorType, base_sensor_type),
                    detector_type=_detector_type,
                    use_gpem_model=use_gpem_model,
                    use_quadratic=use_quadratic
                )
                loc_instance = localizer.get_localizer(
                    localizer_type=loc_name,
                    error_package=error_package_instance,
                    use_gpem_model=use_gpem_model,
                    use_quadratic=use_quadratic
                )
                sensor_packages_for_managers.append((prob, [one_sensor], base_extrinsics, loc_instance))
    else:
        sensors = []
        for i in range(len(config["sensor_types"])):
            _sensor_type = getattr(sensor_type.SensorType, config["sensor_types"][i])
            _detector_type = getattr(detector_type.DetectorType, config["detector_types"][i])
            sensors.append(sensor.Sensor(
                sensor_type=_sensor_type,
                detector_type=_detector_type,
                use_gpem_model=use_gpem_model,
                use_quadratic=use_quadratic
            ))
        default_localizer_type = config.get("localizer_type", "kiss_icp")
        lidar_slam = localizer.get_localizer(
            localizer_type=default_localizer_type,
            error_package=error_package_instance,
            use_gpem_model=use_gpem_model,
            use_quadratic=use_quadratic
        )
        sensor_packages_for_managers = [(1, sensors, config["sensor_extrinsics"], lidar_slam)]
    
    # Setup CAV manager
    cav_manager = utils.VehicleProbabilityManager(
        probability=config["cav_probability"],
        type="CAV",
        sumo_type="CAV_passenger",
        sensor_packages=sensor_packages_for_managers,
        error_package=error_package_instance
    )
    
    # Global fusion
    global_fusion = sensor_fusion.Fusion(sensor_fusion.max_id, use_trust_scoring=False)
    
    # Storage for frame data
    frame_data_list: List[FrameData] = []
    
    # Ground truth history for jump detection
    gt_history: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
    
    total_steps = fast_forward_steps + warmup_steps + record_steps
    
    print(f"\nPhase 1: Fast-forward ({fast_forward_steps} steps)...")
    
    for step in range(total_steps):
        traci_conn.simulationStep()
        sim_time = traci_conn.simulation.getTime()
        
        # Get all vehicle IDs
        all_vehicle_ids = traci_conn.vehicle.getIDList()
        
        # Progress
        if step % 100 == 0:
            print(f"  Step {step}/{total_steps}, vehicles={len(all_vehicle_ids)}")
        
        # Fast-forward phase - just step, no processing
        if step < fast_forward_steps:
            # Still need to update CAV manager during FF to track which are CAVs
            if step >= 1:
                cav_manager.update_vehicles(traci_conn, all_vehicle_ids)
            continue
        
        if step == fast_forward_steps:
            print(f"\nPhase 2: Warmup ({warmup_steps} steps)...")
        
        if step == fast_forward_steps + warmup_steps:
            print(f"\nPhase 3: Recording ({record_steps} steps)...")
        
        # Update CAV manager
        cav_manager.update_vehicles(traci_conn, all_vehicle_ids)
        cav_id_list = cav_manager.get_active_vehicle_ids()
        cav_instances = list(cav_manager.get_active_vehicle_instances())
        
        # Create ground truth
        ground_truth_global = {}
        for vid in all_vehicle_ids:
            gt_obj = ground_truth.create_ground_truth_for_vehicle_by_id(traci_conn, vid)
            if gt_obj:
                ground_truth_global[vid] = gt_obj
        
        # Build spatial index for detection (expects list of GT objects)
        ground_truth_list = list(ground_truth_global.values())
        ground_truth_spatial_index, _ = ground_truth.build_ground_truth_spatial_index(ground_truth_list)
        
        # Determine if we're recording
        is_recording = step >= (fast_forward_steps + warmup_steps)
        
        if is_recording:
            frame = FrameData(step=step, sim_time=sim_time)
            frame.num_vehicles = len(all_vehicle_ids)
            frame.num_cavs = len(cav_id_list)
            
            # Record ground truth
            for vid, gt_obj in ground_truth_global.items():
                frame.ground_truth.append(VehicleState(
                    vehicle_id=vid,
                    x=gt_obj.centroid[0],
                    y=gt_obj.centroid[1],
                    angle=gt_obj.angle,
                    vx=gt_obj.velocity_vector[0],
                    vy=gt_obj.velocity_vector[1],
                    vehicle_type=gt_obj.type
                ))
                # Track history for jump detection
                gt_history[vid].append((gt_obj.centroid[0], gt_obj.centroid[1], sim_time))
        
        # Process each CAV
        for cav_id, cav_instance in zip(cav_id_list, cav_instances):
            cav_gt_obj = ground_truth_global.get(cav_id)
            if cav_gt_obj is None:
                continue
            
            # Run detection
            fusion_result, detected_objects, detectable_gt = \
                cav_instance.create_detection_sets(cav_gt_obj, ground_truth_list, ground_truth_spatial_index, sim_time)
            
            # Feed to global fusion
            global_fusion.processDetectionFrame(sim_time, detected_objects, config["cleanup_time"])
            
            if is_recording:
                # Record detections from this CAV
                for det in detected_objects:
                    cov_trace = np.trace(det.error_covariance) if det.error_covariance is not None else 0.0
                    frame.detections.append(DetectionRecord(
                        source_cav=cav_id,
                        target_vehicle_id=det.vehicle_id,
                        x=det.centroid[0],
                        y=det.centroid[1],
                        angle=det.angle,
                        cov_trace=float(cov_trace),
                        is_self_localization=(det.vehicle_id == cav_id)
                    ))
        
        # Run global fusion
        global_result, global_detections, _, _ = global_fusion.fuseDetectionFrame(sim_time)
        
        if is_recording:
            # Record global tracks
            for track_row in global_result:
                cov = track_row[3]
                cov_trace = np.trace(np.array(cov)) if isinstance(cov, list) else 0.0
                frame.global_tracks.append(TrackRecord(
                    track_id=track_row[0],
                    x=track_row[1],
                    y=track_row[2],
                    cov_trace=float(cov_trace),
                    dx=track_row[4],
                    dy=track_row[5]
                ))
            
            frame_data_list.append(frame)
    
    # Cleanup
    traci_conn.close()
    
    print(f"\n{'=' * 70}")
    print(f"Recorded {len(frame_data_list)} frames")
    print(f"{'=' * 70}")
    
    # Save to JSON
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(script_dir, '..', 'tests', 'data', 
                                    f'diagnostic_penetration_{int(cav_probability*100)}pct_{timestamp}.json')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    output_data = {
        'metadata': {
            'cav_probability': cav_probability,
            'fast_forward_steps': fast_forward_steps,
            'warmup_steps': warmup_steps,
            'record_steps': record_steps,
            'timestamp': datetime.now().isoformat()
        },
        'frames': [f.to_dict() for f in frame_data_list]
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved data to: {output_file}")
    print()
    
    # Analyze the data
    analyze_frame_data(frame_data_list, gt_history)
    
    return frame_data_list, output_file


def analyze_frame_data(frames: List[FrameData], gt_history: Dict[str, List[Tuple[float, float, float]]]):
    """Analyze captured frame data for anomalies."""
    
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    if not frames:
        print("No frames to analyze!")
        return
    
    # 1. Basic statistics
    print("\n1. BASIC STATISTICS")
    print("-" * 50)
    
    avg_vehicles = np.mean([f.num_vehicles for f in frames])
    avg_cavs = np.mean([f.num_cavs for f in frames])
    avg_detections = np.mean([len(f.detections) for f in frames])
    avg_tracks = np.mean([len(f.global_tracks) for f in frames])
    
    print(f"  Average vehicles per frame: {avg_vehicles:.1f}")
    print(f"  Average CAVs per frame: {avg_cavs:.1f}")
    print(f"  Average detections per frame: {avg_detections:.1f}")
    print(f"  Average global tracks per frame: {avg_tracks:.1f}")
    print(f"  Detections per CAV: {avg_detections / max(avg_cavs, 1):.1f}")
    
    # 2. Check for ground truth jumps
    print("\n2. GROUND TRUTH JUMPS (vehicles teleporting)")
    print("-" * 50)
    jump_threshold = 5.0  # meters
    jumps_found = []
    
    for vid, history in gt_history.items():
        if len(history) < 2:
            continue
        for i in range(1, len(history)):
            x1, y1, t1 = history[i-1]
            x2, y2, t2 = history[i]
            dt = t2 - t1
            if dt > 0:
                dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                speed = dist / dt
                if dist > jump_threshold:
                    jumps_found.append({
                        'vehicle': vid,
                        'time': t2,
                        'distance': dist,
                        'speed': speed,
                        'from': (x1, y1),
                        'to': (x2, y2)
                    })
    
    if jumps_found:
        print(f"  Found {len(jumps_found)} jumps (>{jump_threshold}m in one step):")
        for j in jumps_found[:10]:
            print(f"    {j['vehicle']} at t={j['time']:.1f}s: {j['distance']:.1f}m ({j['speed']:.1f} m/s)")
        if len(jumps_found) > 10:
            print(f"    ... and {len(jumps_found) - 10} more")
    else:
        print("  No significant jumps detected")
    
    # 3. Track count vs GT count
    print("\n3. TRACK COUNT VS GROUND TRUTH")
    print("-" * 50)
    
    track_counts = [len(f.global_tracks) for f in frames]
    gt_counts = [len(f.ground_truth) for f in frames]
    
    print(f"  GT vehicles: min={min(gt_counts)}, max={max(gt_counts)}, avg={np.mean(gt_counts):.1f}")
    print(f"  Global tracks: min={min(track_counts)}, max={max(track_counts)}, avg={np.mean(track_counts):.1f}")
    
    # Track-to-GT ratio
    ratios = [t / g if g > 0 else 0 for t, g in zip(track_counts, gt_counts)]
    print(f"  Track/GT ratio: min={min(ratios):.2f}, max={max(ratios):.2f}, avg={np.mean(ratios):.2f}")
    
    if np.mean(ratios) > 1.2:
        print(f"  WARNING: Avg ratio > 1.2 suggests track duplication")
    elif np.mean(ratios) < 0.8:
        print(f"  WARNING: Avg ratio < 0.8 suggests tracks not being created")
    
    # 4. Self-localization check (by position proximity)
    print("\n4. SELF-LOCALIZATION")
    print("-" * 50)
    
    self_loc_counts = []
    for f in frames:
        # Build GT position lookup for this frame
        gt_positions = {gt.vehicle_id: (gt.x, gt.y) for gt in f.ground_truth}
        
        # Count detections within 1m of source CAV position
        count = 0
        for d in f.detections:
            if d.source_cav in gt_positions:
                cav_x, cav_y = gt_positions[d.source_cav]
                dist = math.sqrt((d.x - cav_x)**2 + (d.y - cav_y)**2)
                if dist < 1.0:
                    count += 1
        self_loc_counts.append(count)
    
    print(f"  Self-localizations per frame: min={min(self_loc_counts)}, max={max(self_loc_counts)}, avg={np.mean(self_loc_counts):.1f}")
    print(f"  Expected (num CAVs): ~{avg_cavs:.1f}")
    
    if np.mean(self_loc_counts) < avg_cavs * 0.9:
        print(f"  WARNING: Not all CAVs sending self-localization")
    else:
        print(f"  OK: Self-localization working correctly")
    
    # 5. Track ID stability
    print("\n5. TRACK ID STABILITY")
    print("-" * 50)
    
    track_ids_per_frame = [set(t.track_id for t in f.global_tracks) for f in frames]
    
    if len(track_ids_per_frame) > 1:
        new_count = 0
        lost_count = 0
        for i in range(1, len(track_ids_per_frame)):
            new = track_ids_per_frame[i] - track_ids_per_frame[i-1]
            lost = track_ids_per_frame[i-1] - track_ids_per_frame[i]
            new_count += len(new)
            lost_count += len(lost)
        
        print(f"  Total new track IDs appearing: {new_count}")
        print(f"  Total track IDs lost: {lost_count}")
        print(f"  Churn rate: {(new_count + lost_count) / len(frames):.2f} per frame")
    
    # 6. Detection accuracy sample
    print("\n6. DETECTION ACCURACY (sample)")
    print("-" * 50)
    
    sample_frames = frames[::max(1, len(frames)//5)][:5]
    
    for f in sample_frames:
        gt_positions = {gt.vehicle_id: (gt.x, gt.y) for gt in f.ground_truth}
        
        errors = []
        for d in f.detections:
            if d.target_vehicle_id in gt_positions:
                gx, gy = gt_positions[d.target_vehicle_id]
                err = math.sqrt((d.x - gx)**2 + (d.y - gy)**2)
                errors.append(err)
        
        if errors:
            print(f"  Step {f.step}: {len(errors)} matched, error: mean={np.mean(errors):.2f}m, max={max(errors):.2f}m")
    
    # 7. Track accuracy sample
    print("\n7. TRACK ACCURACY (global fusion vs GT)")
    print("-" * 50)
    
    for f in sample_frames:
        gt_positions = {gt.vehicle_id: (gt.x, gt.y) for gt in f.ground_truth}
        
        # For each track, find nearest GT
        track_errors = []
        for t in f.global_tracks:
            min_err = float('inf')
            for gx, gy in gt_positions.values():
                err = math.sqrt((t.x - gx)**2 + (t.y - gy)**2)
                min_err = min(min_err, err)
            if min_err < 50:  # Reasonable match
                track_errors.append(min_err)
        
        if track_errors:
            print(f"  Step {f.step}: {len(track_errors)} tracks matched, error: mean={np.mean(track_errors):.2f}m, max={max(track_errors):.2f}m")
    
    # 8. Covariance analysis
    print("\n8. COVARIANCE ANALYSIS")
    print("-" * 50)
    
    det_cov_traces = [d.cov_trace for f in frames for d in f.detections if d.cov_trace > 0]
    track_cov_traces = [t.cov_trace for f in frames for t in f.global_tracks if t.cov_trace > 0]
    
    if det_cov_traces:
        print(f"  Detection cov trace: min={min(det_cov_traces):.4f}, max={max(det_cov_traces):.4f}, mean={np.mean(det_cov_traces):.4f}")
    if track_cov_traces:
        print(f"  Track cov trace: min={min(track_cov_traces):.4f}, max={max(track_cov_traces):.4f}, mean={np.mean(track_cov_traces):.4f}")
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Diagnose high penetration fusion issues")
    parser.add_argument("--ff", type=int, default=1000, help="Fast-forward steps (default: 1000)")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup steps (default: 100)")
    parser.add_argument("--record", type=int, default=100, help="Recording steps (default: 100)")
    parser.add_argument("--penetration", type=float, default=1.0, help="CAV penetration rate 0-1 (default: 1.0)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    
    args = parser.parse_args()
    
    run_diagnostic(
        fast_forward_steps=args.ff,
        warmup_steps=args.warmup,
        record_steps=args.record,
        cav_probability=args.penetration,
        output_file=args.output
    )
