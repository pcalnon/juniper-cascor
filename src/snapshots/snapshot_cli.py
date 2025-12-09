#!/usr/bin/env python
"""
Command-line interface for HDF5 network snapshot management.
Provides tools for saving, loading, and managing network snapshots.
"""

import argparse
import os
import sys

# Add parent directories for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .snapshot_serializer import CascadeHDF5Serializer
from .snapshot_utils import HDF5Utils


def save_network_snapshot(
    network_file: str,
    output_file: str,
    include_training: bool = False,
) -> bool:
    """Save a network snapshot to HDF5."""
    print(f"Saving network snapshot to {output_file}...")

    try:
        return _object_to_file(output_file, include_training)
    except Exception as e:
        print(f"✗ Error saving snapshot: {e}")
        return False


def _object_to_file(output_file, include_training) -> bool:
    # For demonstration, create a simple network. In practice, you would load an existing network
    from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
    from cascade_correlation_config.cascade_correlation_config import CascadeCorrelationConfig

    # Create a simple network for demonstration
    config = CascadeCorrelationConfig(input_size=2, output_size=1)
    network = CascadeCorrelationNetwork(config=config)

    if success := network.save_to_hdf5(
            filepath=output_file,
            include_training_state=include_training,
            include_training_data=False,
        ):
        return _get_saved_snapshot_file_data(output_file, success)
    print("✗ Failed to save network snapshot")
    return False


def _get_saved_snapshot_file_data(output_file, success) -> bool:
    print(f"✓ Successfully saved network snapshot to {output_file}, success: {success}")

    # Show file info
    info = HDF5Utils.get_file_info(output_file)
    print(f"  File size: {info.get('size_mb', 0):.2f} MB")
    print(f"  Groups: {len(info.get('groups', []))}")
    print(f"  Datasets: {len(info.get('datasets', []))}")
    return True


def load_network_snapshot(snapshot_file: str, output_file: str = None):
    """Load a network from HDF5 snapshot."""
    print(f"Loading network snapshot from {snapshot_file}...")

    try:
        # Verify file first
        serializer = CascadeHDF5Serializer()
        verification = serializer.verify_saved_network(snapshot_file)

        if not verification.get('valid', False):
            print(f"✗ Invalid snapshot file: {verification.get('error', 'Unknown error')}")
            return False

        print("✓ Snapshot file verification passed")
        print(f"  Network UUID: {verification.get('network_uuid', 'unknown')}")
        print(f"  Architecture: {verification.get('input_size', 0)} → {verification.get('num_hidden_units', 0)} → {verification.get('output_size', 0)}")
        print(f"  Created: {verification.get('created', 'unknown')}")
        print(f"  Format: {verification.get('format', 'unknown')} v{verification.get('format_version', 'unknown')}")

        # Load the network
        from cascade_correlation.cascade_correlation import CascadeCorrelationNetwork
        if network := CascadeCorrelationNetwork.load_from_hdf5(snapshot_file):
            print("✓ Successfully loaded network from snapshot")

            # Show network info
            print(f"  Input size: {network.input_size}")
            print(f"  Output size: {network.output_size}")
            print(f"  Hidden units: {len(network.hidden_units)}")
            print(f"  Activation function: {network.activation_function_name}")

            # Save in different format if requested
            if output_file:
                print(f"Saving loaded network to {output_file}...")
                if network.save_to_hdf5(output_file):
                    print(f"✓ Network saved to {output_file}")
                else:
                    print(f"✗ Failed to save network to {output_file}")

            return True
        else:
            print("✗ Failed to load network from snapshot")
            return False

    except Exception as e:
        print(f"✗ Error loading snapshot: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_snapshots(directory: str):
    """List all network snapshots in a directory."""
    print(f"Scanning directory: {directory}")
    
    try:
        networks = HDF5Utils.list_networks_in_directory(directory)
        
        if not networks:
            print("No valid network snapshots found")
            return
        
        print(f"\nFound {len(networks)} network snapshots:")
        print("-" * 100)
        print(f"{'Filename':<30} {'Size (MB)':<10} {'Architecture':<15} {'Hidden':<8} {'Format':<10} {'Created':<20}")
        print("-" * 100)
        
        for network in sorted(networks, key=lambda x: x.get('created', '')):
            filename = network.get('filename', 'unknown')[:28]
            size_mb = network.get('file_size', 0) / (1024 * 1024)
            arch = f"{network.get('input_size', 0)}→{network.get('output_size', 0)}"
            hidden_units = network.get('num_hidden_units', 0)
            format_ver = network.get('format_version', 'unknown')
            created = str(network.get('created', 'unknown'))[:18]
            
            print(f"{filename:<30} {size_mb:<10.2f} {arch:<15} {hidden_units:<8} {format_ver:<10} {created:<20}")
        
        print("-" * 100)
        
    except Exception as e:
        print(f"✗ Error listing snapshots: {e}")


def verify_snapshot(filepath: str):
    """Verify a network snapshot file."""
    print(f"Verifying snapshot: {filepath}")
    
    try:
        serializer = CascadeHDF5Serializer()
        verification = serializer.verify_saved_network(filepath)
        
        if verification.get('valid', False):
            print("✓ File is a valid network snapshot")
            print(f"  Format: {verification.get('format', 'unknown')} v{verification.get('format_version', 'unknown')}")
            print(f"  Network UUID: {verification.get('network_uuid', 'unknown')}")
            print(f"  Architecture: {verification.get('input_size', 0)} → {verification.get('num_hidden_units', 0)} → {verification.get('output_size', 0)}")
            print(f"  Activation: {verification.get('activation_function', 'unknown')}")
            print(f"  File size: {verification.get('file_size', 0) / (1024 * 1024):.2f} MB")
            print(f"  Created: {verification.get('created', 'unknown')}")
            
            # Check for optional sections
            optional_sections = ['has_history', 'has_mp', 'has_data']
            for section in optional_sections:
                if verification.get(section, False):
                    section_name = section.replace('has_', '').replace('mp', 'multiprocessing')
                    print(f"  ✓ Contains {section_name}")
            
            return True
        else:
            print(f"✗ Invalid snapshot file: {verification.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"✗ Error verifying snapshot: {e}")
        return False


def compare_snapshots(filepath1: str, filepath2: str):
    """Compare two network snapshots."""
    print("Comparing snapshots:")
    print(f"  File 1: {filepath1}")
    print(f"  File 2: {filepath2}")
    
    try:
        comparison = HDF5Utils.compare_networks(filepath1, filepath2)
        
        if not comparison.get('comparable', False):
            print(f"✗ Cannot compare files: {comparison.get('error', 'Unknown error')}")
            return False
        
        print("\nComparison results:")
        print(f"  Same architecture: {'✓' if comparison['same_architecture'] else '✗'}")
        print(f"  Same hidden units: {'✓' if comparison['same_hidden_units'] else '✗'}")
        print(f"  Same activation: {'✓' if comparison['same_activation'] else '✗'}")
        
        if not comparison['same_architecture']:
            diff = comparison['architecture_diff']
            print("\nArchitecture differences:")
            print(f"  Input size: {diff['input_size'][0]} vs {diff['input_size'][1]}")
            print(f"  Output size: {diff['output_size'][0]} vs {diff['output_size'][1]}")
            print(f"  Hidden units: {diff['num_hidden_units'][0]} vs {diff['num_hidden_units'][1]}")
            print(f"  Activation: {diff['activation_function'][0]} vs {diff['activation_function'][1]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error comparing snapshots: {e}")
        return False


def cleanup_old_snapshots(directory: str, keep_count: int = 10):
    """Clean up old snapshot files."""
    print(f"Cleaning up old snapshots in {directory} (keeping {keep_count} most recent)")
    
    try:
        deleted_count = HDF5Utils.cleanup_old_files(directory, keep_count)
        if deleted_count > 0:
            print(f"✓ Deleted {deleted_count} old snapshot files")
        else:
            print("No old files to delete")
        return True
        
    except Exception as e:
        print(f"✗ Error cleaning up files: {e}")
        return False


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="HDF5 Network Snapshot Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s save network.pkl snapshot.h5
    %(prog)s load snapshot.h5
    %(prog)s list ./snapshots/
    %(prog)s verify snapshot.h5
    %(prog)s compare snapshot1.h5 snapshot2.h5
    %(prog)s cleanup ./snapshots/ --keep 5
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Save command
    save_parser = subparsers.add_parser('save', help='Save network to HDF5 snapshot')
    save_parser.add_argument('network_file', help='Input network file')
    save_parser.add_argument('output_file', help='Output HDF5 file')
    save_parser.add_argument('--include-training', action='store_true', help='Include training history')
    
    # Load command
    load_parser = subparsers.add_parser('load', help='Load network from HDF5 snapshot')
    load_parser.add_argument('snapshot_file', help='HDF5 snapshot file')
    load_parser.add_argument('--output', help='Save loaded network to file')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List snapshots in directory')
    list_parser.add_argument('directory', help='Directory to scan')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify snapshot file')
    verify_parser.add_argument('filepath', help='Snapshot file to verify')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two snapshots')
    compare_parser.add_argument('file1', help='First snapshot file')
    compare_parser.add_argument('file2', help='Second snapshot file')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old snapshots')
    cleanup_parser.add_argument('directory', help='Directory to clean')
    cleanup_parser.add_argument('--keep', type=int, default=10, help='Number of recent files to keep (default: 10)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute commands
    success = False
    try:
        if args.command == 'save':
            success = save_network_snapshot(args.network_file, args.output_file, args.include_training)
        elif args.command == 'load':
            success = load_network_snapshot(args.snapshot_file, args.output)
        elif args.command == 'list':
            list_snapshots(args.directory)
            success = True
        elif args.command == 'verify':
            success = verify_snapshot(args.filepath)
        elif args.command == 'compare':
            success = compare_snapshots(args.file1, args.file2)
        elif args.command == 'cleanup':
            success = cleanup_old_snapshots(args.directory, args.keep)
        else:
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return 1
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
