#!/usr/bin/env python3
"""
Test Runner for Training API
Runs all unit tests and integration tests for the training pipeline
"""

import unittest
import sys
import os
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent
src_path = project_root / 'src'
sys.path.insert(0, str(src_path))

def run_all_tests():
    """Run all tests and return results"""
    
    # Discover and run all tests in the tests directory
    test_dir = project_root / 'tests'
    loader = unittest.TestLoader()
    
    # Load all test modules
    test_suite = loader.discover(str(test_dir), pattern='test_*.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    return result

def run_specific_test(test_module):
    """Run a specific test module"""
    test_dir = project_root / 'tests'
    sys.path.insert(0, str(test_dir))
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_module)
    
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    return result

def main():
    """Main test runner"""
    print("=" * 80)
    print("TRAINING API TEST SUITE")
    print("=" * 80)
    print()
    
    if len(sys.argv) > 1:
        # Run specific test
        test_module = sys.argv[1]
        print(f"Running specific test: {test_module}")
        result = run_specific_test(test_module)
    else:
        # Run all tests
        print("Running all tests...")
        result = run_all_tests()
    
    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, trace in result.failures:
            print(f"  {test}: {trace.split(chr(10))[-2] if chr(10) in trace else trace}")
    
    if result.errors:
        print("\nERRORS:")
        for test, trace in result.errors:
            print(f"  {test}: {trace.split(chr(10))[-2] if chr(10) in trace else trace}")
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)