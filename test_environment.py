#!/usr/bin/env python3
"""
Environment Test Script
Tests the current environment and available functionality
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_basic_imports():
    """Test basic Python imports"""
    print("🧪 Testing basic Python functionality...")
    
    tests = [
        ("sys", "System module"),
        ("os", "Operating system module"),
        ("pathlib", "Path handling"),
        ("json", "JSON support"),
        ("time", "Time functions")
    ]
    
    passed = 0
    for module, description in tests:
        try:
            __import__(module)
            print(f"  ✅ {module}: {description}")
            passed += 1
        except ImportError:
            print(f"  ❌ {module}: {description}")
    
    return passed, len(tests)

def test_project_structure():
    """Test project structure"""
    print("\n📁 Testing project structure...")
    
    required_files = [
        ("src/", "Source code directory"),
        ("config/settings.py", "Configuration file"),
        ("main.py", "Main CLI"),
        ("README.md", "Documentation"),
        ("requirements.txt", "Dependencies list")
    ]
    
    passed = 0
    for file_path, description in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ {file_path}: {description}")
            passed += 1
        else:
            print(f"  ❌ {file_path}: {description}")
    
    return passed, len(required_files)

def test_config_import():
    """Test configuration import"""
    print("\n⚙️ Testing configuration...")
    
    try:
        from config.settings import VIDEO_SETTINGS, AI_SETTINGS, DEPENDENCIES, ENVIRONMENT_STATUS
        print(f"  ✅ Configuration imported successfully")
        
        # Show dependency status
        print(f"  📊 Dependency Status:")
        for dep, available in DEPENDENCIES.items():
            status = "✅" if available else "❌"
            print(f"    {status} {dep}: {'Available' if available else 'Missing'}")
        
        # Show environment status
        print(f"  🔍 Environment Status:")
        for feature, available in ENVIRONMENT_STATUS.items():
            if feature != "dependencies":
                status = "✅" if available else "❌"
                print(f"    {status} {feature}: {'Available' if available else 'Not Available'}")
        
        return True, DEPENDENCIES, ENVIRONMENT_STATUS
        
    except ImportError as e:
        print(f"  ❌ Configuration import failed: {e}")
        return False, {}, {}

def test_core_modules():
    """Test core module imports"""
    print("\n🔧 Testing core modules...")
    
    modules = [
        ("modules.video_processor", "Video processing"),
        ("modules.ai_processor", "AI processing"),
        ("modules.video_builder", "Video building"),
        ("upscale_app", "Main application")
    ]
    
    passed = 0
    errors = []
    
    for module, description in modules:
        try:
            __import__(module)
            print(f"  ✅ {module}: {description}")
            passed += 1
        except ImportError as e:
            print(f"  ❌ {module}: {description} - {str(e)}")
            errors.append((module, str(e)))
    
    return passed, len(modules), errors

def test_functionality():
    """Test basic functionality"""
    print("\n🎯 Testing basic functionality...")
    
    try:
        # Test VideoProcessor
        from modules.video_processor import VideoProcessor
        processor = VideoProcessor()
        print("  ✅ VideoProcessor initialization")
        
        # Test upscaling calculation
        new_width, new_height = processor.get_upscaled_resolution(640, 480, 1.5)
        if new_width == 960 and new_height == 720:
            print("  ✅ Resolution calculation")
        else:
            print(f"  ❌ Resolution calculation: got {new_width}x{new_height}, expected 960x720")
        
        # Test SimpleUpscaler (always available)
        from modules.ai_processor import SimpleUpscaler
        upscaler = SimpleUpscaler()
        print("  ✅ SimpleUpscaler initialization")
        
        # Test main app initialization
        from upscale_app import UpScaleApp
        app = UpScaleApp(use_ai=False)
        system_info = app.get_system_info()
        print("  ✅ UpScaleApp initialization")
        print(f"    Platform: {system_info.get('platform', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Functionality test failed: {e}")
        return False

def test_cli_availability():
    """Test CLI script availability"""
    print("\n🖥️ Testing CLI availability...")
    
    cli_files = [
        ("main.py", "Basic CLI"),
        ("main_enhanced.py", "Enhanced CLI"),
        ("setup_environment.py", "Environment setup"),
        ("install_dependencies.py", "Dependency installer")
    ]
    
    passed = 0
    for file_path, description in cli_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ {file_path}: {description}")
            passed += 1
        else:
            print(f"  ❌ {file_path}: {description}")
    
    return passed, len(cli_files)

def generate_report(results):
    """Generate test report"""
    print("\n" + "=" * 60)
    print("📋 ENVIRONMENT TEST REPORT")
    print("=" * 60)
    
    basic_passed, basic_total = results['basic_imports']
    structure_passed, structure_total = results['structure']
    config_success, dependencies, env_status = results['config']
    modules_passed, modules_total, module_errors = results['modules']
    functionality_success = results['functionality']
    cli_passed, cli_total = results['cli']
    
    # Calculate overall score
    total_tests = basic_total + structure_total + modules_total + cli_total + 2  # +2 for config and functionality
    passed_tests = basic_passed + structure_passed + modules_passed + cli_passed
    if config_success:
        passed_tests += 1
    if functionality_success:
        passed_tests += 1
    
    score = (passed_tests / total_tests) * 100
    
    print(f"📊 Overall Score: {score:.1f}% ({passed_tests}/{total_tests} tests passed)")
    print()
    
    # Status indicators
    if score >= 80:
        print("🎉 Environment Status: EXCELLENT")
        print("   Full functionality should be available")
    elif score >= 60:
        print("✅ Environment Status: GOOD")
        print("   Basic functionality should work")
    elif score >= 40:
        print("⚠️  Environment Status: LIMITED")
        print("   Some functionality may be missing")
    else:
        print("❌ Environment Status: POOR")
        print("   Significant issues detected")
    
    print()
    
    # Feature availability
    print("🔍 Feature Availability:")
    if env_status:
        for feature, available in env_status.items():
            if feature != "dependencies":
                status = "✅" if available else "❌"
                print(f"  {status} {feature.replace('_', ' ').title()}")
    
    print()
    
    # Recommendations
    print("💡 Recommendations:")
    
    if not config_success:
        print("  - Fix configuration import errors")
    
    if module_errors:
        print("  - Install missing dependencies:")
        for module, error in module_errors[:3]:  # Show first 3 errors
            print(f"    * {module}: {error}")
        if len(module_errors) > 3:
            print(f"    * ... and {len(module_errors) - 3} more")
    
    if score < 60:
        print("  - Run: python install_dependencies.py")
        print("  - Run: python setup_environment.py")
    
    print("  - Check README.md for setup instructions")
    print("  - Try basic commands: python main.py system")
    
    print("=" * 60)

def main():
    """Main test function"""
    print("🎬 UpScale App - Environment Test")
    print("=" * 60)
    
    # Run all tests
    results = {
        'basic_imports': test_basic_imports(),
        'structure': test_project_structure(),
        'config': test_config_import(),
        'modules': test_core_modules(),
        'functionality': test_functionality(),
        'cli': test_cli_availability()
    }
    
    # Generate report
    generate_report(results)
    
    # Determine exit code
    _, _, env_status = results['config']
    if env_status.get('basic_functionality_available', False):
        return 0  # Success
    else:
        return 1  # Issues detected

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)