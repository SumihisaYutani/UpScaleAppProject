#!/usr/bin/env python3
"""
Issue Diagnosis Tool
Identifies specific problems causing POOR environment status
"""

import sys
import os
from pathlib import Path
import traceback

def diagnose_python_environment():
    """Diagnose Python environment issues"""
    print("🔍 Python Environment Diagnosis")
    print("-" * 40)
    
    issues = []
    
    # Test 1: Basic Python functionality
    try:
        import sys, os, pathlib, json, time
        print("✅ Basic Python modules working")
    except ImportError as e:
        issue = f"❌ CRITICAL: Basic Python modules failed - {e}"
        print(issue)
        issues.append(("CRITICAL", "Python Environment", issue))
    
    # Test 2: Python version
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        issue = f"❌ CRITICAL: Python version too old ({version.major}.{version.minor}), requires 3.8+"
        print(issue)
        issues.append(("CRITICAL", "Python Version", issue))
    else:
        print(f"✅ Python version OK ({version.major}.{version.minor})")
    
    # Test 3: Working directory
    try:
        cwd = Path.cwd()
        print(f"✅ Working directory: {cwd}")
    except Exception as e:
        issue = f"❌ CRITICAL: Cannot access working directory - {e}"
        print(issue)
        issues.append(("CRITICAL", "File System", issue))
    
    return issues

def diagnose_project_structure():
    """Diagnose project structure issues"""
    print("\n📁 Project Structure Diagnosis")  
    print("-" * 40)
    
    issues = []
    critical_files = {
        "src/": "Source code directory",
        "config/": "Configuration directory", 
        "config/settings.py": "Main configuration file",
        "main.py": "Primary CLI script",
        "README.md": "Documentation"
    }
    
    missing_critical = []
    for file_path, description in critical_files.items():
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path} - {description}")
        else:
            issue = f"❌ MISSING: {file_path} - {description}"
            print(issue)
            missing_critical.append(file_path)
            issues.append(("SEVERE", "Project Structure", issue))
    
    # Critical assessment
    if len(missing_critical) >= 3:
        issue = "❌ CRITICAL: Project structure severely corrupted"
        print(issue)
        issues.append(("CRITICAL", "Project Integrity", issue))
    
    return issues

def diagnose_configuration():
    """Diagnose configuration import issues"""
    print("\n⚙️ Configuration Diagnosis")
    print("-" * 40)
    
    issues = []
    
    # Add src to path
    try:
        sys.path.append(str(Path("src")))
        print("✅ Added src to Python path")
    except Exception as e:
        issue = f"❌ CRITICAL: Cannot modify Python path - {e}"
        print(issue)
        issues.append(("CRITICAL", "Python Path", issue))
        return issues
    
    # Test configuration import
    try:
        from config.settings import VIDEO_SETTINGS, AI_SETTINGS
        print("✅ Basic configuration import successful")
    except ImportError as e:
        issue = f"❌ SEVERE: Configuration import failed - {e}"
        print(issue)
        issues.append(("SEVERE", "Configuration", issue))
        return issues
    except Exception as e:
        issue = f"❌ CRITICAL: Configuration error - {e}"
        print(issue)
        issues.append(("CRITICAL", "Configuration", issue))
        return issues
    
    # Test advanced configuration
    try:
        from config.settings import DEPENDENCIES, ENVIRONMENT_STATUS
        print("✅ Advanced configuration import successful")
        
        # Show dependency status
        available_deps = sum(1 for dep, status in DEPENDENCIES.items() if status)
        total_deps = len(DEPENDENCIES)
        print(f"📊 Dependencies available: {available_deps}/{total_deps}")
        
        if available_deps == 0:
            issue = "❌ SEVERE: No dependencies available"
            print(issue)  
            issues.append(("SEVERE", "Dependencies", issue))
        elif available_deps < 3:
            issue = f"⚠️ WARNING: Limited dependencies ({available_deps}/{total_deps})"
            print(issue)
            issues.append(("WARNING", "Dependencies", issue))
            
    except ImportError as e:
        issue = f"⚠️ WARNING: Advanced configuration not available - {e}"
        print(issue)
        issues.append(("WARNING", "Configuration", issue))
    
    return issues

def diagnose_modules():
    """Diagnose module import issues"""
    print("\n🔧 Module Import Diagnosis")
    print("-" * 40)
    
    issues = []
    modules = [
        ("modules.video_processor", "Video processing", "SEVERE"),
        ("modules.ai_processor", "AI processing", "MODERATE"),
        ("modules.video_builder", "Video building", "SEVERE"), 
        ("upscale_app", "Main application", "CRITICAL")
    ]
    
    failed_critical = 0
    failed_severe = 0
    
    for module, description, severity in modules:
        try:
            __import__(module)
            print(f"✅ {module} - {description}")
        except ImportError as e:
            issue = f"❌ {severity}: {module} failed - {e}"
            print(issue)
            issues.append((severity, "Module Import", issue))
            
            if severity == "CRITICAL":
                failed_critical += 1
            elif severity == "SEVERE":
                failed_severe += 1
    
    # Overall module assessment
    if failed_critical > 0:
        issue = f"❌ CRITICAL: {failed_critical} critical modules failed"
        print(issue)
        issues.append(("CRITICAL", "Module System", issue))
    elif failed_severe >= 2:
        issue = f"❌ SEVERE: {failed_severe} severe modules failed"  
        print(issue)
        issues.append(("SEVERE", "Module System", issue))
    
    return issues

def diagnose_permissions():
    """Diagnose file system permission issues"""
    print("\n🔐 Permissions Diagnosis")
    print("-" * 40)
    
    issues = []
    
    # Test directory creation
    try:
        test_dir = Path("temp_test_dir")
        test_dir.mkdir(exist_ok=True)
        test_dir.rmdir()
        print("✅ Directory creation permissions OK")
    except Exception as e:
        issue = f"❌ SEVERE: Cannot create directories - {e}"
        print(issue)
        issues.append(("SEVERE", "Permissions", issue))
    
    # Test file writing
    try:
        test_file = Path("temp_test_file.txt")
        test_file.write_text("test")
        test_file.unlink()
        print("✅ File write permissions OK")
    except Exception as e:
        issue = f"❌ SEVERE: Cannot write files - {e}"
        print(issue)
        issues.append(("SEVERE", "Permissions", issue))
    
    return issues

def calculate_severity_score(all_issues):
    """Calculate overall severity score"""
    severity_weights = {
        "CRITICAL": -30,
        "SEVERE": -20, 
        "MODERATE": -10,
        "WARNING": -5
    }
    
    base_score = 100
    for severity, category, issue in all_issues:
        base_score += severity_weights.get(severity, 0)
    
    return max(0, base_score)

def provide_specific_solutions(all_issues):
    """Provide specific solutions for detected issues"""
    print("\n💡 Specific Solutions")
    print("-" * 40)
    
    # Group issues by category
    issue_categories = {}
    for severity, category, issue in all_issues:
        if category not in issue_categories:
            issue_categories[category] = []
        issue_categories[category].append((severity, issue))
    
    if "Python Environment" in issue_categories:
        print("🐍 Python Environment Issues:")
        print("  Solution: Reinstall Python 3.8+ from https://python.org")
        print("  Make sure to check 'Add Python to PATH' during installation")
        print("")
    
    if "Project Structure" in issue_categories or "Project Integrity" in issue_categories:
        print("📁 Project Structure Issues:")
        print("  Solution: Re-clone the project")
        print("  git clone https://github.com/SumihisaYutani/UpScaleAppProject.git")
        print("")
    
    if "Configuration" in issue_categories:
        print("⚙️ Configuration Issues:")
        print("  Solution: Check file paths and run setup")
        print("  python setup_environment.py")
        print("")
    
    if "Dependencies" in issue_categories:
        print("📦 Dependency Issues:")
        print("  Solution: Install dependencies step by step")
        print("  python install_dependencies.py")
        print("")
    
    if "Module Import" in issue_categories or "Module System" in issue_categories:
        print("🔧 Module Import Issues:")
        print("  Solution: Check Python path and install missing packages")
        print("  python -m pip install -r requirements_minimal.txt")
        print("")
    
    if "Permissions" in issue_categories:
        print("🔐 Permission Issues:")
        print("  Windows: Run as Administrator")
        print("  Linux/macOS: sudo chown -R $USER:$USER .")
        print("")

def main():
    """Main diagnosis function"""
    print("🏥 UpScale App - Issue Diagnosis Tool")
    print("=" * 60)
    
    all_issues = []
    
    try:
        # Run all diagnostic tests
        all_issues.extend(diagnose_python_environment())
        all_issues.extend(diagnose_project_structure())
        all_issues.extend(diagnose_configuration())
        all_issues.extend(diagnose_modules())
        all_issues.extend(diagnose_permissions())
        
    except Exception as e:
        print(f"\n💥 Diagnosis tool error: {e}")
        traceback.print_exc()
        all_issues.append(("CRITICAL", "Diagnosis Tool", f"Tool failure: {e}"))
    
    # Calculate and display results
    print("\n" + "=" * 60)
    print("📊 DIAGNOSIS RESULTS")
    print("=" * 60)
    
    if not all_issues:
        print("🎉 No issues detected! Environment should be working well.")
        return 0
    
    # Categorize issues
    critical_issues = [i for i in all_issues if i[0] == "CRITICAL"]
    severe_issues = [i for i in all_issues if i[0] == "SEVERE"] 
    moderate_issues = [i for i in all_issues if i[0] == "MODERATE"]
    warnings = [i for i in all_issues if i[0] == "WARNING"]
    
    print(f"❌ Critical Issues: {len(critical_issues)}")
    print(f"🟠 Severe Issues: {len(severe_issues)}")
    print(f"🟡 Moderate Issues: {len(moderate_issues)}")
    print(f"⚠️  Warnings: {len(warnings)}")
    
    # Calculate severity
    score = calculate_severity_score(all_issues)
    print(f"\n📈 Environment Health Score: {score}/100")
    
    if score < 40:
        print("💀 Status: POOR - Critical intervention required")
    elif score < 60:
        print("⚠️  Status: LIMITED - Significant issues present")
    elif score < 80:
        print("✅ Status: GOOD - Minor issues present")
    else:
        print("🎉 Status: EXCELLENT")
    
    # Show most critical issues
    if critical_issues:
        print(f"\n🚨 CRITICAL ISSUES (Must fix immediately):")
        for _, category, issue in critical_issues[:3]:
            print(f"  • {issue}")
    
    if severe_issues:
        print(f"\n🟠 SEVERE ISSUES:")
        for _, category, issue in severe_issues[:3]:
            print(f"  • {issue}")
    
    # Provide solutions
    provide_specific_solutions(all_issues)
    
    # Next steps
    print("📋 Recommended Next Steps:")
    if critical_issues:
        print("  1. Address CRITICAL issues first")
        print("  2. Restart diagnosis after each fix")
        print("  3. Consider complete reinstallation")
    else:
        print("  1. Run: python install_dependencies.py")
        print("  2. Run: python test_environment.py")
        print("  3. Try: python main.py system")
    
    return 1 if critical_issues or len(severe_issues) > 2 else 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n🛑 Diagnosis interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Diagnosis failed: {e}")
        sys.exit(1)