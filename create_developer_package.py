#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RideBuddy Pro v2.1.0 - Developer Package Creator
Creates a complete development package with all source code, datasets, and development tools
"""

import os
import zipfile
import json
from pathlib import Path
from datetime import datetime

def create_developer_package():
    """Create complete development package"""
    
    print("🚗 RideBuddy Pro v2.1.0 - Developer Package Creator")
    print("=" * 60)
    print("👨‍💻 Creating complete development package...")
    
    # Package information
    package_name = f"RideBuddy_Pro_v2.1.0_Developer_Complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    zip_filename = f"{package_name}.zip"
    
    # Developer files (everything except large datasets)
    developer_files = [
        # Core Application
        "ridebuddy_optimized_gui.py",
        "ridebuddy_config.ini", 
        "vehicle_config.json",
        
        # Requirements and Dependencies
        "requirements.txt",
        "requirements-minimal.txt",
        "install_dependencies.py",
        
        # Setup and Training Scripts
        "setup_and_train.bat",
        "start_vehicle_mode.bat",
        
        # Development and Training Tools
        "enhanced_dataset_trainer.py",
        "comprehensive_trainer.py", 
        "enhanced_model_integration.py",
        "deploy_enhanced_model.py",
        "model_integration.py",
        "organize_dataset.py",
        "comprehensive_dataset_downloader.py",
        
        # Validation and Diagnostic Tools
        "system_validation.py",
        "camera_diagnostics.py",
        "vehicle_camera_diagnostic.py",
        "validate_model.py",
        
        # Vehicle Deployment
        "vehicle_deployment_guide.py",
        "vehicle_launcher.py",
        
        # Pipeline and Orchestration
        "ridebuddy_pipeline_orchestrator.py",
        
        # Documentation Converters
        "convert_md_to_html.py",
        "convert_md_to_rtf.py",
        "convert_md_to_docx.py",
        
        # All Documentation (Markdown)
        "README.md",
        "INSTALLATION_AND_SETUP_GUIDE.md",
        "RIDEBUDDY_SYSTEM_DOCUMENTATION.md", 
        "RESPONSIVE_DESIGN_UPDATE.md",
        "DEPLOYMENT_READY_GUIDE.md",
        "VEHICLE_DEPLOYMENT.md",
        "QUICK_REFERENCE.md",
        "DATA_ANALYSIS_REPORT.md",
        "DATA_COLLECTION_STRATEGY.md",
        "IMPLEMENTATION_GUIDE.md",
        "IMPROVEMENTS_SUMMARY.md",
        "PRODUCTION_READINESS_ASSESSMENT.md",
        "TRAINING_COMPLETION_REPORT.md",
        "RIDEBUDDY_FINAL_SETUP_GUIDE.md",
        "CAMERA_FIX_REPORT.md",
        "DEPLOYMENT_STATUS.md",
        "VEHICLE_LAUNCHER_TROUBLESHOOTING.md",
        "DOCUMENTATION_CONVERSION_SUMMARY.md",
        
        # Formatted Documentation
        "RideBuddy_System_Documentation.html",
        "RideBuddy_System_Documentation.rtf",
        "RideBuddy_Installation_Setup_Guide.html", 
        "RideBuddy_Installation_Setup_Guide.rtf",
        
        # Database and Configuration
        "ridebuddy_data.db"
    ]
    
    # Development directories
    development_directories = [
        "src/",
        "configs/", 
        "examples/",
        "trained_models/",
        "logs/",
        "test_reports/",
        ".github/",
        ".vscode/"
    ]
    
    try:
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            
            # Add developer files
            print("\n📁 Adding development files...")
            files_added = 0
            
            for file_path in developer_files:
                if os.path.exists(file_path):
                    zipf.write(file_path, file_path)
                    print(f"   ✅ {file_path}")
                    files_added += 1
                else:
                    print(f"   ⚠️ {file_path} (not found)")
            
            # Add development directories
            print("\n📂 Adding development directories...")
            dirs_added = 0
            
            for dir_path in development_directories:
                if os.path.exists(dir_path) and os.path.isdir(dir_path):
                    for root, dirs, files in os.walk(dir_path):
                        for file in files:
                            # Skip very large files (>200MB) but include development assets
                            file_path = os.path.join(root, file)
                            if os.path.getsize(file_path) > 200 * 1024 * 1024:
                                print(f"   ⏭️ Skipping very large file: {file}")
                                continue
                                
                            arcname = file_path.replace('\\', '/')
                            zipf.write(file_path, arcname)
                    print(f"   ✅ {dir_path}")
                    dirs_added += 1
                else:
                    print(f"   ⚠️ {dir_path} (not found)")
            
            # Add selective data directory (smaller datasets only)
            print("\n📊 Adding development datasets (selective)...")
            data_dir = "data/"
            if os.path.exists(data_dir):
                for root, dirs, files in os.walk(data_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        file_size = os.path.getsize(file_path)
                        
                        # Skip very large dataset files (>100MB) 
                        if file_size > 100 * 1024 * 1024:
                            print(f"   ⏭️ Skipping large dataset: {file} ({file_size/(1024*1024):.1f}MB)")
                            continue
                        
                        # Include smaller datasets and metadata
                        arcname = file_path.replace('\\', '/')
                        zipf.write(file_path, arcname)
                        
                print(f"   ✅ data/ (selective, <100MB files)")
                dirs_added += 1
            
            # Add comprehensive datasets directory (metadata only)
            comprehensive_dir = "comprehensive_datasets/"
            if os.path.exists(comprehensive_dir):
                metadata_count = 0
                for root, dirs, files in os.walk(comprehensive_dir):
                    for file in files:
                        # Only include metadata, configs, and small files
                        if file.endswith(('.json', '.txt', '.md', '.yml', '.yaml', '.ini', '.cfg')):
                            file_path = os.path.join(root, file)
                            arcname = file_path.replace('\\', '/')
                            zipf.write(file_path, arcname)
                            metadata_count += 1
                        else:
                            file_path = os.path.join(root, file)
                            file_size = os.path.getsize(file_path)
                            if file_size < 10 * 1024 * 1024:  # Include files < 10MB
                                arcname = file_path.replace('\\', '/')
                                zipf.write(file_path, arcname)
                                metadata_count += 1
                            
                print(f"   ✅ comprehensive_datasets/ (metadata and configs, {metadata_count} files)")
            
            # Create developer manifest
            print("\n📋 Creating developer manifest...")
            
            manifest = {
                "package_info": {
                    "name": "RideBuddy Pro Developer Package",
                    "version": "2.1.0",
                    "package_type": "Complete Development Environment",
                    "created": datetime.now().isoformat(),
                    "description": "Complete development package with source code, training tools, and documentation"
                },
                "target_audience": [
                    "AI/ML developers and researchers",
                    "Computer vision engineers", 
                    "System integrators and customizers",
                    "Advanced users requiring source access",
                    "Contributors to RideBuddy development"
                ],
                "development_features": [
                    "🔬 Complete source code access",
                    "🧠 AI model training and customization tools",
                    "📊 Dataset organization and management",
                    "🔧 Advanced configuration options", 
                    "📈 Performance analysis and optimization",
                    "🚀 Deployment automation scripts",
                    "🧪 Comprehensive testing framework",
                    "📚 Complete documentation source"
                ],
                "included_development_tools": {
                    "training_pipeline": [
                        "enhanced_dataset_trainer.py - Advanced model training",
                        "comprehensive_trainer.py - Multi-dataset training", 
                        "enhanced_model_integration.py - Model deployment",
                        "ridebuddy_pipeline_orchestrator.py - Full pipeline automation"
                    ],
                    "data_management": [
                        "organize_dataset.py - Dataset organization tools",
                        "comprehensive_dataset_downloader.py - Data acquisition",
                        "validate_model.py - Model validation and testing"
                    ],
                    "deployment_tools": [
                        "deploy_enhanced_model.py - Model deployment automation", 
                        "vehicle_deployment_guide.py - Automotive deployment",
                        "system_validation.py - System compatibility checking"
                    ],
                    "documentation_tools": [
                        "convert_md_to_html.py - HTML documentation generation",
                        "convert_md_to_rtf.py - RTF document creation",
                        "convert_md_to_docx.py - Word document generation"
                    ]
                },
                "development_setup": {
                    "environment": "Python 3.8+ development environment",
                    "dependencies": "Full development requirements in requirements.txt",
                    "gpu_support": "CUDA-enabled training (optional)",
                    "ide_config": "VS Code configuration included (.vscode/)",
                    "version_control": "Git configuration and workflows (.github/)"
                },
                "source_code_structure": {
                    "main_application": "ridebuddy_optimized_gui.py - Main GUI application",
                    "core_modules": "src/ - Core AI and processing modules", 
                    "configuration": "configs/ - Configuration templates and presets",
                    "examples": "examples/ - Usage examples and tutorials",
                    "models": "trained_models/ - Pre-trained AI models",
                    "testing": "test_reports/ - Test results and validation reports"
                },
                "customization_capabilities": [
                    "AI model architecture modification",
                    "Custom alert thresholds and behaviors",
                    "GUI theme and layout customization",
                    "Vehicle-specific configuration profiles", 
                    "Fleet management integration APIs",
                    "Custom data processing pipelines",
                    "Advanced performance optimization",
                    "Multi-language interface support"
                ],
                "research_and_development": {
                    "ai_research": "Access to training data and model architectures",
                    "computer_vision": "OpenCV-based processing pipeline source",
                    "performance_optimization": "Edge computing optimization techniques",
                    "automotive_integration": "Vehicle deployment methodologies",
                    "user_interface": "Responsive design implementation patterns"
                },
                "documentation_coverage": [
                    "Complete system architecture documentation", 
                    "API reference and integration guides",
                    "Development workflow and contribution guidelines",
                    "Training data collection and processing methods",
                    "Model optimization and deployment strategies",
                    "Vehicle integration and testing procedures",
                    "Performance benchmarking and analysis tools"
                ],
                "advanced_features": {
                    "model_customization": "Train custom models for specific use cases",
                    "data_pipeline": "Automated data collection and processing",
                    "performance_tuning": "System optimization for various hardware",
                    "integration_apis": "Fleet and third-party system integration",
                    "monitoring_dashboard": "Advanced analytics and reporting",
                    "deployment_automation": "CI/CD pipeline for production updates"
                }
            }
            
            # Add manifest to ZIP
            manifest_json = json.dumps(manifest, indent=2)
            zipf.writestr("DEVELOPER_MANIFEST.json", manifest_json)
            
            # Create developer setup guide
            setup_guide = """# RideBuddy Pro v2.1.0 - Developer Setup Guide

## 🚀 DEVELOPMENT ENVIRONMENT SETUP

### Prerequisites
- Python 3.8+ (3.9-3.11 recommended for development)
- Git (for version control)
- IDE (VS Code configuration included)
- 16GB+ RAM recommended for training
- GPU with CUDA support (optional, for training)

### Quick Development Setup
```bash
# 1. Extract developer package
unzip RideBuddy_Pro_v2.1.0_Developer_Complete.zip
cd RideBuddy_Pro_v2.1.0_Developer_Complete

# 2. Create development environment
python -m venv ridebuddy_dev
source ridebuddy_dev/bin/activate  # Linux/Mac
# OR
ridebuddy_dev\\Scripts\\activate    # Windows

# 3. Install development dependencies
pip install -r requirements.txt

# 4. Verify installation
python system_validation.py

# 5. Run development version
python ridebuddy_optimized_gui.py
```

## 🧠 AI MODEL DEVELOPMENT

### Training Custom Models
```bash
# Organize training data
python organize_dataset.py

# Train enhanced model
python enhanced_dataset_trainer.py

# Validate model performance
python validate_model.py

# Deploy trained model
python deploy_enhanced_model.py
```

### Model Architecture Customization
- Edit `src/models/` for custom architectures
- Modify `enhanced_model_integration.py` for deployment
- Use `comprehensive_trainer.py` for multi-dataset training

## 🔧 DEVELOPMENT TOOLS

### Code Organization
```
src/
├── models/          # AI model architectures
├── data/           # Data processing utilities
├── gui/            # GUI components
├── vehicle/        # Vehicle integration
└── utils/          # Helper utilities

configs/
├── training/       # Training configurations
├── deployment/     # Deployment presets
└── vehicle/        # Vehicle-specific configs
```

### Testing and Validation
```bash
# System compatibility test
python system_validation.py

# Camera hardware test  
python camera_diagnostics.py

# Vehicle integration test
python vehicle_camera_diagnostic.py

# Model performance validation
python validate_model.py
```

### Documentation Generation
```bash
# Generate HTML documentation
python convert_md_to_html.py

# Generate RTF documents
python convert_md_to_rtf.py

# Generate Word documents (if docx available)
python convert_md_to_docx.py
```

## 🚗 VEHICLE DEVELOPMENT

### Automotive Testing
```bash
# Vehicle camera diagnostics
python vehicle_camera_diagnostic.py

# Vehicle deployment testing
python vehicle_deployment_guide.py

# Fleet integration testing
set RIDEBUDDY_FLEET_MODE=true
python vehicle_launcher.py
```

### Custom Vehicle Profiles
- Modify `vehicle_config.json` for specific vehicles
- Create custom alert thresholds
- Configure dashboard layouts
- Set power management profiles

## 📊 DATA AND TRAINING

### Dataset Management
```bash
# Download additional datasets
python comprehensive_dataset_downloader.py

# Organize training data
python organize_dataset.py

# Train comprehensive model
python comprehensive_trainer.py
```

### Performance Optimization
- GPU acceleration configuration
- Model quantization options
- Edge deployment optimization
- Memory usage optimization

## 🔍 DEBUGGING AND PROFILING

### Development Debugging
```bash
# Enable debug logging
export RIDEBUDDY_LOG_LEVEL=DEBUG
python ridebuddy_optimized_gui.py

# Performance profiling
python -m cProfile ridebuddy_optimized_gui.py
```

### Common Development Tasks
- **Add new detection**: Modify AI model in `src/models/`
- **Custom GUI themes**: Edit theme configuration in main app
- **Vehicle integration**: Extend vehicle modules in `src/vehicle/`
- **Fleet features**: Develop fleet management in `src/fleet/`

## 🧪 TESTING FRAMEWORK

### Automated Testing
```bash
# Run full test suite (if available)
python -m pytest tests/

# Integration testing
python test_integration.py

# Performance benchmarking
python benchmark_performance.py
```

### Manual Testing Scenarios
- Different lighting conditions
- Various camera angles and positions
- Multiple driver profiles
- Different vehicle environments
- Edge case scenarios

## 📚 DOCUMENTATION DEVELOPMENT

### Contributing to Documentation
- All documentation in Markdown format
- Use documentation converters for distribution
- Follow existing documentation patterns
- Include code examples and screenshots

### API Documentation
- Document all public APIs
- Include usage examples
- Maintain backward compatibility notes
- Version change documentation

## 🚀 DEPLOYMENT AUTOMATION

### Pipeline Orchestration
```bash
# Full development pipeline
python ridebuddy_pipeline_orchestrator.py

# Automated model deployment
python deploy_enhanced_model.py

# Production package creation
python create_production_package.py
```

### CI/CD Integration
- GitHub Actions workflows included (`.github/`)
- Automated testing on commit
- Model validation pipeline
- Documentation generation

## 🔧 ADVANCED CUSTOMIZATION

### Custom Features Development
1. **New AI Models**: Add to `src/models/` and update training pipeline
2. **GUI Extensions**: Modify main application and add new tabs/features  
3. **Vehicle Integration**: Extend vehicle modules for new hardware
4. **Fleet Features**: Develop centralized management capabilities
5. **Performance Tools**: Add monitoring and optimization features

### Integration APIs
- Fleet management system APIs
- Third-party monitoring integration
- Cloud service connections
- Mobile app communication protocols

## 📞 DEVELOPER SUPPORT

### Resources Included
- Complete source code with comments
- Architecture documentation
- API reference guides  
- Development workflow documentation
- Testing and validation procedures

### Development Community
- Contribution guidelines in documentation
- Code review processes
- Feature request procedures
- Bug reporting and tracking

## ✅ DEVELOPMENT CHECKLIST

### Initial Setup
- [ ] Development environment activated
- [ ] All dependencies installed
- [ ] System validation passing
- [ ] Camera hardware working
- [ ] Application launching successfully

### Before Contributing
- [ ] Code follows style guidelines
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Performance impact assessed
- [ ] Vehicle compatibility verified

**Happy Development!** 🚗👨‍💻
"""
            
            zipf.writestr("DEVELOPER_SETUP_GUIDE.md", setup_guide)
            
        # Package creation complete
        package_size = os.path.getsize(zip_filename) / (1024 * 1024)  # Size in MB
        
        print(f"\n✅ Developer package created successfully!")
        print(f"📦 Package: {zip_filename}")
        print(f"📏 Size: {package_size:.1f} MB")
        print(f"📁 Files added: {files_added}")
        print(f"📂 Directories added: {dirs_added}")
        
        # Verification
        print(f"\n🔍 Package verification:")
        with zipfile.ZipFile(zip_filename, 'r') as zipf:
            file_list = zipf.namelist()
            print(f"   📋 Total files in package: {len(file_list)}")
            
            # Check for key development files
            key_dev_files = [
                "ridebuddy_optimized_gui.py",
                "enhanced_dataset_trainer.py",
                "comprehensive_trainer.py",
                "ridebuddy_pipeline_orchestrator.py",
                "DEVELOPER_MANIFEST.json",
                "DEVELOPER_SETUP_GUIDE.md"
            ]
            
            all_present = True
            for key_file in key_dev_files:
                if key_file in file_list:
                    print(f"   ✅ {key_file}")
                else:
                    print(f"   ❌ {key_file} (MISSING)")
                    all_present = False
        
        print(f"\n📋 Developer Package Summary:")
        print(f"   🔬 Complete Source: Full codebase with development tools")
        print(f"   🧠 AI Development: Training, validation, and customization tools")
        print(f"   📊 Data Pipeline: Dataset management and processing")
        print(f"   🚗 Vehicle Dev: Automotive integration development tools")
        print(f"   📚 Full Docs: Complete documentation source and generators")
        print(f"   🧪 Testing: Comprehensive validation and diagnostic tools")
        
        if all_present:
            print(f"\n🎉 DEVELOPER PACKAGE READY!")
            print(f"   ✅ Complete development environment")
            print(f"   ✅ AI training and customization tools")
            print(f"   ✅ Full source code access")
            print(f"   ✅ Advanced vehicle integration")
            print(f"   ✅ Comprehensive documentation")
        
        return zip_filename
        
    except Exception as e:
        print(f"\n❌ Error creating developer package: {e}")
        return None

if __name__ == "__main__":
    package_file = create_developer_package()
    
    if package_file:
        print(f"\n📦 DEVELOPER PACKAGE COMPLETE!")
        print(f"📄 File: {package_file}")
        print(f"📍 Location: {os.path.abspath(package_file)}")
        print(f"\n🎯 READY FOR DEVELOPMENT:")
        print(f"   ✅ AI/ML researchers and developers")
        print(f"   ✅ Computer vision engineers")
        print(f"   ✅ System integrators")
        print(f"   ✅ Advanced customization projects")
        print(f"\n💡 DEVELOPMENT FEATURES:")
        print(f"   🧠 Custom AI model training")
        print(f"   🔧 Advanced configuration options")
        print(f"   🚗 Vehicle integration development")
        print(f"   📊 Performance optimization tools")
        print(f"   📚 Complete documentation source")
    else:
        print(f"\n❌ FAILED: Could not create developer package")