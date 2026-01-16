"""
Main Execution Script for Fake News Detection Project
Provides interactive menu to run different parts of the pipeline
"""

import os
import sys
import subprocess

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils import (
    print_banner, print_section, check_dataset_exists,
    check_model_exists, check_processed_data_exists,
    setup_project_structure, display_download_instructions,
    validate_environment
)


def display_menu():
    """Display main menu"""
    print_banner("FAKE NEWS DETECTION SYSTEM")
    
    print("""
    📋 Main Menu:
    
    1. 📥 Download Dataset Instructions
    2. 🔄 Preprocess Data
    3. 🤖 Train Model
    4. 📊 Evaluate Model
    5. 🚀 Launch Streamlit App
    6. ⚡ Run Complete Pipeline (2+3+4+5)
    7. 🔍 Check System Status
    0. 🚪 Exit
    """)


def check_system_status():
    """Check and display system status"""
    print_section("SYSTEM STATUS CHECK")
    
    print("\n📁 Project Structure:")
    if os.path.exists('data'):
        print("  ✅ data/ directory exists")
    else:
        print("  ❌ data/ directory not found")
    
    if os.path.exists('models'):
        print("  ✅ models/ directory exists")
    else:
        print("  ❌ models/ directory not found")
    
    if os.path.exists('src'):
        print("  ✅ src/ directory exists")
    else:
        print("  ❌ src/ directory not found")
    
    print("\n📊 Dataset:")
    if check_dataset_exists():
        print("  ✅ Raw dataset files found")
    else:
        print("  ❌ Raw dataset files not found")
        print("     Run option 1 for download instructions")
    
    if check_processed_data_exists():
        print("  ✅ Processed data exists")
    else:
        print("  ⚠️  Processed data not found (run preprocessing)")
    
    print("\n🤖 Model:")
    if check_model_exists():
        print("  ✅ Trained model exists")
    else:
        print("  ⚠️  Model not trained yet")
    
    print("\n🐍 Python Environment:")
    validate_environment()


def run_preprocessing():
    """Run data preprocessing"""
    print_section("DATA PREPROCESSING")
    
    if not check_dataset_exists():
        print("❌ Error: Dataset files not found!")
        print("\nPlease download the dataset first (Option 1)")
        return False
    
    print("Starting preprocessing pipeline...")
    print("This may take a few minutes...\n")
    
    try:
        from data_preprocessing import main as preprocess_main
        preprocess_main()
        print("\n✅ Preprocessing completed successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {e}")
        return False


def run_training():
    """Run model training"""
    print_section("MODEL TRAINING")
    
    if not check_processed_data_exists():
        print("❌ Error: Processed data not found!")
        print("\nPlease run preprocessing first (Option 2)")
        return False
    
    print("Starting model training...")
    print("This may take several minutes...\n")
    
    try:
        from model_training import main as training_main
        training_main()
        print("\n✅ Model training completed successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        return False


def run_evaluation():
    """Run model evaluation"""
    print_section("MODEL EVALUATION")
    
    if not check_model_exists():
        print("❌ Error: Trained model not found!")
        print("\nPlease train the model first (Option 3)")
        return False
    
    print("Starting model evaluation...")
    print("Generating visualizations and metrics...\n")
    
    try:
        from model_evaluation import main as evaluation_main
        evaluation_main()
        print("\n✅ Model evaluation completed successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        return False


def launch_streamlit():
    """Launch Streamlit app"""
    print_section("LAUNCHING STREAMLIT APP")
    
    if not check_model_exists():
        print("⚠️  Warning: Model not found!")
        print("Some features may not work properly.\n")
    
    print("Starting Streamlit app...")
    print("The app will open in your default browser.")
    print("\nPress Ctrl+C to stop the server.\n")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            "app/streamlit_app.py"
        ])
    except KeyboardInterrupt:
        print("\n\n👋 Streamlit app stopped.")
    except Exception as e:
        print(f"\n❌ Error launching Streamlit: {e}")


def run_complete_pipeline():
    """Run complete pipeline"""
    print_banner("COMPLETE PIPELINE EXECUTION")
    
    if not check_dataset_exists():
        print("❌ Error: Dataset files not found!")
        print("\nPlease download the dataset first (Option 1)")
        return
    
    print("This will run the complete pipeline:")
    print("  1. Data Preprocessing")
    print("  2. Model Training")
    print("  3. Model Evaluation")
    print("  4. Launch Streamlit App")
    print("\n⚠️  This may take 10-20 minutes depending on your system.\n")
    
    confirm = input("Do you want to continue? (y/n): ")
    
    if confirm.lower() != 'y':
        print("Pipeline execution cancelled.")
        return
    
    # Step 1: Preprocessing
    if not run_preprocessing():
        print("\n❌ Pipeline stopped due to preprocessing error.")
        return
    
    input("\n✅ Preprocessing done. Press Enter to continue to training...")
    
    # Step 2: Training
    if not run_training():
        print("\n❌ Pipeline stopped due to training error.")
        return
    
    input("\n✅ Training done. Press Enter to continue to evaluation...")
    
    # Step 3: Evaluation
    if not run_evaluation():
        print("\n❌ Pipeline stopped due to evaluation error.")
        return
    
    input("\n✅ Evaluation done. Press Enter to launch Streamlit app...")
    
    # Step 4: Launch app
    launch_streamlit()


def main():
    """Main function"""
    # Setup project structure
    setup_project_structure()
    
    while True:
        display_menu()
        
        try:
            choice = input("Enter your choice (0-7): ").strip()
            
            if choice == '0':
                print_banner("GOODBYE!")
                print("Thank you for using the Fake News Detection System!")
                sys.exit(0)
            
            elif choice == '1':
                display_download_instructions()
                input("\nPress Enter to continue...")
            
            elif choice == '2':
                run_preprocessing()
                input("\nPress Enter to continue...")
            
            elif choice == '3':
                run_training()
                input("\nPress Enter to continue...")
            
            elif choice == '4':
                run_evaluation()
                input("\nPress Enter to continue...")
            
            elif choice == '5':
                launch_streamlit()
            
            elif choice == '6':
                run_complete_pipeline()
            
            elif choice == '7':
                check_system_status()
                input("\nPress Enter to continue...")
            
            else:
                print("\n❌ Invalid choice! Please enter a number between 0-7.")
                input("Press Enter to continue...")
        
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            input("Press Enter to continue...")


if __name__ == "__main__":
    main()