import yaml
import subprocess
import shutil
import tempfile
from pathlib import Path


def load_instructions(agent_name: str) -> str:
    """Load instructions from the YAML file."""
    # Get the directory where this file is located
    current_dir = Path(__file__).parent
    instructions_path = current_dir / "instructions.yaml"
    with open(instructions_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data[agent_name]


def download_scientific_skills(
    target_dir: str = "sandbox/.claude/skills",
    github_repo: str = "K-Dense-AI/claude-scientific-skills",
    source_path: str = "scientific-skills",
    branch: str = "main"
) -> None:
    """
    Download all directories from the scientific-skills folder in the GitHub repository
    and place them in the target directory using git clone.
    
    Args:
        target_dir: Local directory to save the skills to
        github_repo: GitHub repository in format "owner/repo"
        source_path: Path within the repo to download from
        branch: Git branch to download from
    """
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        repo_url = f"https://github.com/{github_repo}.git"
        
        try:
            # Clone the repository with depth 1 for faster download
            print("Cloning Claude Scientific Skills repository (this may take a moment)...")
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", branch, repo_url, str(temp_path)],
                check=True,
                capture_output=True,
                text=True
            )
            
            # Path to the scientific-skills folder in the cloned repo
            source_dir = temp_path / source_path
            
            if not source_dir.exists():
                raise FileNotFoundError(f"Source path '{source_path}' not found in repository")
            
            # Copy all skill directories from scientific-skills to target
            print(f"\nCopying skills to {target_path}...")
            skill_count = 0
            
            for skill_dir in source_dir.iterdir():
                if skill_dir.is_dir():
                    dest_dir = target_path / skill_dir.name
                    
                    # Remove existing directory if it exists
                    if dest_dir.exists():
                        shutil.rmtree(dest_dir)
                    
                    # Copy the skill directory
                    shutil.copytree(skill_dir, dest_dir)
                    print(f"  [+] {skill_dir.name}")
                    skill_count += 1
            
            print(f"\nSuccessfully downloaded {skill_count} scientific skills to {target_path.absolute()}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error cloning repository: {e.stderr}")
            raise
        except Exception as e:
            print(f"Error: {e}")
            raise

def setup_uv_environment(
    sandbox_path: Path,
    ml_packages: list[str] | None = None
) -> None:
    """
    Create a uv virtual environment and install machine learning packages.
    
    Args:
        sandbox_path: Path to the sandbox directory
        ml_packages: List of packages to install. If None, installs default ML packages.
    """
    if ml_packages is None:
        ml_packages = [
            "numpy",
            "pandas",
            "scikit-learn",
            "matplotlib",
            "seaborn",
            "torch",
            "torchvision",
            "torchaudio",
            "transformers",
            "datasets",
            "scipy",
            "requests",
            "pytorch-lightning",
            "torch-geometric",
        ]
    
    print("\nCreating uv virtual environment...")
    venv_path = sandbox_path / ".venv"
    
    try:
        # Create uv environment
        subprocess.run(
            # ["uv", "venv", str(venv_path)],
            ["uv", "venv", str(venv_path)],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"  [+] Virtual environment created at {venv_path}")
        
        # Install machine learning packages
        print("\nInstalling machine learning packages...")
        
        # Use uv pip to install packages
        install_cmd = ["uv", "pip", "install", "--python", str(venv_path / "Scripts" / "python.exe")] + ml_packages
        
        print(f"  Installing: {', '.join(ml_packages)}")
        print("  (This may take a few minutes...)")
        
        subprocess.run(
            install_cmd,
            check=True,
            capture_output=True,
            text=True
        )
        
        print("  [+] All packages installed successfully")
    
        
        print(f"\nVirtual environment ready at {venv_path.absolute()}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error setting up uv environment: {e.stderr}")
        print("Make sure uv is installed: https://github.com/astral-sh/uv")
        raise
    except FileNotFoundError:
        print("Error: 'uv' command not found")
        print("Please install uv: https://github.com/astral-sh/uv")
        print("  curl -LsSf https://astral.sh/uv/install.sh | sh")
        raise


def copy_env_file() -> None:
    """
    Copy the .env file from karpathy directory to sandbox directory.
    """
    karpathy_env = Path("karpathy/.env")
    sandbox_env = Path("sandbox/.env")
    
    if not karpathy_env.exists():
        print("\nWarning: .env file not found in karpathy directory")
        print(f"  Looked for: {karpathy_env.absolute()}")
        return
    
    try:
        # Create sandbox directory if it doesn't exist
        sandbox_env.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy the .env file
        shutil.copy2(karpathy_env, sandbox_env)
        print(f"\n[+] Copied .env file to {sandbox_env.absolute()}")
        
    except Exception as e:
        print(f"Error copying .env file: {e}")
        raise


def setup_sandbox() -> None:
    """
    Setup the sandbox directory with scientific skills and uv environment.
    """
    sandbox_path = Path("sandbox")
    sandbox_path.mkdir(exist_ok=True)
    
    # Copy .env file from karpathy to sandbox
    copy_env_file()
    
    # Download scientific skills
    print("\nSetting up scientific skills...")
    download_scientific_skills(target_dir="sandbox/.claude/skills")
    
    # Create uv virtual environment and install ML packages
    setup_uv_environment(sandbox_path)
    
    print("\nSandbox setup complete!")
    return True


if __name__ == "__main__":
    setup_sandbox()