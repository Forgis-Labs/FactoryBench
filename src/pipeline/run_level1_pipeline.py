import argparse
import subprocess
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

def main():
    parser = argparse.ArgumentParser(description="End-to-End Pipeline for FactoryBench Level 1")
    parser.add_argument("-n", "--num-questions", type=int, default=100, help="Number of questions to generate")
    parser.add_argument("--dataset-repo", type=str, default="the authors/FactoryNet_Dataset")
    parser.add_argument("--kg-repo", type=str, default="the authors/FactoryBench-KnowledgeGraph")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode (faster generation)")
    parser.add_argument("--model", type=str, default=None, help="Specific Azure OpenAI model deployment to use")
    parser.add_argument("--eval-level", type=str, default="level_1", help="Evaluation level tag for Opik tracing (default: level_1)")
    
    args = parser.parse_args()
    
    base_name = "level1_pipeline"
    q_dir = Path(f"datasets/questions/{base_name}")
    p_dir = Path(f"datasets/prompts/{base_name}")
    r_dir = Path(f"datasets/replies/{base_name}")
    f_dir = Path(f"figures/{base_name}")
    
    print("="*60)
    print("1. Downloading Knowledge Graph from Hugging Face...")
    print("="*60)
    kg_path = hf_hub_download(repo_id=args.kg_repo, repo_type="dataset", filename="machines.json")
    print(f"Knowledge Graph downloaded to: {kg_path}")
    try:
        grippers_path = hf_hub_download(repo_id=args.kg_repo, repo_type="dataset", filename="grippers.json")
        print(f"Grippers downloaded to: {grippers_path}")
    except Exception as exc:
        grippers_path = None
        print(f"Grippers file not available on HF ({exc}); skipping gripper info in prompts.")
    try:
        dataset_index_path = hf_hub_download(repo_id=args.kg_repo, repo_type="dataset", filename="dataset.json")
        print(f"Dataset index downloaded to: {dataset_index_path}")
    except Exception as exc:
        dataset_index_path = None
        print(f"Dataset index not available on HF ({exc}); skipping dataset->gripper mapping.")
    print("")
    
    print("="*60)
    print("2. Generating Questions...")
    print("="*60)
    cmd1 = [
        sys.executable, "-m", "src.question_generation.level1.level1", 
        "-n", str(args.num_questions), 
        "--dataset-repo", args.dataset_repo, 
        "--output-dir", str(q_dir)
    ]
    if args.test_mode:
        cmd1.append("--test-mode")
    subprocess.run(cmd1, check=True)
    print("\n")
    
    print("="*60)
    print("3. Building Prompts...")
    print("="*60)
    cmd2 = [
        sys.executable, "-m", "src.question_generation.build_prompts_from_questions",
        "--input", str(q_dir),
        "--output", str(p_dir),
        "--machines", kg_path,
    ]
    if grippers_path:
        cmd2.extend(["--grippers", grippers_path])
    if dataset_index_path:
        cmd2.extend(["--dataset-index", dataset_index_path])
    subprocess.run(cmd2, check=True)
    print("\n")
    
    print("="*60)
    print("4. Running LLM Evaluation...")
    print("="*60)
    cmd3 = [
        sys.executable, "-m", "src.evaluation.test_gpt_5mini",
        "--input", str(p_dir),
        "--output-dir", str(r_dir),
        "--questions", str(q_dir),
        "--overwrite"
    ]
    if args.model:
        cmd3.extend(["--model", args.model])
    if args.eval_level:
        cmd3.extend(["--eval-level", args.eval_level])
    subprocess.run(cmd3, check=True)
    print("\n")
    
    print("="*60)
    print("5. Analyzing Results...")
    print("="*60)
    cmd4 = [
        sys.executable, "-m", "src.evaluation.analyze_results",
        "--input", str(r_dir),
        "--questions", str(q_dir),
        "--figures-dir", str(f_dir)
    ]
    subprocess.run(cmd4, check=True)
    print("\nPipeline Completed Successfully!")

if __name__ == "__main__":
    main()
