import os
import sys
import subprocess
import openai
import re
import argparse

# --- 1. CONFIGURATION ---
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://192.168.2.134:8033/v1")
REASONING_URL = os.getenv("REASONING_URL", "http://192.168.2.137:8033/v1")
CLIENT_TIMEOUT = (10.0, 600.0)

# Client Setup
orch_client = openai.OpenAI(
    base_url=ORCHESTRATOR_URL, 
    api_key="not-needed",
    timeout=CLIENT_TIMEOUT
)
reason_client = openai.OpenAI(
    base_url=REASONING_URL, 
    api_key="not-needed",
    timeout=CLIENT_TIMEOUT
)

# --- 2. GIT HELPER FUNCTIONS ---
def run_git_command(args, check=True):
    """Executes a git command and returns the output."""
    try:
        result = subprocess.run(
            ["git"] + args,
            text=True,
            capture_output=True,
            check=check
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        if check:
            print(f"❌ [Git Error] Failed running 'git {' '.join(args)}'")
            print(f"Error Output: {e.stderr.strip()}")
            sys.exit(1)
        return e.stdout.strip()

def get_current_branch():
    return run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])

def list_local_branches():
    """Scans and prints all local git branches."""
    print("🔍 Scanning local repository for branches...\n")
    output = run_git_command(["branch", "--format=%(refname:short)"])
    if output:
        branches = output.split('\n')
        current = get_current_branch()
        for branch in branches:
            if branch == current:
                print(f"  👉 * {branch} (current)")
            else:
                print(f"     {branch}")
    else:
        print("⚠️ No branches found or not a git repository.")

def get_conflicting_files():
    """Finds all files currently in a conflicted state."""
    output = run_git_command(["diff", "--name-only", "--diff-filter=U"])
    if not output:
        return []
    return output.split('\n')

def detect_merge_state():
    """Detects if a merge is in progress and identifies the source/target branches."""
    try:
        git_dir = run_git_command(["rev-parse", "--git-dir"])
        merge_head_path = os.path.join(git_dir, "MERGE_HEAD")
        merge_msg_path = os.path.join(git_dir, "MERGE_MSG")
        
        if not os.path.exists(merge_head_path):
            return None, None
        
        target = get_current_branch()
        source = "unknown-branch"
        
        # Extract source branch from the auto-generated MERGE_MSG
        if os.path.exists(merge_msg_path):
            with open(merge_msg_path, "r", encoding="utf-8", errors="ignore") as f:
                msg = f.read()
                match = re.search(r"Merge branch '([^']+)'", msg)
                if match:
                    source = match.group(1)
        
        return target, source
    except Exception as e:
        print(f"⚠️ Could not detect merge state: {e}")
        return None, None

# --- 3. THE AI RESOLUTION ENGINE ---
def clean_llm_output(text):
    """Extracts code from markdown blocks to prevent corrupting the source files."""
    text = text.strip()
    
    # Safely extract code if the LLM wrapped it in markdown, ignoring conversational text
    match = re.search(r'```[a-zA-Z]*\n(.*?)```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
        
    # Fallback stripping if the match fails but backticks exist
    text = re.sub(r'^```[\w]*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n```$', '', text, flags=re.MULTILINE)
    return text.strip()

def resolve_file_with_ai(file_path):
    """Reads a conflicted file, sends it to the Reasoner and Orchestrator, and writes the fix."""
    print(f"\n📄 Processing conflict in: {file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        print(f"⚠️ UTF-8 decode failed for {file_path}, falling back to latin-1.")
        try:
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()
        except Exception as e:
            print(f"⚠️ Could not read {file_path}: {e}")
            return False
    except Exception as e:
        print(f"⚠️ Could not read {file_path}: {e}")
        return False

    if "<<<<<<<" not in content or "=======" not in content:
        print(f"💡 No standard conflict markers found in {file_path}. Skipping.")
        return False

    # STEP A: Reasoner Drafts the Fix
    print("   🧠 [Reasoner] Analyzing markers and drafting merge strategy...")
    reasoner_prompt = f"""
    You are an expert developer. The following file has a git merge conflict.
    Resolve the conflict intelligently. Consolidate documentation, keep the best code features from both branches, or fix breaking changes between them.
    
    FILE CONTENT:
    {content}
    
    INSTRUCTIONS: Output ONLY the fully resolved, merged file content. Do not include markdown code blocks. Do not explain your choices. Just the raw, valid code/text.
    """
    
    try:
        reasoning_response = reason_client.chat.completions.create(
            model="qwen-3.5-35b",
            messages=[
                {"role": "system", "content": "You are a code merging engine. Output raw code only."},
                {"role": "user", "content": reasoner_prompt}
            ],
            max_tokens=8192 
        )
        draft = reasoning_response.choices[0].message.content
    except Exception as e:
        print(f"   ❌ [Reasoner Error]: {e}")
        return False

    # STEP B: Orchestrator Sanity Check
    print("   ⚖️ [Orchestrator] Verifying code integrity and marker removal...")
    editor_prompt = f"""
    Review this drafted code resolution for a git merge conflict.
    
    DRAFT:
    {draft}
    
    TASK: Ensure there are absolutely NO leftover git markers (<<<<<<<, =======, >>>>>>>) in the code. Ensure the code is syntactically logical. 
    Output ONLY the final code. DO NOT wrap it in ```markdown``` or ```python``` blocks.
    """

    try:
        verification_response = orch_client.chat.completions.create(
            model="nemotron-orchestrator-8b",
            messages=[
                {"role": "system", "content": "You are a strict code formatter. Output raw code/text only."},
                {"role": "user", "content": editor_prompt}
            ]
        )
        final_code = clean_llm_output(verification_response.choices[0].message.content)
    except Exception as e:
        print(f"   ❌ [Orchestrator Error]: {e}")
        return False

    if "<<<<<<<" in final_code or "=======" in final_code:
        print("   ❌ [Error] AI failed to remove conflict markers. Manual intervention required.")
        return False

    try:
        # Write back using the same encoding that succeeded, defaulting to utf-8
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(final_code)
        print(f"   ✅ Successfully resolved and saved {file_path}")
        return True
    except Exception as e:
        print(f"   ❌ Failed to write to {file_path}: {e}")
        return False

# --- 4. THE WORKFLOW ---
def perform_ai_merge(target_branch, source_branch, merge_in_progress=False):
    print("🤖 Starting AI-Driven Git Merge")
    print("=" * 40)
    
    safe_source = source_branch.replace("/", "-")
    safe_target = target_branch.replace("/", "-")
    merge_branch_name = f"ai-merge-{safe_source}-into-{safe_target}"

    if not merge_in_progress:
        # If no merge is happening, start from scratch
        status = run_git_command(["status", "--porcelain"])
        if status:
            print("❌ Your working directory is not clean. Please commit or stash your changes first.")
            sys.exit(1)

        print("🔄 Setting up branches...")
        run_git_command(["checkout", target_branch])
        print(f"🌿 Creating isolated branch: {merge_branch_name}")
        run_git_command(["checkout", "-b", merge_branch_name])
        print(f"🔀 Attempting standard git merge of '{source_branch}'...")
        run_git_command(["merge", source_branch], check=False)
    else:
        # If a merge is actively failing, move the broken state to a new branch safely
        print(f"⚠️ Actively fixing current merge: '{source_branch}' into '{target_branch}'")
        print(f"🌿 Shifting conflicted state to isolated branch: {merge_branch_name}")
        try:
            run_git_command(["checkout", "-b", merge_branch_name])
        except SystemExit:
            print(f"⚠️ Git prevented branch creation during conflict. Remaining on '{target_branch}'.")

    conflicting_files = get_conflicting_files()
    
    if not conflicting_files:
        print("✅ Merge state is clean. No conflicts found.")
        sys.exit(0)

    print(f"⚠️ Conflicts detected in {len(conflicting_files)} file(s). Engaging AI Resolver...")
    
    resolution_failures = 0
    for file_path in conflicting_files:
        success = resolve_file_with_ai(file_path)
        if success:
            run_git_command(["add", file_path])
        else:
            resolution_failures += 1

    print("\n" + "=" * 40)
    if resolution_failures == 0:
        print("🎉 All conflicts resolved by AI. Committing results...")
        run_git_command(["commit", "-m", f"AI automated merge resolution: {source_branch} into {target_branch}"])
        print(f"✅ Success! Your resolved code is waiting on branch: {merge_branch_name}")
    else:
        print(f"⚠️ AI resolved some files, but {resolution_failures} files failed or require manual review.")
        print(f"   The branch '{merge_branch_name}' is currently in a merging state.")
        print("   Please review the files, resolve manually, and run 'git commit'.")

# --- 5. CLI EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-Driven Git Merge Conflict Resolver")
    parser.add_argument("target_branch", nargs="?", help="The branch you want to merge INTO")
    parser.add_argument("source_branch", nargs="?", help="The branch you want to merge FROM")
    parser.add_argument("-s", "--scan", action="store_true", help="Scan and list all local branches")
    
    args = parser.parse_args()

    if args.scan:
        list_local_branches()
        sys.exit(0)

    # Manual mode: Pass branches directly
    if args.target_branch and args.source_branch:
        perform_ai_merge(args.target_branch, args.source_branch, merge_in_progress=False)
        
    # Zero-argument mode: Auto-detect existing conflicts
    elif not args.target_branch and not args.source_branch:
        target_b, source_b = detect_merge_state()
        if target_b and source_b:
            perform_ai_merge(target_b, source_b, merge_in_progress=True)
        else:
            print("❌ No active merge conflict detected. Either start a merge, or provide branches manually.")
            print("To see available commands, run: python ai_merge.py --help")
            sys.exit(1)
            
    else:
        print("❌ Error: You must provide BOTH a target and source branch, or provide NONE to auto-detect a conflict.")
        print("To see available commands, run: python ai_merge.py --help")
        sys.exit(1)
