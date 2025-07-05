import os
import gradio as gr
import requests
import pandas as pd
import openai

# Load OpenAI API Key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Improved Smart Agent ---
class SmartAgent:
    def __init__(self):
        print("SmartAgent initialized.")

    def __call__(self, question: str) -> str:
        print(f"[Agent] Processing question: {question[:60]}...")
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a highly logical AI assistant specialized in answering "
                            "knowledge and reasoning questions precisely and concisely. "
                            "Provide only the final answer, with no extra commentary or labels."
                        )
                    },
                    {"role": "user", "content": question}
                ],
                temperature=0.2,  # Lower temp for more deterministic answers
                max_tokens=300,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )
            answer = response.choices[0].message["content"].strip()
            print(f"[Agent] Answer: {answer}")
            return answer
        except Exception as e:
            print(f"[Agent Error]: {e}")
            return "Error: Could not generate answer."

# --- Submission Logic ---
def run_and_submit_all(profile: gr.OAuthProfile | None):
    space_id = os.getenv("SPACE_ID", "your_username/your_space_name")

    if profile:
        username = profile.username
        print(f"User logged in: {username}")
    else:
        return " Please log in to Hugging Face first.", None

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    agent = SmartAgent()

    try:
        response = requests.get(f"{DEFAULT_API_URL}/questions", timeout=15)
        response.raise_for_status()
        questions = response.json()
    except Exception as e:
        return f" Failed to fetch questions: {e}", None

    results_log = []
    answers_payload = []

    for item in questions:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or not question_text:
            continue
        try:
            answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": answer})
        except Exception as e:
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"Error: {e}"})

    submission = {
        "username": username,
        "agent_code": agent_code,
        "answers": answers_payload
    }

    try:
        resp = requests.post(f"{DEFAULT_API_URL}/submit", json=submission, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        score = result.get("score", "N/A")
        correct = result.get("correct_count", "?")
        total = result.get("total_attempted", "?")

        msg = f"✅ Submission Successful!\nUser: {username}\nScore: {score}% ({correct}/{total} correct)"
        return msg, pd.DataFrame(results_log)
    except Exception as e:
        return f" Submission failed: {e}", pd.DataFrame(results_log)

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("## GAIA Agent Submission Tool")
    gr.Markdown("Click below to log in and run your agent on the GAIA benchmark.")
    gr.LoginButton()
    run_btn = gr.Button("Run Evaluation & Submit All Answers")
    status = gr.Textbox(label="Status", lines=5)
    table = gr.DataFrame(label="Results")

    run_btn.click(fn=run_and_submit_all, outputs=[status, table])

# --- Optional local test for debugging ---
def local_test():
    print("Running local test...")
    agent = SmartAgent()
    test_questions = [
        "How many studio albums did Mercedes Sosa publish between 2000 and 2009?",
        "What is the capital city of France?",
        "List the vegetables in this list: milk, eggs, flour, whole bean coffee, Oreos, sweet potatoes, fresh basil, plums, green beans, rice, corn, bell pepper, whole allspice, acorns, broccoli, celery, zucchini, lettuce, peanuts."
    ]
    for q in test_questions:
        ans = agent(q)
        print(f"Q: {q}\nA: {ans}\n{'-'*40}")

if __name__ == "__main__":
    print("Launching GAIA Agent App")
    local_test()
    demo.launch()