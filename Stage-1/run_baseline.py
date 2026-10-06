import os
import time
import base64
import json
import requests
import psutil

SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
IMAGE_PATH = "../data/convs.png"
PROMPT = "Опиши подробно, что изображено на этом кадре."
MAX_TOKENS = 256

def get_process_ram_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def get_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def run_gguf_benchmark(image_path, prompt_text):
    print("=" * 60)
    print("QWEN2-VL-2B Q4_K_M BENCHMARK (STREAMING)")
    print("=" * 60)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    start_image = time.perf_counter()
    img_b64 = get_image_base64(image_path)
    image_prepare_time = time.perf_counter() - start_image
    image_url = f"data:image/png;base64,{img_b64}"

    payload = {
        "model": "Qwen2-VL-2B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.2,
        "stream": True  # Включаем стриминг
    }

    print("\nRunning inference...\n--- Response Start ---")

    start_inference = time.perf_counter()
    first_token_time = None
    generated_text = ""
    chunk_count = 0

    response = requests.post(SERVER_URL, json=payload, stream=True, timeout=300)
    response.raise_for_status()

    for line in response.iter_lines():
        if not line:
            continue
        line_str = line.decode('utf-8')
        if line_str.startswith("data: "):
            data_content = line_str[6:].strip()
            if data_content == "[DONE]":
                break
            try:
                chunk = json.loads(data_content)
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    if first_token_time is None:
                        first_token_time = time.perf_counter() - start_inference
                    print(delta, end="", flush=True)
                    generated_text += delta
                    chunk_count += 1
            except json.JSONDecodeError:
                pass

    total_time = time.perf_counter() - start_inference
    print("\n--- Response End ---")

    # Метрики
    ttft = first_token_time if first_token_time else total_time
    gen_time = total_time - ttft
    # Оценка генерации: chunk_count близка к числу сгенерированных токенов
    gen_tps = chunk_count / gen_time if gen_time > 0 else 0
    overall_tps = chunk_count / total_time if total_time > 0 else 0

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print("--- Timing ---")
    print(f"Image preparation:  {image_prepare_time:.3f} s")
    print(f"TTFT (Processing):  {ttft:.3f} s  <-- Время до 1-го токена (Prompt/Vision)")
    print(f"Generation time:    {gen_time:.3f} s")
    print(f"Total time:         {total_time:.3f} s")

    print("\n--- Speed ---")
    print(f"Tokens (chunks):    {chunk_count}")
    print(f"Generation TPS:     {gen_tps:.2f} tokens/sec  <-- Скорость самой генерации")
    print(f"Overall TPS:        {overall_tps:.2f} tokens/sec")
    print("=" * 60)

if __name__ == "__main__":
    run_gguf_benchmark(IMAGE_PATH, PROMPT)