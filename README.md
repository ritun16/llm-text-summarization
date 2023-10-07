# LLM-based Text Summarization: Novice to Maestro 🚀
A comprehensive guide and codebase for text summarization harnessing the capabilities of Large Language Models (LLMs). Delve deep into techniques, from chunking to clustering, and maximize the potential of LLMs like GPT-3.5 and GPT-4.

📄 **Article**: [I highly recommend reading this article before diving into the code.](https://sourajit16-02-93.medium.com/text-summarization-unleashed-novice-to-maestro-with-llms-and-instant-code-solutions-8d26747689c4)

## Architecture
![Full_Architecture](https://github.com/ritun16/llm-text-summarization/assets/44939374/5e6f62a4-5356-43ff-9500-11f8e4d60dd7)


## 🚀 Getting Started
1. **Clone the Repository**
2. **Install Dependencies**: 
    ```bash
    python3 -m pip install -r requirements.txt
    ```
3. **Install Spacy's English Dataset**: 
    ```bash
    python3 -m spacy download en_core_web_sm
    ```
4. **Set Up OpenAI API Key**: 
    ```bash
    export OPENAI_API_KEY='sk-...'
    ```
5. **Configure IO**: Navigate to `src/config.yaml` and update the `input_file` and `output_file` parameters under `io_config`.
6. **File Handling**: For the input file, only `.txt` is accepted. For the output, `.json` is preferred. Place the input file in the `input` folder. The generated summary will be in the `output` folder.
7. **Run the Program**: 
    ```bash
    cd src/
    python3 main.py
    ```

## 🛠 Understanding the `config.yaml` File
- `summary_type_token_limit`: Determines how to categorize the input text: short, medium, or long.
- `sentence_splitter`: Adjust `approx_total_doc_tokens`. Keep it around 1000 for medium-sized texts and up to 6000 for longer texts.
- `cod`: Configuration for Chain of Density (CoD) prompting.
- `map_reduce`: To further condense the final summary with CoD, set `final_dense` to `true`.
- `cluster_summarization`: Adjust `num_closest_points_per_cluster` (max value: 3) for the `top-k` best chunks. Vary `num_clusters` (hyper-parameter for k-means) to optimize results.
- Remaining configs are self-explanatory.

## 📋 Output JSON Format
The output JSON comprises:
```json
{
    "summary": "Descriptive final summary...",
    "keywords": ["Keyword1", "Keyword2", "..."],
    "metadata": {
        "total_tokens": 3625,
        "total_cost": 0.082,
        "total_time": 86.23
    }
}
```
- `summary`: The final summary output
- `keywords`: important keywords and phrases
- `metadata`: Provides total time (in seconds) taken to execute your summary, total cost (in USD) for openai, and total token counts in the whole process

❤️ If this repository helps, please star ⭐, and share ✔️! <br>
If you also found the [article](https://sourajit16-02-93.medium.com/text-summarization-unleashed-novice-to-maestro-with-llms-and-instant-code-solutions-8d26747689c4) informative and think it could be beneficial to others, I'd be grateful if you could like 👍, follow 👉, and share✔️ the piece with others. <br>
Happy coding!
