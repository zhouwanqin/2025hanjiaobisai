def extract_scores(response_content):
    # This function extracts scores from the model's response content.
    # It assumes the response is in a specific tabular format.
    scores = {}
    lines = response_content.splitlines()
    
    for line in lines:
        if "评分维度" in line:
            continue  # Skip header line
        if "|" in line:
            parts = line.split("|")
            if len(parts) >= 3:
                dimension = parts[1].strip()
                score = int(parts[2].strip())
                scores[dimension] = score
    
    return scores

def format_scores_for_visualization(scores):
    # This function formats the extracted scores for visualization.
    formatted_scores = []
    for dimension, score in scores.items():
        formatted_scores.append({"dimension": dimension, "score": score})
    return formatted_scores

def validate_response(response_content):
    # This function validates the response content to ensure it contains the expected format.
    if not isinstance(response_content, str) or len(response_content) == 0:
        raise ValueError("Response content is empty or not a string.")
    if "评分维度" not in response_content:
        raise ValueError("Response content does not contain scoring dimensions.")