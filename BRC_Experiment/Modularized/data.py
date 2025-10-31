from typing import List, Tuple
import json
import os


def load_train_dataset(dataset_name: str, data_base_path: str = "data") -> List[Tuple[str, str]]:
    """Load training dataset and return list of (positive_prompt, negative_prompt) pairs.

    This function loads train data from the specified dataset directory and creates
    positive/negative prompt pairs with choices pre-filled for building steering vectors.

    Args:
        dataset_name: Name of the dataset (e.g., "deference", "reassurance", "satisficing", "sycophancy")
        data_base_path: Base path where dataset directories are located

    Returns:
        List of tuples with (positive_prompt, negative_prompt) pairs for training
    """
    data_path = os.path.join(data_base_path, dataset_name, f"{dataset_name}_train.json")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training dataset file not found: {data_path}")

    with open(data_path, 'r') as f:
        data = json.load(f)

    prompt_pairs: List[Tuple[str, str]] = []

    for item in data:
        question_with_choices = item["question"]

        # Create prompts with the appropriate choice pre-filled
        matching_choice = item["answer_matching_behavior"]
        non_matching_choice = item["answer_not_matching_behavior"]

        # Extract the choice number/letter "(A)" -> "A")
        matching_choice_clean = matching_choice.strip("()")
        non_matching_choice_clean = non_matching_choice.strip("()")

        # Create prompts that end just before the choice
        base_prompt = f"{question_with_choices}\n\nI choose ("

        # Add the specific choice for each prompt
        positive_prompt = f"{base_prompt}{matching_choice_clean}"
        negative_prompt = f"{base_prompt}{non_matching_choice_clean}"

        prompt_pairs.append((positive_prompt, negative_prompt))

    return prompt_pairs


def load_test_dataset(dataset_name: str, data_base_path: str = "data") -> List[str]:
    """Load test dataset and return list of base prompts without choices filled in.

    This function loads test data from the specified dataset directory and returns
    base prompts for evaluation (letting the model generate choices naturally).

    Returns:
        List of base prompts for testing (without choices pre-filled)
    """
    data_path = os.path.join(data_base_path, dataset_name, f"{dataset_name}_test.json")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Test dataset file not found: {data_path}")

    with open(data_path, 'r') as f:
        data = json.load(f)

    base_prompts: List[str] = []

    for item in data:
        question_with_choices = item["question"]

        # For testing, we want the model to generate the choice naturally
        # So we use the base prompt that ends with "I choose ("
        base_prompt = f"{question_with_choices}\n\nI choose ("
        base_prompts.append(base_prompt)

    return base_prompts
