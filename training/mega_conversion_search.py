"""Research-only conversion search entry point."""
from src.models.response.factory import candidate_model_names

if __name__ == "__main__":
    print("Candidate response models:", candidate_model_names())
