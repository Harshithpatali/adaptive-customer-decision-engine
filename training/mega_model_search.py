"""Research-only model comparison/search scaffold."""
from src.models.response.factory import candidate_model_names


def main() -> None:
    print("Candidates:", ", ".join(candidate_model_names()))
    print("Final production response model: xgboost")


if __name__ == "__main__":
    main()
