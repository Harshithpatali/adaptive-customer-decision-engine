"""Research-only conversion benchmark entry point."""
from src.models.response.factory import production_model_name

if __name__ == "__main__":
    print(f"Production response model: {production_model_name()}")
