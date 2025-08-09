from dotenv import load_dotenv
import os
load_dotenv()

AZURE_END_POINT = os.getenv("azure_end_point")
# print(AZURE_END_POINT)
AZURE_OPENAI_KEY = os.getenv("azure_openai_api_key")
# print(AZURE_OPENAI_KEY)
