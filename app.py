from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel, load_tool, tool
import yaml
from tools.final_answer import FinalAnswerTool
from tools.visit_webpage import VisitWebpageTool
from typing import Optional
from PIL import Image
from io import BytesIO

from Gradio_UI import GradioUI

@tool
def gimme_a_meme(subreddit: str = 'me_irl') -> Image:
    """A tool that grabs a random meme from a given subreddit
    Args:
        subreddit: a string representing a valid subreddit (e.g., 'dankmemes', 'memes', 'me_irl') default value: 'me_irl'
    """
    try:
        import requests
        url = f"https://meme-api.com/gimme/{subreddit}"
        response = requests.get(url)
        response.raise_for_status()

        meme_data = response.json()
        image_url = meme_data.get("url")
        image_response = requests.get(image_url)
        image_response.raise_for_status()

        img = Image.open(BytesIO(image_response.content))
        return img
    except requests.exceptions.RequestException as e:
        print(f"Error fetching meme: {e}")
        return None


final_answer = FinalAnswerTool()
visit_webpage = VisitWebpageTool()

model = HfApiModel(
    max_tokens=2096,
    temperature=0.1,
    #model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    model_id="https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud",
    custom_role_conversions=None
)

image_generation_tool = load_tool("agents-course/text-to-image", trust_remote_code=True)

with open("prompts.yaml", 'r') as stream:
    prompt_templates = yaml.safe_load(stream)

agent = CodeAgent(
    model=model,
    tools=[final_answer, DuckDuckGoSearchTool(), gimme_a_meme, image_generation_tool, visit_webpage],
    max_steps=4,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)


GradioUI(agent).launch()