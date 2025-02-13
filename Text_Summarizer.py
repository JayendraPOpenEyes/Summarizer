import os
import re
import requests
import io
import logging
import json
import base64
from bs4 import BeautifulSoup
from PIL import Image
from pdf2image import convert_from_bytes
from dotenv import load_dotenv
import pdfkit
import tiktoken
from openai import OpenAI
import pytesseract  # Add Tesseract for OCR

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Base directory to save all data
SAVE_DIR = "saved_data"
os.makedirs(SAVE_DIR, exist_ok=True)  # Create base directory if it doesn't exist

class TextProcessor:
    def __init__(self, model="gpt-4o-mini"):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is missing. Ensure the OpenAI API key is set in the .env file.")
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.model = model  # Model used for summarization and JSON generation

    def get_save_directory(self, base_name):
        """
        Create and return a dedicated folder (under SAVE_DIR) for the link based on base_name.
        All files (images, PDFs, HTML, JSON, etc.) for that link will be saved here.
        """
        folder_path = os.path.join(SAVE_DIR, base_name)
        os.makedirs(folder_path, exist_ok=True)
        return folder_path

    def get_base_name_from_link(self, link):
        """
        Extract a base name from the link to be used for file/folder naming.
        This method removes file extensions and any unwanted parts.
        """
        parts = link.split('/')
        meaningful_parts = [part for part in parts[-4:] if part and part.lower() not in ['pdf', 'html', 'htm']]
        base_name = '_'.join(meaningful_parts)
        if not base_name:
            base_name = '_'.join([part for part in parts if part])
        base_name = re.sub(r"\.(htm|html|pdf)$", "", base_name, flags=re.IGNORECASE)
        base_name = re.sub(r"[^\w\-_\. ]", "_", base_name)
        max_length = 50
        if len(base_name) > max_length:
            base_name = base_name[:max_length]
        if not base_name:
            base_name = "default_name"
        return base_name

    def is_google_cache_link(self, link):
        """Return True if the link is a Google Cache link."""
        return "webcache.googleusercontent.com" in link

    def is_blank_pdf(self, text):
        """
        Check if the extracted text from a PDF is blank or too short.
        Adjust the threshold as needed.
        """
        clean_text = re.sub(r"\s+", "", text).strip()
        return len(clean_text) < 100

    def process_image_with_tesseract(self, image_path):
        """
        Process an image using Tesseract OCR to extract text.
        """
        try:
            text = pytesseract.image_to_string(Image.open(image_path))
            return text
        except Exception as e:
            logging.error(f"Error processing image with Tesseract: {str(e)}")
            return ""

    def extract_text_from_pdf(self, pdf_content, link):
        """
        Convert a PDF (from pdf_content) to images, save each image in the dedicated folder,
        and extract text from each image using Tesseract OCR.
        """
        base_name = self.get_base_name_from_link(link)
        folder = self.get_save_directory(base_name)
        images = convert_from_bytes(pdf_content.read())
        combined_text = ""
        for i, img in enumerate(images):
            img_filename = f"{base_name}_page_{i + 1}.png"
            img_path = os.path.join(folder, img_filename)
            img.save(img_path, 'PNG')
            logging.info(f"Saved image permanently: {img_path}")
            image_text = self.process_image_with_tesseract(img_path)
            combined_text += image_text + "\n"
        return combined_text

    def extract_text_from_html(self, html_content, link):
        """
        Extract visible text from an HTML page.
        Also removes script and style elements.
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        text = soup.get_text(separator=' ').strip()
        return text

    def extract_text_from_url(self, url):
        """
        Fetch content from a URL and determine if it is a PDF or HTML.
        For PDFs, save the file and process via image conversion.
        For HTML, save the file and extract text directly.
        Returns a dictionary with the extracted text, content type, and any error encountered.
        """
        try:
            if self.is_google_cache_link(url):
                return {"text": "", "content_type": None, "error": "google_cache"}

            response = requests.get(url)
            response.raise_for_status()  # Raise an error for bad HTTP status codes
            content_type = response.headers.get('Content-Type', '').lower()
            base_name = self.get_base_name_from_link(url)
            folder = self.get_save_directory(base_name)

            # If URL or content type indicates a PDF, process accordingly.
            if url.lower().endswith('.pdf') or 'application/pdf' in content_type:
                # Save the original PDF file
                pdf_path = os.path.join(folder, f"{base_name}.pdf")
                with open(pdf_path, 'wb') as f:
                    f.write(response.content)
                logging.info(f"Saved PDF permanently: {pdf_path}")

                text = self.extract_text_from_pdf(io.BytesIO(response.content), url)
                if self.is_blank_pdf(text):
                    return {"text": "", "content_type": "pdf", "error": "blank_pdf"}
                return {"text": text, "content_type": "pdf", "error": None}

            # If HTML (or .htm/.html) is detected, process as HTML.
            elif url.lower().endswith('.htm') or url.lower().endswith('.html') or 'text/html' in content_type:
                # Save the original HTML file
                html_path = os.path.join(folder, f"{base_name}.html")
                with open(html_path, 'wb') as f:
                    f.write(response.content)
                logging.info(f"Saved HTML permanently: {html_path}")

                text = self.extract_text_from_html(response.content, url)
                return {"text": text, "content_type": "html", "error": None}
            else:
                return {"text": "", "content_type": None, "error": "unsupported_type"}
        except Exception as e:
            logging.error(f"Error fetching URL {url}: {str(e)}")
            return {"text": "", "content_type": None, "error": str(e)}

    def preprocess_text(self, text):
        """
        Clean and preprocess the extracted text by normalizing whitespace and removing non-ASCII characters.
        """
        text = re.sub(r"[\r\n]{2,}", "\n", text)
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text.strip()

    def generate_html_structure(self, text):
        """
        Convert the cleaned text into an HTML structure.
        Longer paragraphs are wrapped in <p> tags while shorter ones are treated as headings (<h1> tags).
        """
        paragraphs = text.split('\n')
        html = ""
        for para in paragraphs:
            if len(para.split()) > 10:
                html += f"<p>{para.strip()}</p>\n"
            else:
                html += f"<h1>{para.strip()}</h1>\n"
        return html
    
    def generate_json_with_prompt(self, html, base_name):
        """
        Convert HTML to a structured JSON (with keys 'h1' and 'p') using an OpenAI prompt.
        The generated JSON is then saved in the dedicated folder.
        """
        soup = BeautifulSoup(html, 'html.parser')
        text = soup.get_text(separator=' ').strip()
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Convert the following text into a structured JSON format with keys 'h1' and 'p'. "
                        "Only output the JSON without any additional text.\n\n" + text
                    )
                }],
                max_tokens=1000,
            )
            json_output = response.choices[0].message.content

            # Ensure valid JSON format from the response
            if isinstance(json_output, dict):  # If already a dictionary, return it directly
                json_data = json_output
            elif json_output.strip().startswith("{") and json_output.strip().endswith("}"):
                try:
                    json_data = json.loads(json_output)  # Parse JSON string into a dictionary
                except json.JSONDecodeError:
                    logging.error("Invalid JSON response from OpenAI. The response could not be parsed.")
                    return {}
            else:
                logging.error("OpenAI response is not a valid JSON format.")
                return {}

            # Save the JSON file in the dedicated folder
            base_folder = self.get_save_directory(base_name)
            json_path = os.path.join(base_folder, f"{base_name}.json")
            with open(json_path, 'w') as json_file:
                json.dump(json_data, json_file, indent=4)
            logging.info(f"Saved JSON permanently: {json_path}")

            return json_data  # Return the parsed dictionary
        except Exception as e:
            logging.error(f"Error generating JSON with OpenAI: {str(e)}")
            return {} 
          
    def truncate_text(self, text, max_tokens=3000):
        """
        Truncate the text so that it stays within the token limit.
        This is useful for ensuring the prompt does not exceed model limits.
        """
        encoding = tiktoken.encoding_for_model(self.model)
        tokens = encoding.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        return encoding.decode(tokens)

    def generate_summaries_with_chatgpt(self, combined_text):
        """
        Generate an extractive summary, an abstractive summary, and highlights in one API call.
        The prompt has been updated to instruct the model not to include extra or repeated text.
        """
        combined_text = self.truncate_text(combined_text, max_tokens=4000)
        prompt = f"""
Generate the following summaries for the text below. Please adhere to these instructions:

For Abstractive Summary:
- The summary should be concise and not very long.
- It should cover all the key points very shortly.
- Summarize the content in one short paragraph (maximum 8 sentences).

For Extractive Summary:
- Generate a minimum of 2 paragraphs if the content is sufficiently long; adjust accordingly if the content is short.
- Provide a sensible extractive summary capturing the main ideas.

For Highlights & Analysis:
- Produce 15 to 20 bullet points grouped under 4 meaningful headings.
- Each heading should be relevant to the content and include bullet points with key details.
- Highlights should be in the form of headings only, followed by bullet points.

Use the following markers exactly for each section:

Abstractive Summary:
[Abstractive]

Extractive Summary:
[Extractive]

Highlights & Analysis:
[Highlights]

Only output the text within these markers without any additional commentary.

Text:
{combined_text}
"""
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=1500,
            )
            summaries = response.choices[0].message.content

            # Corrected regex patterns in generate_summaries_with_chatgpt():
            abstractive_match = re.search(r"\[Abstractive\](.*?)\[Extractive\]", summaries, re.DOTALL)
            extractive_match = re.search(r"\[Extractive\](.*?)\[Highlights\]", summaries, re.DOTALL)
            highlights_match = re.search(r"\[Highlights\](.*)", summaries, re.DOTALL)

            return {
                "extractive": extractive_match.group(1).strip() if extractive_match else "Extractive summary not found.",
                "abstractive": abstractive_match.group(1).strip() if abstractive_match else "Abstractive summary not found.",
                "highlights": highlights_match.group(1).strip() if highlights_match else "Highlights not found."
            }
        except Exception as e:
            logging.error(f"Error generating summaries: {str(e)}")
            return {
                "extractive": "Error generating extractive summary.",
                "abstractive": "Error generating abstractive summary.",
                "highlights": "Error generating highlights."
            }

    def process_full_text_to_json(self, text, base_name):
        """
        Process the full text at once to generate a JSON structure.
        """
        html = self.generate_html_structure(text)
        json_output = self.generate_json_with_prompt(html, base_name)
        return json_output

def process_input(input_link, model="gpt-4o-mini"):
    """
    Main function to process the input URL:
      - Fetch and extract text from the URL (PDF or HTML).
      - Preprocess the text.
      - Generate summaries using ChatGPT.
      - Process the full text into JSON.
      - All files are saved in a dedicated folder based on the URL.
    Returns a dictionary with the model used and the generated summaries.
    """
    try:
        processor = TextProcessor(model=model)
        logging.info(f"Processing input: {input_link}")
        result = processor.extract_text_from_url(input_link)

        # Handle error cases such as Google Cache links, blank PDFs, or unsupported types.
        if result["error"] == "google_cache":
            return {"error": "Skipping Google Cache link - no summary generated", "model": model}
        if result["content_type"] == "pdf" and result["error"] == "blank_pdf":
            return {"error": "Link doesn't have content to summarize - blank PDF", "model": model}
        if not result["text"]:
            return {"error": f"Error processing content: {result.get('error', 'Unknown error')}", "model": model}

        clean_text = processor.preprocess_text(result["text"])
        summaries = processor.generate_summaries_with_chatgpt(clean_text)
        base_name = processor.get_base_name_from_link(input_link)
        processor.process_full_text_to_json(clean_text, base_name)

        return {
            "model": model,
            "extractive": summaries["extractive"],
            "abstractive": summaries["abstractive"],
            "highlights": summaries["highlights"]
        }
    except Exception as e:
        logging.error(f"Error processing input: {str(e)}")
        return {"error": f"An error occurred: {str(e)}", "model": model}