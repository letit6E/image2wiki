import os
import json
import random
from pathlib import Path
from PIL import Image

from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import uuid

import torch
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoImageProcessor
from peft import PeftModel

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# --- Model Loading ---
MODEL_NAME = 'tuman/vit-rugpt2-image-captioning'
ADAPTER_DIR = 'letitbE/image2wiki-adapter'
DATA_ROOT = Path('.')

print("Loading base model...")
base_model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
print("Loading tokenizer and feature extractor...")
try:
    tok = AutoTokenizer.from_pretrained(ADAPTER_DIR)
except:
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
fe = AutoImageProcessor.from_pretrained(MODEL_NAME)

print("Resizing embeddings...")
base_model.decoder.resize_token_embeddings(len(tok))

print("Loading LoRA adapter...")
try:
    base_model.decoder = PeftModel.from_pretrained(base_model.decoder, ADAPTER_DIR)
except Exception as e:
    print(f"WARNING: Could not load adapter {ADAPTER_DIR}: {e}. Using base model.")
base_model.eval()

# --- Data Loading ---
print("Loading dataset...")
valid_arts = []
try:
    with open('data/metadata.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            art = json.loads(line)
            img_path = Path(art.get('image_path', ''))
            if not img_path.is_absolute():
                img_path = DATA_ROOT / img_path
            if img_path.exists():
                valid_arts.append(art)
    print(f"Found {len(valid_arts)} valid articles with images.")
except Exception as e:
    print(f"Error loading dataset: {e}")

def build_target(article):
    # Same logic as in finetune.ipynb
    parts = []
    if article.get('title'):
        parts.append(f"<title>{article['title']}")
    if article.get('lead'):
        parts.append(f"<lead>{article['lead']}")
    for sec in article.get('sections', []):
        if sec.get('title'):
            parts.append(f"<section>{sec['title']}")
        if sec.get('text'):
            parts.append(f"<paragraph>{sec['text']}")
    return "\n".join(parts)

def parse_generated_text(text):
    """Parses the raw generated text into HTML for the Wikipedia template."""
    title = "Сгенерированная статья"
    
    # Extract title if present
    if text.startswith('<title>'):
        parts = text.split('<title>', 1)[1]
        # Find next tag
        next_tag_idx = len(parts)
        for tag in ['<lead>', '<section>', '<paragraph>']:
            idx = parts.find(tag)
            if idx != -1 and idx < next_tag_idx:
                next_tag_idx = idx
        title = parts[:next_tag_idx].strip()
        text = parts[next_tag_idx:]
    elif '<lead>' in text:
        title = text.split('<lead>')[0].strip()
        text = '<lead>' + text.split('<lead>', 1)[1]

    # Replace tags with HTML
    html = text
    html = html.replace('<lead>', '<p>')
    html = html.replace('<paragraph>', '</p><p>')
    
    toc_items = []
    
    def section_replacer(match):
        content = match.group(1)
        # Split by first period or newline
        split_idx = len(content)
        period_idx = content.find('.')
        if period_idx != -1:
            split_idx = period_idx + 1
            
        heading = content[:split_idx].strip()
        rest = content[split_idx:].strip()
        
        sec_id = heading.replace(' ', '_').replace('"', '').replace("'", "")
        toc_items.append((sec_id, heading))
        
        res = f'</p><div class="mw-heading mw-heading2"><h2 id="{sec_id}">{heading}</h2></div>'
        if rest:
            res += f'<p>{rest}'
        return res

    import re
    html = re.sub(r'<section>(.*?)(?=<section>|<paragraph>|<lead>|$)', section_replacer, html, flags=re.DOTALL)
    
    # Clean up empty paragraphs
    html = html.replace('<p></p>', '')
    if not html.endswith('</p>'):
        html += '</p>'
        
    # Generate TOC HTML
    toc_html = ""
    for i, (sec_id, heading) in enumerate(toc_items, 1):
        toc_html += f'''
        <li id="toc-{sec_id}" class="vector-toc-list-item vector-toc-level-1">
            <a class="vector-toc-link" href="#{sec_id}">
                <div class="vector-toc-text">
                    <span class="vector-toc-numb">{i}</span>
                    <span>{heading}</span>
                </div>
            </a>
        </li>
        '''
        
    return title, html, toc_html

def generate_article_raw(image, model, tokenizer, feature_extractor):
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
    
    with torch.no_grad():
        output_ids = model.generate(
            pixel_values,
            max_new_tokens=512,
            # Для локального тестирования на CPU отключаем beam search (num_beams=1)
            # чтобы генерация работала в разы быстрее
            num_beams=1,
            no_repeat_ngram_size=3,
            decoder_start_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    # Remove bos/eos
    generated_text = generated_text.replace(tokenizer.bos_token, '').replace(tokenizer.eos_token, '').strip()
    return generated_text

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "search.html", {})

@app.post("/generate", response_class=HTMLResponse)
async def generate_article(request: Request, image: UploadFile = File(...)):
    # Save the uploaded image temporarily
    upload_dir = Path("app/static/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    file_ext = image.filename.split('.')[-1] if '.' in image.filename else 'jpg'
    filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = upload_dir / filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)
        
    # Open image for the model
    try:
        pil_image = Image.open(file_path).convert('RGB')
        
        # Generate text
        print(f"Generating article for {filename}...")
        generated_raw = generate_article_raw(pil_image, base_model, tok, fe)
        
        if not generated_raw.startswith('<title>'):
            generated_raw = "<title>" + generated_raw
            
        title, content_html, toc_html = parse_generated_text(generated_raw)
        
    except Exception as e:
        print(f"Error during generation: {e}")
        title = "Ошибка генерации"
        content_html = f"<p>Произошла ошибка при создании статьи: {str(e)}</p>"
        toc_html = ""
        
    img_url = f"/static/uploads/{filename}"
        
    return templates.TemplateResponse(request, "index.html", {
        "title": title,
        "content": content_html,
        "toc": toc_html,
        "image_url": img_url,
        "target_text": ""
    })

# Mount the root directory to serve images
import os

if os.path.exists("data"):
    app.mount("/data", StaticFiles(directory="data"), name="data")
