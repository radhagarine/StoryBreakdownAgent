from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import uvicorn
import os
import logging
import uuid
import json
from pathlib import Path
import io
import re
from datetime import datetime

# Additional imports for file format handling
import PyPDF2
import docx
import pandas as pd
import csv
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Load environment variables
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL')
client = AsyncIOMotorClient(mongo_url)
db = client.get_database(os.environ.get('DB_NAME', 'scriptbreakdown'))

app = FastAPI(title="Script Breakdown API")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----- Models -----

class Character(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    appearance: Optional[str] = None
    traits: Optional[List[str]] = None
    image_prompt: Optional[str] = None
    script_id: str

class Scene(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scene_number: Optional[str] = None
    heading: str
    description: Optional[str] = None
    characters: List[str] = []
    location: Optional[str] = None
    time_of_day: Optional[str] = None
    image_prompt: Optional[str] = None
    script_id: str

class Shot(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    shot_number: Optional[str] = None
    description: str
    camera_angle: Optional[str] = None
    scene_id: str
    image_prompt: Optional[str] = None
    script_id: str

class Script(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    content: str
    format: str
    uploaded_at: datetime = Field(default_factory=datetime.now)
    parsed: bool = False

class ImagePrompt(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    entity_type: str  # "character", "scene", or "shot"
    entity_id: str
    prompt: str
    script_id: str
    created_at: datetime = Field(default_factory=datetime.now)

# ----- Helper Functions -----

async def parse_script(script_content, script_id):
    """
    Parse a movie script and extract characters, scenes, and shots using NLP techniques.
    """
    logger.info(f"Parsing script {script_id}")
    
    try:
        # Process the script with spaCy for NLP analysis
        doc = nlp(script_content[:1000000])  # Limit size to avoid memory issues
        
        # Initialize sentiment analyzer
        sia = SentimentIntensityAnalyzer()
        
        # Improved regex patterns for script elements
        scene_heading_pattern = r'(INT\.|EXT\.|INT\/EXT\.)\s*(.*?)(?=\n)'
        # Improved character pattern to better catch script character names
        character_pattern = r'\n([A-Z][A-Z\s]+)(?:\s*\(.*?\))?\n'
        
        # Extract scenes
        scenes = []
        scene_matches = list(re.finditer(scene_heading_pattern, script_content, re.MULTILINE))
        
        logger.info(f"Found {len(scene_matches)} potential scenes")
        
        for i, match in enumerate(scene_matches):
            try:
                heading = match.group(0).strip()
                location = heading.split(' - ')[0] if ' - ' in heading else heading
                time_of_day = heading.split(' - ')[1] if ' - ' in heading else None
                
                # Get content until next scene or end of script
                if i < len(scene_matches) - 1:
                    next_match = scene_matches[i + 1]
                    description = script_content[match.end():next_match.start()].strip()
                else:
                    description = script_content[match.end():].strip()
                
                # Analyze scene sentiment/mood using NLTK
                if description:
                    scene_mood = "neutral"
                    sentiment_scores = sia.polarity_scores(description[:1000])  # Analyze first 1000 chars
                    if sentiment_scores['compound'] >= 0.25:
                        scene_mood = "positive"
                    elif sentiment_scores['compound'] <= -0.25:
                        scene_mood = "negative"
                    
                    # More specific mood detection
                    if "danger" in description.lower() or "fear" in description.lower():
                        scene_mood = "tense"
                    elif "love" in description.lower() or "kiss" in description.lower():
                        scene_mood = "romantic"
                    elif "laugh" in description.lower() or "joke" in description.lower():
                        scene_mood = "comedic"
                else:
                    scene_mood = "neutral"
                
                # Create scene with UUID
                scene_id = str(uuid.uuid4())
                scene = {
                    "id": scene_id,
                    "scene_number": str(i + 1),
                    "heading": heading,
                    "description": description,
                    "location": location,
                    "time_of_day": time_of_day,
                    "mood": scene_mood,
                    "characters": [],
                    "script_id": script_id
                }
                
                # Generate image prompt for the scene
                scene["image_prompt"] = generate_scene_prompt(scene)
                
                scenes.append(scene)
                
                # Store in database
                logger.info(f"Saving scene: {scene['heading']}")
                result = await db.scenes.insert_one(scene)
                logger.info(f"Scene saved with ID: {scene_id}")
                
            except Exception as e:
                logger.error(f"Error processing scene {i}: {str(e)}")
        
        # Extract characters
        characters = {}
        character_matches = list(re.finditer(character_pattern, script_content, re.MULTILINE))
        
        logger.info(f"Found {len(character_matches)} potential characters")
        
        for match in character_matches:
            try:
                character_name = match.group(1).strip()
                
                # Skip if it's a scene heading or already processed
                if character_name in characters or "INT." in character_name or "EXT." in character_name:
                    continue
                
                logger.info(f"Processing character: {character_name}")
                
                # Find character dialogue - gather all lines for this character
                dialogue_matches = re.finditer(rf'\n{re.escape(character_name)}\s*(?:\(.*?\))?\n(.*?)(?=\n\n|\n[A-Z][A-Z\s]+(?:\(.*?\))?\n)', script_content, re.DOTALL)
                
                dialogue_samples = []
                for d_match in dialogue_matches:
                    if d_match.group(1):
                        dialogue_samples.append(d_match.group(1).strip())
                
                # Extract character traits using NLP
                traits = []
                appearance = []
                
                # Analyze character's dialogue for sentiment/personality traits
                if dialogue_samples:
                    combined_dialogue = " ".join(dialogue_samples[:5])  # Use first 5 samples
                    
                    # Sentiment analysis for emotional traits
                    sentiment = sia.polarity_scores(combined_dialogue)
                    if sentiment['compound'] >= 0.25:
                        traits.append("optimistic")
                    elif sentiment['compound'] <= -0.25:
                        traits.append("pessimistic")
                    
                    # Look for descriptive words in surrounding context
                    char_pos = script_content.find(character_name)
                    context = script_content[max(0, char_pos-300):min(len(script_content), char_pos+300)]
                    
                    # Use spaCy for more accurate description extraction
                    context_doc = nlp(context)
                    
                    # Look for adjectives that might describe the character
                    for sent in context_doc.sents:
                        sent_text = sent.text.lower()
                        if character_name.lower() in sent_text:
                            for token in sent:
                                if token.pos_ == "ADJ":
                                    if len(token.text) > 2:  # Avoid small words
                                        traits.append(token.text.lower())
                                # Look for appearance descriptions
                                if token.pos_ == "NOUN" and any(word in token.text.lower() for word in ["hair", "eyes", "face", "tall", "short", "wearing"]):
                                    appearance.append(token.text.lower())
                
                # Find character descriptions - look for surrounding context
                char_pos = script_content.find(character_name)
                surrounding_text = script_content[max(0, char_pos-300):min(len(script_content), char_pos+300)]
                
                # Look for description in surrounding text
                description_match = re.search(rf'{character_name}[,\.]?\s*(.*?)(?=\n|\.|,)', surrounding_text)
                character_desc = description_match.group(1).strip() if description_match else f"Character from the script named {character_name}"
                
                # Extract appearance details if available
                appearance_text = ", ".join(appearance[:3]) if appearance else "Appearance details not specified in script"
                
                # Character ID with UUID
                character_id = str(uuid.uuid4())
                character = {
                    "id": character_id,
                    "name": character_name,
                    "description": character_desc,
                    "appearance": appearance_text,
                    "traits": traits[:5],  # Limit to top 5 traits
                    "dialogue_samples": dialogue_samples[:3] if dialogue_samples else [],
                    "script_id": script_id
                }
                
                # Generate image prompt for the character
                character["image_prompt"] = generate_character_prompt(character)
                
                characters[character_name] = character
                
                # Store in database
                logger.info(f"Saving character: {character['name']}")
                result = await db.characters.insert_one(character)
                logger.info(f"Character saved with ID: {character_id}")
                
                # Update scenes where this character appears
                for scene in scenes:
                    if character_name in scene["description"]:
                        scene["characters"].append(character_name)
                        await db.scenes.update_one(
                            {"id": scene["id"]}, 
                            {"$set": {"characters": scene["characters"]}}
                        )
            except Exception as e:
                logger.error(f"Error processing character {match.group(1) if match.group(1) else 'unknown'}: {str(e)}")
        
        # Extract shots (simplified - in real scripts, shots are more complex)
        # Enhanced shot detection with camera movement and framing keywords
        shot_pattern = r'(?:ANGLE ON|CLOSE ON|WIDE SHOT|POV|TRACKING SHOT|DOLLY IN|PAN TO|ZOOM IN|ZOOM OUT|CRANE SHOT)(.*?)(?=\n\n|$)'
        
        for scene in scenes:
            try:
                # Look for potential shot descriptions
                shot_descriptions = re.findall(shot_pattern, scene["description"], re.MULTILINE | re.IGNORECASE)
                
                for i, shot_desc in enumerate(shot_descriptions):
                    shot_id = str(uuid.uuid4())
                    
                    # Analyze shot sentiment/mood
                    shot_mood = "neutral"
                    if shot_desc:
                        sentiment_scores = sia.polarity_scores(shot_desc)
                        if sentiment_scores['compound'] >= 0.25:
                            shot_mood = "positive"
                        elif sentiment_scores['compound'] <= -0.25:
                            shot_mood = "negative"
                    
                    shot = {
                        "id": shot_id,
                        "shot_number": f"{scene['scene_number']}-{i+1}",
                        "description": shot_desc.strip(),
                        "camera_angle": extract_camera_angle(shot_desc),
                        "mood": shot_mood,
                        "scene_id": scene["id"],
                        "script_id": script_id
                    }
                    
                    # Generate image prompt for the shot
                    shot["image_prompt"] = generate_shot_prompt(shot, scene)
                    
                    # Store in database
                    logger.info(f"Saving shot: {shot['shot_number']}")
                    result = await db.shots.insert_one(shot)
                    logger.info(f"Shot saved with ID: {shot_id}")
            except Exception as e:
                logger.error(f"Error processing shots for scene {scene.get('id')}: {str(e)}")
        
        # Mark script as parsed
        await db.scripts.update_one(
            {"id": script_id},
            {"$set": {"parsed": True}}
        )
        
        return {
            "scenes": len(scenes),
            "characters": len(characters)
        }
    except Exception as e:
        logger.error(f"Error in parse_script: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Script parsing failed: {str(e)}")
        

def extract_camera_angle(shot_description):
    """Extract camera angle from shot description"""
    angles = ["WIDE", "CLOSE UP", "MEDIUM", "OVERHEAD", "LOW ANGLE", "POV", "TRACKING"]
    for angle in angles:
        if angle in shot_description.upper():
            return angle
    return "Not specified"

def generate_character_prompt(character):
    """Generate a detailed prompt for AI image generation of a character"""
    prompt = f"A detailed portrait of {character['name']}, "
    
    if character.get("description"):
        prompt += f"{character['description']}. "
    
    if character.get("appearance"):
        prompt += f"Physical appearance: {character['appearance']}. "
    
    if character.get("traits") and len(character["traits"]) > 0:
        traits_text = ", ".join(character["traits"])
        prompt += f"Character traits include: {traits_text}. "
    
    # Add emotional context if available
    if character.get("emotion"):
        prompt += f"Their expression shows {character['emotion']}. "
    
    # Add clothing context if available
    if character.get("clothing"):
        prompt += f"Dressed in {character['clothing']}. "
    
    # Add enhanced visual style guidance
    prompt += "The image should have professional cinematic lighting with rim lighting on the subject, "
    prompt += "shallow depth of field with background bokeh, high detail, photorealistic style with dramatic composition. "
    prompt += "The portrait should capture the character's essence with subtle facial nuances, appropriate for a movie poster or character study."
    
    return prompt

def generate_scene_prompt(scene):
    """Generate a detailed prompt for AI image generation of a scene"""
    prompt = f"A cinematic shot of {scene['heading']}. "
    
    if scene.get("description"):
        # Take just the first 200 characters of description to keep prompt focused
        short_desc = scene["description"][:200] + "..." if len(scene["description"]) > 200 else scene["description"]
        prompt += f"Scene description: {short_desc}. "
    
    if scene.get("time_of_day"):
        # Enhanced time of day descriptions
        time_desc = scene.get("time_of_day").lower()
        if "morning" in time_desc:
            prompt += f"Time of day: Early morning with warm golden sunlight casting long shadows. "
        elif "noon" in time_desc or "day" in time_desc:
            prompt += f"Time of day: Bright midday with harsh overhead lighting and short shadows. "
        elif "afternoon" in time_desc:
            prompt += f"Time of day: Late afternoon with warm, amber lighting and medium-length shadows. "
        elif "evening" in time_desc:
            prompt += f"Time of day: Early evening with soft, diffused golden hour lighting. "
        elif "night" in time_desc:
            prompt += f"Time of day: Nighttime with cool blue moonlight or artificial lighting creating strong contrast. "
        else:
            prompt += f"Time of day: {scene['time_of_day']}. "
    
    if scene.get("location"):
        prompt += f"Location: {scene['location']}. "
    
    # Add weather/atmosphere if mentioned in the description
    if scene.get("description"):
        desc_lower = scene["description"].lower()
        if any(word in desc_lower for word in ["rain", "raining", "storm", "thunder"]):
            prompt += "The scene has rainy weather with wet surfaces reflecting light. "
        elif any(word in desc_lower for word in ["snow", "snowing", "winter", "cold"]):
            prompt += "The scene has snowy conditions with a cold, blue-tinted color palette. "
        elif any(word in desc_lower for word in ["fog", "foggy", "mist", "misty"]):
            prompt += "The scene has fog or mist creating an atmospheric, diffused lighting effect. "
    
    # Add enhanced visual style guidance
    prompt += "The image should have professional cinematic lighting with motivated light sources, "
    prompt += "film grain texture for authenticity, dramatic composition with proper depth of field following the rule of thirds, "
    prompt += "high detail in the focal points with atmospheric perspective, photorealistic style with color grading appropriate for the scene's mood."
    
    return prompt

def generate_shot_prompt(shot, related_scene):
    """Generate a detailed prompt for AI image generation of a specific shot"""
    # Enhanced camera angle descriptions
    camera_angle = shot.get('camera_angle', '').lower()
    if "wide" in camera_angle:
        prompt = "A wide establishing shot showing the full scene with emphasis on the environment and spatial relationships. "
    elif "close up" in camera_angle or "closeup" in camera_angle:
        prompt = "An intimate close-up shot focusing on facial expressions or important details with shallow depth of field. "
    elif "medium" in camera_angle:
        prompt = "A medium shot framing characters from approximately waist up, balanced to show both facial expressions and body language. "
    elif "overhead" in camera_angle:
        prompt = "A dramatic overhead shot looking directly down on the scene, creating a voyeuristic or omniscient perspective. "
    elif "low angle" in camera_angle:
        prompt = "A dynamic low-angle shot looking upward at the subject, creating a sense of power or dominance. "
    elif "pov" in camera_angle:
        prompt = "A subjective point-of-view shot from a character's perspective, slightly unsteady with natural motion. "
    elif "tracking" in camera_angle:
        prompt = "A smooth tracking shot following the subject in motion, with slight motion blur on the background. "
    else:
        prompt = f"A {shot.get('camera_angle', 'cinematic')} shot from a movie scene. "
    
    if shot.get("description"):
        prompt += f"Shot description: {shot['description']}. "
    
    if related_scene.get("location"):
        prompt += f"Location: {related_scene['location']}. "
    
    if related_scene.get("time_of_day"):
        prompt += f"Time of day: {related_scene['time_of_day']}. "
    
    # Add mood/tone analysis based on description
    if shot.get("description"):
        desc_lower = shot["description"].lower()
        
        # Analyze for tension/suspense
        if any(word in desc_lower for word in ["tense", "suspense", "fear", "scary", "afraid", "nervous"]):
            prompt += "The shot has a suspenseful mood with high contrast lighting, shadows, and uneasy composition. "
        
        # Analyze for action
        elif any(word in desc_lower for word in ["action", "fight", "chase", "run", "explosion", "fast"]):
            prompt += "The shot captures dynamic action with motion blur, energetic composition, and heightened contrast. "
        
        # Analyze for romance/intimacy
        elif any(word in desc_lower for word in ["love", "kiss", "embrace", "intimate", "tender", "romance"]):
            prompt += "The shot has romantic mood with soft, flattering lighting, warm color palette, and intimate framing. "
        
        # Analyze for sadness/melancholy
        elif any(word in desc_lower for word in ["sad", "grief", "cry", "tears", "mourn", "sorrow"]):
            prompt += "The shot has a melancholic mood with desaturated colors, soft lighting, and somber composition. "
    
    # Add enhanced visual style guidance
    prompt += "The image should have professional cinematic lighting with motivated light sources and appropriate shadows, "
    prompt += "film grain texture for authentic cinematic feel, dramatic composition following cinematography principles, "
    prompt += "appropriate depth of field for the shot type, high detail in the focal points, photorealistic style with "
    prompt += "color grading that enhances the emotional tone of the scene."
    
    return prompt

# ----- API Endpoints -----

@app.get("/api")
async def root():
    return {"message": "Script Breakdown API", "status": "active"}

@app.post("/api/scripts/upload")
async def upload_script(
    file: UploadFile = File(...),
    title: str = Form(None)
):
    """Upload and process a script file"""
    try:
        # Read file content
        content = await file.read()
        
        # Determine file format
        file_format = file.filename.split('.')[-1].lower()
        
        # Use filename as title if not provided
        if not title:
            title = file.filename.split('.')[0]
        
        # Process file based on format
        if file_format == 'pdf':
            script_content = extract_text_from_pdf(content)
        elif file_format in ['doc', 'docx']:
            script_content = extract_text_from_docx(content)
        elif file_format == 'csv':
            script_content = extract_text_from_csv(content)
        else:
            # Default to treating as text
            script_content = content.decode('utf-8')
        
        # Create script record
        script = {
            "id": str(uuid.uuid4()),
            "title": title,
            "content": script_content,
            "format": file_format,
            "uploaded_at": datetime.now(),
            "parsed": False
        }
        
        # Save to database
        await db.scripts.insert_one(script)
        
        # Parse script (in a real app, this might be a background task)
        parse_results = await parse_script(script_content, script["id"])
        
        return {
            "message": f"Script uploaded and processed successfully (format: {file_format})",
            "script_id": script["id"],
            "title": script["title"],
            "statistics": parse_results
        }
    except Exception as e:
        logger.error(f"Error processing script: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process script: {str(e)}")

def extract_text_from_pdf(content):
    """Extract text from PDF file"""
    try:
        with io.BytesIO(content) as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")

def extract_text_from_docx(content):
    """Extract text from DOCX file"""
    try:
        with io.BytesIO(content) as docx_file:
            doc = docx.Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {str(e)}")
        raise ValueError(f"Failed to extract text from DOCX: {str(e)}")

def extract_text_from_csv(content):
    """Extract text from CSV file (assuming it's a screenplay in CSV format)"""
    try:
        with io.StringIO(content.decode('utf-8')) as csv_file:
            reader = csv.reader(csv_file)
            text = ""
            for row in reader:
                # Join row elements with spaces
                text += " ".join(row) + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from CSV: {str(e)}")
        raise ValueError(f"Failed to extract text from CSV: {str(e)}")

@app.get("/api/scripts")
async def get_scripts():
    """Get list of all scripts"""
    scripts = await db.scripts.find({}, {"content": 0}).to_list(length=100)
    return jsonable_encoder(scripts)

@app.get("/api/scripts/{script_id}")
async def get_script(script_id: str):
    """Get script details"""
    script = await db.scripts.find_one({"id": script_id})
    if not script:
        raise HTTPException(status_code=404, detail="Script not found")
    return jsonable_encoder(script)

@app.get("/api/scripts/{script_id}/characters")
async def get_script_characters(script_id: str):
    """Get all characters from a script"""
    try:
        # Convert MongoDB cursor to list of dictionaries
        characters = []
        async for char in db.characters.find({"script_id": script_id}):
            # Ensure _id is not included in the response
            if "_id" in char:
                del char["_id"]
            characters.append(char)
        
        logger.info(f"Retrieved {len(characters)} characters for script {script_id}")
        return jsonable_encoder(characters)
    except Exception as e:
        logger.error(f"Error retrieving characters: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve characters: {str(e)}")

@app.get("/api/scripts/{script_id}/scenes")
async def get_script_scenes(script_id: str):
    """Get all scenes from a script"""
    try:
        # Convert MongoDB cursor to list of dictionaries
        scenes = []
        async for scene in db.scenes.find({"script_id": script_id}):
            # Ensure _id is not included in the response
            if "_id" in scene:
                del scene["_id"]
            scenes.append(scene)
        
        logger.info(f"Retrieved {len(scenes)} scenes for script {script_id}")
        return jsonable_encoder(scenes)
    except Exception as e:
        logger.error(f"Error retrieving scenes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve scenes: {str(e)}")

@app.get("/api/scripts/{script_id}/shots")
async def get_script_shots(script_id: str):
    """Get all shots from a script"""
    try:
        # Convert MongoDB cursor to list of dictionaries
        shots = []
        async for shot in db.shots.find({"script_id": script_id}):
            # Ensure _id is not included in the response
            if "_id" in shot:
                del shot["_id"]
            shots.append(shot)
        
        logger.info(f"Retrieved {len(shots)} shots for script {script_id}")
        return jsonable_encoder(shots)
    except Exception as e:
        logger.error(f"Error retrieving shots: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve shots: {str(e)}")


@app.get("/api/characters/{character_id}")
async def get_character(character_id: str):
    """Get details of a specific character"""
    character = await db.characters.find_one({"id": character_id})
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    
    # Remove _id field which is not JSON serializable
    if "_id" in character:
        del character["_id"]
        
    return jsonable_encoder(character)

@app.get("/api/scenes/{scene_id}")
async def get_scene(scene_id: str):
    """Get details of a specific scene"""
    scene = await db.scenes.find_one({"id": scene_id})
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")
    
    # Remove _id field which is not JSON serializable
    if "_id" in scene:
        del scene["_id"]
        
    return jsonable_encoder(scene)

@app.get("/api/shots/{shot_id}")
async def get_shot(shot_id: str):
    """Get details of a specific shot"""
    shot = await db.shots.find_one({"id": shot_id})
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")
    
    # Remove _id field which is not JSON serializable
    if "_id" in shot:
        del shot["_id"]
        
    return jsonable_encoder(shot)

@app.put("/api/characters/{character_id}")
async def update_character(character_id: str, character: dict = Body(...)):
    """Update character details"""
    # Ensure id cannot be changed
    if "id" in character:
        del character["id"]
    
    result = await db.characters.update_one(
        {"id": character_id}, 
        {"$set": character}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Character not found")
    
    updated_character = await db.characters.find_one({"id": character_id})
    return jsonable_encoder(updated_character)

@app.put("/api/scenes/{scene_id}")
async def update_scene(scene_id: str, scene: dict = Body(...)):
    """Update scene details"""
    # Ensure id cannot be changed
    if "id" in scene:
        del scene["id"]
    
    result = await db.scenes.update_one(
        {"id": scene_id}, 
        {"$set": scene}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Scene not found")
    
    updated_scene = await db.scenes.find_one({"id": scene_id})
    return jsonable_encoder(updated_scene)

@app.put("/api/shots/{shot_id}")
async def update_shot(shot_id: str, shot: dict = Body(...)):
    """Update shot details"""
    # Ensure id cannot be changed
    if "id" in shot:
        del shot["id"]
    
    result = await db.shots.update_one(
        {"id": shot_id}, 
        {"$set": shot}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Shot not found")
    
    updated_shot = await db.shots.find_one({"id": shot_id})
    return jsonable_encoder(updated_shot)

@app.post("/api/characters/{character_id}/generate-prompt")
async def generate_character_image_prompt(character_id: str):
    """Generate or regenerate an image prompt for a character"""
    character = await db.characters.find_one({"id": character_id})
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    
    # Generate new prompt
    prompt = generate_character_prompt(character)
    
    # Update character
    await db.characters.update_one(
        {"id": character_id},
        {"$set": {"image_prompt": prompt}}
    )
    
    # Store prompt in prompts collection
    image_prompt = {
        "id": str(uuid.uuid4()),
        "entity_type": "character",
        "entity_id": character_id,
        "prompt": prompt,
        "script_id": character["script_id"],
        "created_at": datetime.now()
    }
    
    await db.image_prompts.insert_one(image_prompt)
    
    return {
        "character_id": character_id,
        "character_name": character["name"],
        "image_prompt": prompt
    }

@app.post("/api/scenes/{scene_id}/generate-prompt")
async def generate_scene_image_prompt(scene_id: str):
    """Generate or regenerate an image prompt for a scene"""
    scene = await db.scenes.find_one({"id": scene_id})
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")
    
    # Generate new prompt
    prompt = generate_scene_prompt(scene)
    
    # Update scene
    await db.scenes.update_one(
        {"id": scene_id},
        {"$set": {"image_prompt": prompt}}
    )
    
    # Store prompt in prompts collection
    image_prompt = {
        "id": str(uuid.uuid4()),
        "entity_type": "scene",
        "entity_id": scene_id,
        "prompt": prompt,
        "script_id": scene["script_id"],
        "created_at": datetime.now()
    }
    
    await db.image_prompts.insert_one(image_prompt)
    
    return {
        "scene_id": scene_id,
        "scene_heading": scene["heading"],
        "image_prompt": prompt
    }

@app.post("/api/shots/{shot_id}/generate-prompt")
async def generate_shot_image_prompt(shot_id: str):
    """Generate or regenerate an image prompt for a shot"""
    shot = await db.shots.find_one({"id": shot_id})
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")
    
    # Get related scene info
    scene = await db.scenes.find_one({"id": shot["scene_id"]})
    if not scene:
        scene = {}  # Fallback if scene not found
    
    # Generate new prompt
    prompt = generate_shot_prompt(shot, scene)
    
    # Update shot
    await db.shots.update_one(
        {"id": shot_id},
        {"$set": {"image_prompt": prompt}}
    )
    
    # Store prompt in prompts collection
    image_prompt = {
        "id": str(uuid.uuid4()),
        "entity_type": "shot",
        "entity_id": shot_id,
        "prompt": prompt,
        "script_id": shot["script_id"],
        "created_at": datetime.now()
    }
    
    await db.image_prompts.insert_one(image_prompt)
    
    return {
        "shot_id": shot_id,
        "description": shot["description"],
        "image_prompt": prompt
    }

@app.get("/api/prompts/{entity_type}/{entity_id}")
async def get_prompts_for_entity(entity_type: str, entity_id: str):
    """Get all prompts generated for a specific entity"""
    prompts = await db.image_prompts.find({
        "entity_type": entity_type,
        "entity_id": entity_id
    }).sort("created_at", -1).to_list(length=50)
    
    return jsonable_encoder(prompts)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
