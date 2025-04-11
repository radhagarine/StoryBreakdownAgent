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
    Parse a movie script and extract characters, scenes, and shots.
    """
    logger.info(f"Parsing script {script_id}")
    
    # Basic regex patterns for script elements
    scene_heading_pattern = r'(INT\.|EXT\.|INT/EXT\.)(.*?)(?=\n)'
    character_pattern = r'\n([A-Z][A-Z\s]+)(?:\(.*?\))?\n'
    
    # Extract scenes
    scenes = []
    scene_matches = re.finditer(scene_heading_pattern, script_content, re.MULTILINE)
    
    for i, match in enumerate(scene_matches):
        heading = match.group(0).strip()
        location = heading.split(' - ')[0] if ' - ' in heading else heading
        time_of_day = heading.split(' - ')[1] if ' - ' in heading else None
        
        # Get content until next scene or end of script
        next_pos = script_content.find('INT.', match.end())
        if next_pos == -1:
            next_pos = script_content.find('EXT.', match.end())
        
        if next_pos == -1:
            description = script_content[match.end():].strip()
        else:
            description = script_content[match.end():next_pos].strip()
        
        # Create scene
        scene = {
            "id": str(uuid.uuid4()),
            "scene_number": str(i + 1),
            "heading": heading,
            "description": description,
            "location": location,
            "time_of_day": time_of_day,
            "characters": [],
            "script_id": script_id
        }
        
        # Generate image prompt for the scene
        scene["image_prompt"] = generate_scene_prompt(scene)
        
        scenes.append(scene)
        
        # Store in database
        await db.scenes.insert_one(scene)
    
    # Extract characters
    characters = {}
    character_matches = re.finditer(character_pattern, script_content, re.MULTILINE)
    
    for match in character_matches:
        character_name = match.group(1).strip()
        
        if character_name not in characters and not character_name.isupper():
            # Skip scene headings and other all-caps text that aren't characters
            
            # Find character descriptions - assume they might be near first mention
            char_pos = script_content.find(character_name)
            description_area = script_content[max(0, char_pos-200):char_pos+200]
            
            # Simple description extraction - could be enhanced with NLP
            description = f"Character from the script named {character_name}"
            
            character = {
                "id": str(uuid.uuid4()),
                "name": character_name,
                "description": description,
                "appearance": "Appearance details not specified in script",
                "traits": [],
                "script_id": script_id
            }
            
            # Generate image prompt for the character
            character["image_prompt"] = generate_character_prompt(character)
            
            characters[character_name] = character
            
            # Store in database
            await db.characters.insert_one(character)
            
            # Update scenes where this character appears
            for scene in scenes:
                if character_name in scene["description"]:
                    scene["characters"].append(character_name)
                    await db.scenes.update_one(
                        {"id": scene["id"]}, 
                        {"$set": {"characters": scene["characters"]}}
                    )
    
    # Extract shots (simplified - in real scripts, shots are more complex)
    for scene in scenes:
        # Look for potential shot descriptions
        shot_descriptions = re.findall(r'(?:ANGLE ON|CLOSE ON|WIDE SHOT|POV)(.*?)(?=\n\n)', scene["description"], re.MULTILINE)
        
        for i, shot_desc in enumerate(shot_descriptions):
            shot = {
                "id": str(uuid.uuid4()),
                "shot_number": f"{scene['scene_number']}-{i+1}",
                "description": shot_desc.strip(),
                "camera_angle": extract_camera_angle(shot_desc),
                "scene_id": scene["id"],
                "script_id": script_id
            }
            
            # Generate image prompt for the shot
            shot["image_prompt"] = generate_shot_prompt(shot, scene)
            
            # Store in database
            await db.shots.insert_one(shot)
    
    # Mark script as parsed
    await db.scripts.update_one(
        {"id": script_id},
        {"$set": {"parsed": True}}
    )
    
    return {
        "scenes": len(scenes),
        "characters": len(characters)
    }

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
    
    prompt += "The image should have professional cinematic lighting, high detail, photorealistic style with dramatic composition."
    
    return prompt

def generate_scene_prompt(scene):
    """Generate a detailed prompt for AI image generation of a scene"""
    prompt = f"A cinematic shot of {scene['heading']}. "
    
    if scene.get("description"):
        # Take just the first 200 characters of description to keep prompt focused
        short_desc = scene["description"][:200] + "..." if len(scene["description"]) > 200 else scene["description"]
        prompt += f"Scene description: {short_desc}. "
    
    if scene.get("time_of_day"):
        prompt += f"Time of day: {scene['time_of_day']}. "
    
    if scene.get("location"):
        prompt += f"Location: {scene['location']}. "
    
    prompt += "The image should have professional cinematic lighting, film grain texture, dramatic composition with depth of field, high detail, photorealistic style."
    
    return prompt

def generate_shot_prompt(shot, related_scene):
    """Generate a detailed prompt for AI image generation of a specific shot"""
    prompt = f"A {shot.get('camera_angle', 'cinematic')} shot from a movie scene. "
    
    if shot.get("description"):
        prompt += f"Shot description: {shot['description']}. "
    
    if related_scene.get("location"):
        prompt += f"Location: {related_scene['location']}. "
    
    if related_scene.get("time_of_day"):
        prompt += f"Time of day: {related_scene['time_of_day']}. "
    
    prompt += "The image should have professional cinematic lighting, film grain texture, dramatic composition with appropriate depth of field, high detail, photorealistic style."
    
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
        
        # For simplicity, we're treating all formats as text in this MVP
        # In a production app, we'd use specialized libraries for each format
        script_content = content.decode('utf-8')
        
        # Use filename as title if not provided
        if not title:
            title = file.filename.split('.')[0]
        
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
            "message": "Script uploaded and processed successfully",
            "script_id": script["id"],
            "title": script["title"],
            "statistics": parse_results
        }
    except Exception as e:
        logger.error(f"Error processing script: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process script: {str(e)}")

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
    characters = await db.characters.find({"script_id": script_id}).to_list(length=100)
    return jsonable_encoder(characters)

@app.get("/api/scripts/{script_id}/scenes")
async def get_script_scenes(script_id: str):
    """Get all scenes from a script"""
    scenes = await db.scenes.find({"script_id": script_id}).to_list(length=100)
    return jsonable_encoder(scenes)

@app.get("/api/scripts/{script_id}/shots")
async def get_script_shots(script_id: str):
    """Get all shots from a script"""
    shots = await db.shots.find({"script_id": script_id}).to_list(length=100)
    return jsonable_encoder(shots)

@app.get("/api/characters/{character_id}")
async def get_character(character_id: str):
    """Get details of a specific character"""
    character = await db.characters.find_one({"id": character_id})
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    return jsonable_encoder(character)

@app.get("/api/scenes/{scene_id}")
async def get_scene(scene_id: str):
    """Get details of a specific scene"""
    scene = await db.scenes.find_one({"id": scene_id})
    if not scene:
        raise HTTPException(status_code=404, detail="Scene not found")
    return jsonable_encoder(scene)

@app.get("/api/shots/{shot_id}")
async def get_shot(shot_id: str):
    """Get details of a specific shot"""
    shot = await db.shots.find_one({"id": shot_id})
    if not shot:
        raise HTTPException(status_code=404, detail="Shot not found")
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
