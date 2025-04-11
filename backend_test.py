import requests
import os
import pytest
from pathlib import Path

# Get the backend URL from environment or use default
BACKEND_URL = os.environ.get('REACT_APP_BACKEND_URL', 'http://localhost:8001')

class TestScriptBreakdownAPI:
    def setup_method(self):
        """Setup for each test method"""
        self.api_url = BACKEND_URL
        self.test_script_path = Path('/app/test_script.txt')
        self.script_id = None
        
        # Ensure test script exists
        assert self.test_script_path.exists(), "Test script file not found"

    def test_api_health(self):
        """Test API health endpoint"""
        response = requests.get(f"{self.api_url}/api")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Script Breakdown API"
        assert data["status"] == "active"
        print("✅ API health check passed")

    def test_script_upload(self):
        """Test script upload functionality"""
        # Prepare file upload
        files = {
            'file': ('test_script.txt', open(self.test_script_path, 'rb'), 'text/plain')
        }
        data = {'title': 'Test Script'}

        # Upload script
        response = requests.post(
            f"{self.api_url}/api/scripts/upload",
            files=files,
            data=data
        )
        
        assert response.status_code == 200, f"Upload failed with status {response.status_code}"
        
        upload_data = response.json()
        assert "script_id" in upload_data
        assert "statistics" in upload_data
        
        # Store script_id for subsequent tests
        self.script_id = upload_data["script_id"]
        print(f"✅ Script upload successful. Script ID: {self.script_id}")
        print(f"📊 Statistics: {upload_data['statistics']}")

    def test_character_extraction(self):
        """Test character extraction"""
        assert self.script_id, "Script ID not set - run upload test first"
        
        response = requests.get(f"{self.api_url}/api/scripts/{self.script_id}/characters")
        assert response.status_code == 200
        
        characters = response.json()
        character_names = [char["name"] for char in characters]
        
        # Check for expected characters
        assert "SARAH" in character_names, "SARAH not found in characters"
        assert "JAMES" in character_names, "JAMES not found in characters"
        
        print("✅ Character extraction test passed")
        print(f"📝 Found characters: {', '.join(character_names)}")

    def test_scene_extraction(self):
        """Test scene extraction"""
        assert self.script_id, "Script ID not set - run upload test first"
        
        response = requests.get(f"{self.api_url}/api/scripts/{self.script_id}/scenes")
        assert response.status_code == 200
        
        scenes = response.json()
        scene_headings = [scene["heading"] for scene in scenes]
        
        # Check for expected scenes
        expected_scenes = [
            "INT. COFFEE SHOP - MORNING",
            "EXT. COFFEE SHOP - MOMENTS LATER",
            "INT. COFFEE SHOP - FLASHBACK"
        ]
        
        for expected in expected_scenes:
            assert any(expected in heading for heading in scene_headings), f"Scene '{expected}' not found"
        
        print("✅ Scene extraction test passed")
        print(f"📝 Found scenes: {', '.join(scene_headings)}")

    def test_image_prompt_generation(self):
        """Test image prompt generation for a character"""
        assert self.script_id, "Script ID not set - run upload test first"
        
        # Get characters
        response = requests.get(f"{self.api_url}/api/scripts/{self.script_id}/characters")
        assert response.status_code == 200
        
        characters = response.json()
        assert len(characters) > 0, "No characters found"
        
        # Generate prompt for first character
        character = characters[0]
        response = requests.post(f"{self.api_url}/api/characters/{character['id']}/generate-prompt")
        assert response.status_code == 200
        
        prompt_data = response.json()
        assert "image_prompt" in prompt_data
        assert prompt_data["image_prompt"], "Empty prompt generated"
        
        print("✅ Image prompt generation test passed")
        print(f"🎭 Character: {character['name']}")
        print(f"🖼️ Generated prompt: {prompt_data['image_prompt']}")

if __name__ == "__main__":
    # Run tests
    test = TestScriptBreakdownAPI()
    test.setup_method()
    
    try:
        test.test_api_health()
        test.test_script_upload()
        test.test_character_extraction()
        test.test_scene_extraction()
        test.test_image_prompt_generation()
        print("\n✨ All tests passed successfully!")
    except Exception as e:
        print(f"\n❌ Tests failed: {str(e)}")
