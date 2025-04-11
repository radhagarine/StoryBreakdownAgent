import requests
import pytest
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get backend URL from environment
BACKEND_URL = "https://ecd1d434-35c7-4302-9091-26f6bdc8e2ab.preview.emergentagent.com"

class TestScriptBreakdownAPI:
    def __init__(self):
        self.base_url = BACKEND_URL
        self.test_script_content = """
INT. COFFEE SHOP - MORNING

The morning sun streams through large windows. SARAH (28, ambitious writer) sits at a corner table, laptop open.

JOHN (35, rugged detective) enters, scanning the room with tired eyes.

SARAH
(looking up)
You look like you haven't slept.

JOHN
Murder cases don't sleep.
(sits down)
Neither do the detectives working them.

TRACKING SHOT - Following a waitress as she brings coffee to their table.

CLOSE UP on Sarah's hands as she closes her laptop.

SARAH
Tell me about the case.

WIDE SHOT of the coffee shop as morning customers stream in.
"""
        self.script_id = None
        self.character_id = None
        self.scene_id = None
        self.shot_id = None

    def test_api_health(self):
        """Test API health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/api")
            assert response.status_code == 200
            assert response.json()["message"] == "Script Breakdown API"
            logger.info("✅ API health check passed")
            return True
        except Exception as e:
            logger.error(f"❌ API health check failed: {str(e)}")
            return False

    def test_script_upload(self):
        """Test script upload endpoint"""
        try:
            # Create a temporary file
            with open("test_script.txt", "w") as f:
                f.write(self.test_script_content)

            # Upload the script
            files = {
                'file': ('test_script.txt', open('test_script.txt', 'rb'), 'text/plain')
            }
            data = {
                'title': 'Test Script'
            }
            response = requests.post(
                f"{self.base_url}/api/scripts/upload",
                files=files,
                data=data
            )

            assert response.status_code == 200
            result = response.json()
            assert "script_id" in result
            self.script_id = result["script_id"]
            logger.info("✅ Script upload test passed")
            logger.info(f"Script ID: {self.script_id}")
            
            # Clean up
            os.remove("test_script.txt")
            return True
        except Exception as e:
            logger.error(f"❌ Script upload test failed: {str(e)}")
            return False

    def test_get_scripts(self):
        """Test get all scripts endpoint"""
        try:
            response = requests.get(f"{self.base_url}/api/scripts")
            assert response.status_code == 200
            scripts = response.json()
            assert isinstance(scripts, list)
            logger.info("✅ Get scripts test passed")
            return True
        except Exception as e:
            logger.error(f"❌ Get scripts test failed: {str(e)}")
            return False

    def test_get_script_details(self):
        """Test get script details endpoint"""
        try:
            if not self.script_id:
                logger.error("No script ID available")
                return False

            response = requests.get(f"{self.base_url}/api/scripts/{self.script_id}")
            assert response.status_code == 200
            script = response.json()
            assert script["id"] == self.script_id
            logger.info("✅ Get script details test passed")
            return True
        except Exception as e:
            logger.error(f"❌ Get script details test failed: {str(e)}")
            return False

    def test_get_characters(self):
        """Test get characters endpoint"""
        try:
            if not self.script_id:
                logger.error("No script ID available")
                return False

            response = requests.get(f"{self.base_url}/api/scripts/{self.script_id}/characters")
            assert response.status_code == 200
            characters = response.json()
            assert isinstance(characters, list)
            if len(characters) > 0:
                self.character_id = characters[0]["id"]
            logger.info("✅ Get characters test passed")
            return True
        except Exception as e:
            logger.error(f"❌ Get characters test failed: {str(e)}")
            return False

    def test_get_scenes(self):
        """Test get scenes endpoint"""
        try:
            if not self.script_id:
                logger.error("No script ID available")
                return False

            response = requests.get(f"{self.base_url}/api/scripts/{self.script_id}/scenes")
            assert response.status_code == 200
            scenes = response.json()
            assert isinstance(scenes, list)
            if len(scenes) > 0:
                self.scene_id = scenes[0]["id"]
            logger.info("✅ Get scenes test passed")
            return True
        except Exception as e:
            logger.error(f"❌ Get scenes test failed: {str(e)}")
            return False

    def test_get_shots(self):
        """Test get shots endpoint"""
        try:
            if not self.script_id:
                logger.error("No script ID available")
                return False

            response = requests.get(f"{self.base_url}/api/scripts/{self.script_id}/shots")
            assert response.status_code == 200
            shots = response.json()
            assert isinstance(shots, list)
            if len(shots) > 0:
                self.shot_id = shots[0]["id"]
            logger.info("✅ Get shots test passed")
            return True
        except Exception as e:
            logger.error(f"❌ Get shots test failed: {str(e)}")
            return False

    def test_generate_prompts(self):
        """Test prompt generation endpoints"""
        try:
            # Test character prompt generation
            if self.character_id:
                response = requests.post(f"{self.base_url}/api/characters/{self.character_id}/generate-prompt")
                assert response.status_code == 200
                assert "image_prompt" in response.json()
                logger.info("✅ Character prompt generation test passed")

            # Test scene prompt generation
            if self.scene_id:
                response = requests.post(f"{self.base_url}/api/scenes/{self.scene_id}/generate-prompt")
                assert response.status_code == 200
                assert "image_prompt" in response.json()
                logger.info("✅ Scene prompt generation test passed")

            # Test shot prompt generation
            if self.shot_id:
                response = requests.post(f"{self.base_url}/api/shots/{self.shot_id}/generate-prompt")
                assert response.status_code == 200
                assert "image_prompt" in response.json()
                logger.info("✅ Shot prompt generation test passed")

            return True
        except Exception as e:
            logger.error(f"❌ Prompt generation tests failed: {str(e)}")
            return False

    def run_all_tests(self):
        """Run all tests in sequence"""
        logger.info("Starting API tests...")
        
        tests = [
            self.test_api_health,
            self.test_script_upload,
            self.test_get_scripts,
            self.test_get_script_details,
            self.test_get_characters,
            self.test_get_scenes,
            self.test_get_shots,
            self.test_generate_prompts
        ]
        
        results = []
        for test in tests:
            results.append(test())
            
        success_rate = (sum(1 for r in results if r) / len(results)) * 100
        logger.info(f"\nTest Results: {success_rate}% tests passed")
        return success_rate == 100.0

if __name__ == "__main__":
    tester = TestScriptBreakdownAPI()
    tester.run_all_tests()
