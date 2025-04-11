import requests
import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScriptBreakdownAPITester:
    def __init__(self):
        self.base_url = "http://localhost:8001/api"
        self.script_id = None
        self.test_script_path = "/app/test_script.txt"
        self.tests_run = 0
        self.tests_passed = 0

    def run_test(self, name, test_func):
        """Run a single test with logging"""
        self.tests_run += 1
        logger.info(f"\n🔍 Testing: {name}")
        try:
            result = test_func()
            if result:
                self.tests_passed += 1
                logger.info(f"✅ Passed: {name}")
            else:
                logger.error(f"❌ Failed: {name}")
            return result
        except Exception as e:
            logger.error(f"❌ Failed: {name} - Error: {str(e)}")
            return False

    def test_api_health(self):
        """Test API health endpoint"""
        response = requests.get(f"{self.base_url}")
        return response.status_code == 200 and response.json()["status"] == "active"

    def test_upload_script(self):
        """Test script upload endpoint"""
        with open(self.test_script_path, 'rb') as f:
            files = {'file': ('test_script.txt', f)}
            data = {'title': 'Test Script'}
            response = requests.post(f"{self.base_url}/scripts/upload", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                self.script_id = result.get('script_id')
                logger.info(f"Script uploaded with ID: {self.script_id}")
                logger.info(f"Statistics: {result.get('statistics')}")
                return True
            return False

    def test_get_characters(self):
        """Test getting characters endpoint"""
        if not self.script_id:
            logger.error("No script ID available")
            return False
            
        response = requests.get(f"{self.base_url}/scripts/{self.script_id}/characters")
        if response.status_code == 200:
            characters = response.json()
            logger.info(f"Found {len(characters)} characters")
            for char in characters:
                logger.info(f"Character: {char.get('name')}")
            return len(characters) > 0
        return False

    def test_get_scenes(self):
        """Test getting scenes endpoint"""
        if not self.script_id:
            logger.error("No script ID available")
            return False
            
        response = requests.get(f"{self.base_url}/scripts/{self.script_id}/scenes")
        if response.status_code == 200:
            scenes = response.json()
            logger.info(f"Found {len(scenes)} scenes")
            for scene in scenes:
                logger.info(f"Scene: {scene.get('heading')}")
            return len(scenes) > 0
        return False

    def test_character_prompt_generation(self):
        """Test character prompt generation"""
        if not self.script_id:
            logger.error("No script ID available")
            return False
            
        # First get a character ID
        response = requests.get(f"{self.base_url}/scripts/{self.script_id}/characters")
        if response.status_code != 200 or not response.json():
            return False
            
        character_id = response.json()[0]['id']
        response = requests.post(f"{self.base_url}/characters/{character_id}/generate-prompt")
        
        if response.status_code == 200:
            prompt = response.json()
            logger.info(f"Generated prompt for character: {prompt.get('image_prompt')}")
            return True
        return False

def main():
    tester = ScriptBreakdownAPITester()
    
    # Run all tests
    tests = [
        ("API Health", tester.test_api_health),
        ("Script Upload", tester.test_upload_script),
        ("Get Characters", tester.test_get_characters),
        ("Get Scenes", tester.test_get_scenes),
        ("Character Prompt Generation", tester.test_character_prompt_generation)
    ]
    
    for test_name, test_func in tests:
        tester.run_test(test_name, test_func)
    
    # Print summary
    logger.info(f"\n📊 Test Summary:")
    logger.info(f"Total Tests: {tester.tests_run}")
    logger.info(f"Passed: {tester.tests_passed}")
    logger.info(f"Failed: {tester.tests_run - tester.tests_passed}")
    
    return 0 if tester.tests_passed == tester.tests_run else 1

if __name__ == "__main__":
    exit(main())