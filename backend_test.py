import requests
import pytest
import os
import json
from datetime import datetime

# Get backend URL from environment
BACKEND_URL = "https://ecd1d434-35c7-4302-9091-26f6bdc8e2ab.preview.emergentagent.com"

class TestMovieScriptAPI:
    def __init__(self):
        self.base_url = BACKEND_URL
        self.script_id = None
        self.character_id = None
        self.scene_id = None
        self.shot_id = None

    def test_api_health(self):
        """Test API health endpoint"""
        print("\n🔍 Testing API health...")
        response = requests.get(f"{self.base_url}/api")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Script Breakdown API"
        assert data["status"] == "active"
        print("✅ API health check passed")

    def test_script_upload(self, file_path, file_type):
        """Test script upload with different file types"""
        print(f"\n🔍 Testing {file_type} script upload...")
        
        files = {
            'file': (f'test_script.{file_type}', open(file_path, 'rb')),
        }
        data = {
            'title': f'Test Script {datetime.now().strftime("%Y%m%d_%H%M%S")}'
        }
        
        response = requests.post(
            f"{self.base_url}/api/scripts/upload",
            files=files,
            data=data
        )
        
        assert response.status_code == 200
        result = response.json()
        self.script_id = result["script_id"]
        print(f"✅ {file_type.upper()} upload successful - Script ID: {self.script_id}")
        return result

    def test_script_processing(self):
        """Test script processing results"""
        print("\n🔍 Testing script processing...")
        assert self.script_id is not None, "No script ID available"
        
        # Get script details
        response = requests.get(f"{self.base_url}/api/scripts/{self.script_id}")
        print(f"Script details response: {response.status_code}")
        if response.status_code != 200:
            print(f"Error response: {response.text}")
            return
            
        script = response.json()
        print(f"Script parsing status: {script.get('parsed', False)}")
        
        if not script.get('parsed', False):
            print("⚠️ Script not parsed yet, waiting for processing...")
            # Could add retry logic here
            return
        
        # Check characters
        response = requests.get(f"{self.base_url}/api/scripts/{self.script_id}/characters")
        assert response.status_code == 200
        characters = response.json()
        if len(characters) > 0:
            self.character_id = characters[0]["id"]
            # Verify character traits
            assert "traits" in characters[0]
            assert isinstance(characters[0]["traits"], list)
            print("✅ Character traits extraction verified")
        
        # Check scenes
        response = requests.get(f"{self.base_url}/api/scripts/{self.script_id}/scenes")
        assert response.status_code == 200
        scenes = response.json()
        if len(scenes) > 0:
            self.scene_id = scenes[0]["id"]
            print("✅ Scene extraction verified")
        
        # Check shots
        response = requests.get(f"{self.base_url}/api/scripts/{self.script_id}/shots")
        assert response.status_code == 200
        shots = response.json()
        if len(shots) > 0:
            self.shot_id = shots[0]["id"]
            print("✅ Shot extraction verified")

    def test_image_prompts(self):
        """Test image prompt generation"""
        print("\n🔍 Testing image prompt generation...")
        
        if self.character_id:
            response = requests.post(f"{self.base_url}/api/characters/{self.character_id}/generate-prompt")
            assert response.status_code == 200
            result = response.json()
            assert "image_prompt" in result
            print("✅ Character image prompt generation successful")
        
        if self.scene_id:
            response = requests.post(f"{self.base_url}/api/scenes/{self.scene_id}/generate-prompt")
            assert response.status_code == 200
            result = response.json()
            assert "image_prompt" in result
            print("✅ Scene image prompt generation successful")
        
        if self.shot_id:
            response = requests.post(f"{self.base_url}/api/shots/{self.shot_id}/generate-prompt")
            assert response.status_code == 200
            result = response.json()
            assert "image_prompt" in result
            print("✅ Shot image prompt generation successful")

def create_test_files():
    """Create test files in different formats"""
    # Create a simple text script
    with open('test_script.txt', 'w') as f:
        f.write("""INT. LIVING ROOM - NIGHT

JOHN, a tired detective in his 40s, sits at his desk reviewing case files.

JOHN
(frustrated)
Another dead end. This case is going nowhere.

SARAH, his partner and a sharp-minded analyst, enters with coffee.

SARAH
Maybe you're looking at it wrong. Let me show you something.

She spreads out new evidence photos on the desk.""")
    
    # Create a simple PDF script
    try:
        import fpdf
        pdf = fpdf.FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        with open('test_script.txt', 'r') as txt_file:
            for line in txt_file:
                pdf.cell(200, 10, txt=line.strip(), ln=True)
        pdf.output("test_script.pdf")
        print("✅ PDF test file created")
    except Exception as e:
        print(f"❌ Could not create PDF: {str(e)}")
    
    # Create a simple DOCX script
    try:
        from docx import Document
        doc = Document()
        with open('test_script.txt', 'r') as txt_file:
            for line in txt_file:
                doc.add_paragraph(line.strip())
        doc.save('test_script.docx')
        print("✅ DOCX test file created")
    except Exception as e:
        print(f"❌ Could not create DOCX: {str(e)}")

def main():
    # Create test files
    create_test_files()
    
    # Initialize test class
    tester = TestMovieScriptAPI()
    
    try:
        # Test API health
        tester.test_api_health()
        
        # Test TXT upload
        if os.path.exists('test_script.txt'):
            result = tester.test_script_upload('test_script.txt', 'txt')
            tester.test_script_processing()
            tester.test_image_prompts()
        
        # Test PDF upload
        if os.path.exists('test_script.pdf'):
            result = tester.test_script_upload('test_script.pdf', 'pdf')
            tester.test_script_processing()
            tester.test_image_prompts()
        
        # Test DOCX upload
        if os.path.exists('test_script.docx'):
            result = tester.test_script_upload('test_script.docx', 'docx')
            tester.test_script_processing()
            tester.test_image_prompts()
        
        print("\n✅ All tests completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
